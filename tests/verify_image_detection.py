"""Synthetic annotation acceptance tests via real MCP and unmodified road_pci.py.

These fixtures test tools/geometry/accounting; they are not model detection accuracy results.
"""
import asyncio
from contextlib import asynccontextmanager
import csv
from itertools import product
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.pci_core import load_curves,score_inventory


@asynccontextmanager
async def connect(runs):
    params=StdioServerParameters(command='/bin/sh',args=[str(ROOT/'launch-mcp.sh'),'--agent','detect'],
        env={'PCI_AGENT_PYTHON':sys.executable,'PCI_IMAGE_PYTHON':os.environ.get('PCI_IMAGE_PYTHON',sys.executable),'PCI_AGENT_RUNS':str(runs)})
    async with stdio_client(params) as (reader,writer):
        async with ClientSession(reader,writer,read_timeout_seconds=180) as client:
            await client.initialize()
            names={t.name for t in (await client.list_tools()).tools}
            assert names=={'prepare_image_detection_agent','measure_image_detected_distresses','read_image_detection_crops',
                           'score_image_detected_pci','read_pavement_windows','save_pavement_window_reviews'}
            yield client


async def call(client,name,args,error=None):
    r=await client.call_tool(name,args)
    if error:
        assert r.is_error and error in r.content[0].text,r
        return
    assert not r.is_error and r.structured_content,r
    return r.structured_content


def annotation(i,kind='linear_crack',y=1):
    d=dict(instance_id=i,kind=kind,class_rationale='Synthetic RGB-annotation class fixture.',
           evidence='Synthetic geometry fixture, not a real model observation.',geometry_limitations='Synthetic exact geometry; real model boundaries would be uncertain.')
    if kind=='linear_crack':d['polylines']=[[[1,y],[10,y]]]
    else:d['polygon']=[[4,4],[7,4],[7,7],[4,7]]
    return d


def grade(i,lo='L',hi='M'):
    return dict(mask_id=i,severity_lower=lo,severity_upper=hi,evidence='Synthetic fixture grade.',
        severity_basis='rgb_crop',severity_rationale='Synthetic per-instance observed condition fixture.',
        geometry_review_evidence='Synthetic RGB/model-geometry crop matches expected fixture boundary.',
        excluded_grade_evidence={g:'Synthetic fixture excludes this grade.' for g in ('L','M','H') if g not in ('L','M','H')[('L','M','H').index(lo):('L','M','H').index(hi)+1]})


async def crops(client,run):
    cursor=0
    while cursor is not None:
        r=await call(client,'read_image_detection_crops',dict(measurement_run_id=run,cursor=cursor,limit=4));cursor=r['next_cursor']


async def verify(folder):
    photo=folder/'source.pgm';photo.write_bytes(b'P5\n12 12\n255\n'+bytes([110]*144))
    runs=folder/'runs'
    async with connect(runs) as c:
        ask=await call(c,'prepare_image_detection_agent',dict(image_path=str(photo)))
        assert ask['status']=='ask_user_for_scale' and not runs.exists()
        ready=await call(c,'prepare_image_detection_agent',dict(image_path=str(photo),physical_scale=dict(status='known',m_per_px=.1)))
        assert ready['mode']=='detect' and not ready['masks'] and 'road_pci.py' in ready['model_instructions']
        rp=await call(c,'prepare_image_detection_agent',dict(image_path=str(photo),physical_scale=dict(status='known',m_per_px=.1),pavement_polygon=[[0,0],[6,0],[6,12],[0,12]]))
        assert rp['pavement_area_px2']==72 and math.isclose(rp['scale']['pavement_area_m2'],.72)
        assert rp['pavement_roi_explicit'] and rp['pavement_roi_source']=='conversation_model_original_RGB_pavement_polygon'
        rm=await call(c,'measure_image_detected_distresses',dict(run_id=rp['run_id'],detections=[annotation('roi_clipped_line')]))
        assert rm['measurements'][0]['length_px']==4 and math.isclose(rm['measurements'][0]['measured_length_m'],.4)
        ds=[annotation('line'),annotation('patch','patch')]
        measured=await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=ds))
        rows=measured['measurements'];assert rows[0]['length_px']==9 and math.isclose(rows[0]['measured_length_m'],.9)
        assert rows[1]['measured_area_px2']==9 and math.isclose(rows[1]['measured_area_m2'],.09),rows
        args=dict(measurement_run_id=measured['run_id'],judgments=[grade('line'),grade('patch','M','H')],inventory_review_evidence='Synthetic full fixture inventory checked.')
        await call(c,'score_image_detected_pci',args,error='Inspect every RGB/model-geometry crop')
        await crops(c,measured['run_id'])
        scored=await call(c,'score_image_detected_pci',args)
        assert scored['pci_interval'] and scored['pci_proxy_interval'] is None
        assert scored['search']['method']=='exhaustive_road_pci_cli' and scored['search']['assignments_evaluated']==4
        paths=scored['audit_files'];cli=json.loads(Path(paths['pci_results.json']).read_text())
        assert cli['version']=='road-pci-1.0' and len(cli['samples'])==4
        assignments=json.loads(Path(paths['severity_assignments.json']).read_text());assert any(a['line']=='L' and a['patch']=='H' for a in assignments.values())
        expected=[];curves=load_curves(ROOT/'engine/curves/faa_road_ac_tables.json')
        for a,b in product(('L','M'),('M','H')):
            expected.append(score_inventory([dict(distress='linear_crack',unit='m',quantity=.9,severity=a),dict(distress='patch',unit='m2',quantity=.09,severity=b)],1.44,curves)['pci'])
        assert math.isclose(scored['pci_interval']['lower'],min(expected),abs_tol=1e-10)
        assert math.isclose(scored['pci_interval']['upper'],max(expected),abs_tol=1e-10)
        # Missing individual severity cannot be silently expanded to L-H.
        incomplete={**args,'judgments':[dict(mask_id='line',evidence='Synthetic missing grade',geometry_review_evidence='Synthetic crop reviewed'),grade('patch','M','H')]}
        block=await call(c,'score_image_detected_pci',incomplete);assert block['status']=='needs_instance_or_geometry_review' and block['pci_interval'] is None
        await call(c,'score_image_detected_pci',{**args,'judgments':[grade('line','L','H'),grade('patch','M','H')]},error='not an evidence-narrowed')
        await call(c,'score_image_detected_pci',{**args,'judgments':[{**grade('line'),'kind':'alligator'},grade('patch','M','H')]},error='Class changed')
        await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=[{**annotation('bad','patch'),'polygon':[[0,0],[5,5],[0,5],[5,0]]}]),error='polygon must be simple')
        await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=[{**annotation('bad','patch'),'polygon':[],'box':[0,0,10,10]}]),error='Extra inputs')
        await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=[{**annotation('outside'),'polylines':[[[0,0],[13,0]]]}]),error='coordinates must be within')
        # No scale is a pixel proxy, not physical PCI.
        proxy=await call(c,'prepare_image_detection_agent',dict(image_path=str(photo),physical_scale=dict(status='unavailable')))
        pm=await call(c,'measure_image_detected_distresses',dict(run_id=proxy['run_id'],detections=[annotation('line')]))
        assert pm['measurements'][0]['measured_length_m'] is None and pm['measurements'][0]['quantity_px_basis']==9
        await crops(c,pm['run_id']);ps=await call(c,'score_image_detected_pci',dict(measurement_run_id=pm['run_id'],judgments=[grade('line')],inventory_review_evidence='Synthetic pixel fixture checked.'))
        assert ps['pci_interval'] is None and ps['pci_proxy_interval']
        # Recognized crack family with provisional subtype cannot masquerade as a full class envelope.
        amb=annotation('candidate');amb.update(kind='crack',preferred_crack_kind='linear_crack',class_candidates=['linear_crack','alligator'])
        am=await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=[amb]));await crops(c,am['run_id'])
        ac=await call(c,'score_image_detected_pci',dict(measurement_run_id=am['run_id'],judgments=[grade('candidate')],inventory_review_evidence='Synthetic candidate fixture checked.'))
        assert ac['status']=='preferred_class_estimate_only' and ac['pci_interval'] is None and ac['preferred_class_pci_interval']
        # Empty detections require explicit absence evidence; then still invoke the real CLI.
        await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=[]),error='Empty detection inventory')
        empty=await call(c,'measure_image_detected_distresses',dict(run_id=ready['run_id'],detections=[],confirmed_no_distress=True,absence_evidence='Synthetic constant pavement fixture checked.'))
        healthy=await call(c,'score_image_detected_pci',dict(measurement_run_id=empty['run_id'],judgments=[],inventory_review_evidence='Synthetic absence fixture checked.'))
        assert healthy['pci_interval']==dict(lower=100,upper=100) and healthy['engine_sha256']
        # Independent intervals beyond exact enumeration: outer envelope plus actual mixed-grade CLI checks.
        large=folder/'many.pgm';large.write_bytes(b'P5\n64 64\n255\n'+bytes([110]*4096))
        p=await call(c,'prepare_image_detection_agent',dict(image_path=str(large),physical_scale=dict(status='known',m_per_px=.1)))
        many=[annotation(f'line_{i}',y=4+i*5) for i in range(9)]
        mm=await call(c,'measure_image_detected_distresses',dict(run_id=p['run_id'],detections=many));await crops(c,mm['run_id'])
        ms=await call(c,'score_image_detected_pci',dict(measurement_run_id=mm['run_id'],judgments=[grade(d['instance_id']) for d in many],inventory_review_evidence='Synthetic nine independent traces checked.'))
        assert ms['search']['method']=='cae_outer_bound' and int(ms['search']['assignment_count'])==512
        assert 3<ms['search']['road_cli_feasible_assignments']<=27
        exact=[score_inventory([dict(distress='linear_crack',unit='m',quantity=.9,severity=g) for g in grades],40.96,curves)['pci'] for grades in product(('L','M'),repeat=9)]
        assert ms['pci_interval']['lower']<=min(exact)+1e-8 and ms['pci_interval']['upper']>=max(exact)-1e-8
        # Native RGB coverage is mandatory for a large source, regardless of model detections.
        wide=folder/'wide.pgm';wide.write_bytes(b'P5\n1700 16\n255\n'+bytes([110]*(1700*16)))
        wp=await call(c,'prepare_image_detection_agent',dict(image_path=str(wide),physical_scale=dict(status='unavailable')))
        await call(c,'measure_image_detected_distresses',dict(run_id=wp['run_id'],detections=[annotation('line')]),error='source_view_ids')
        batch=await call(c,'read_pavement_windows',dict(run_id=wp['run_id'],cursor=0,limit=1));view=batch['windows'][0]['view_id']
        await call(c,'save_pavement_window_reviews',dict(run_id=wp['run_id'],reviews=[dict(view_id=view,evidence='Synthetic RGB coverage fixture; no real detection claim.')]))
        wm=await call(c,'measure_image_detected_distresses',dict(run_id=wp['run_id'],detections=[{**annotation('line'),'source_view_ids':[view]}]));await crops(c,wm['run_id'])
        await call(c,'score_image_detected_pci',dict(measurement_run_id=wm['run_id'],judgments=[grade('line')],inventory_review_evidence='Synthetic incomplete native fixture.'),error='Native-window review incomplete')
    print(json.dumps(dict(fourth_route_tools=6,actual_road_cli_verified=True,metric_geometry_verified=True,pixel_proxy_verified=True,
        independent_grade_assignments_verified=4,outer_bound_checked_against_exhaustive_assignments=512,large_native_coverage_guard=True,
        crop_review_guard=True,unknown_grade_not_filled=True,empty_inventory_guard=True,ambiguous_class_not_full_interval=True,
        model_pavement_ROI_denominator_and_clipping=True,synthetic_fixtures_only=True,model_api_called=False),indent=2))


if __name__=='__main__':
    with TemporaryDirectory(prefix='pci detection synthetic ') as p:asyncio.run(verify(Path(p)))
