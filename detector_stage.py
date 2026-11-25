"""Run the configured detector and return its image evidence to the MCP vision client."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from audit_record import immutable
import math
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from fastmcp.tools import ToolResult
import server_detector
import image_agents as vision
import pci_workflow as workflow


ROAD_INSTRUCTIONS = (
    "Use the RGB overview and read_carriageway_detail where boundaries are small or ambiguous. "
    "Identify the travel lanes being assessed using edge lines, curbs, pavement edges and their continuity. "
    "Exclude paved or unpaved shoulders, verges, grass, soil, sidewalks, medians and unrelated pavement. "
    "Keep lane markings, shadows, repairs and distressed pavement inside the carriageway. "
    "Call segment_carriageway_and_detect yourself. Trace mixed scenes in cropped SU pixel coordinates; "
    "use holes for enclosed islands and separate regions for disconnected lanes. "
    "Use all_road only with positive visual evidence that every valid pixel belongs to the travel lanes. "
    "Use uncertain if even detail views do not resolve the road/shoulder boundary."
)
Point = Annotated[list[Annotated[float,Field(ge=0,allow_inf_nan=False)]],Field(min_length=2,max_length=2)]
Ring = Annotated[list[Point],Field(min_length=3,max_length=2048)]


class RoadRegion(BaseModel):
    boundary: Ring
    holes: list[Ring] = Field(default_factory=list)


INSTRUCTIONS = (
    'Start with prepare_sample_unit. Assemble adjoining orthophotos for a complete SU. Crop to the SU boundary before inference. '
    'Ask the user for the detector address for this assessment (local .pt path or HTTP(S) inference URL). '
    'Use only the address the user supplies; do not discover or reuse a model automatically. '
    'Ask for physical scale for Quantity and separately for a Sample Unit for Density. '
    + ROAD_INSTRUCTIONS + ' '
    'Inspect the numbered detector overview and call read_detector_candidates for every page. '
    'Each crop pairs original RGB with its detector box. Preserve the detection ID. Review its '
    'label, trace the visible centerline or affected polygon in cropped SU pixels, or record a reason '
    'to dismiss it. Document possible misses. For a large sample also check native RGB windows. '
    'Call measure_image_detected_distresses, inspect the returned geometry using '
    'read_image_detection_crops, then give per-instance RGB severity with evidence to '
    'score_image_detected_pci for road_pci.py scoring. Box dimensions and confidence do not measure severity.'
)


def detect(images, scale, roi, polygon, model_address, confidence, sample_unit=None, bounds=None):
    import image_detection as detection
    address = (model_address or '').strip()
    if not address:
        question = 'Which detector should process this image? Provide a local .pt path or a server inference URL.'
        if scale is None:
            question += ' Is a physical scale available for Quantity?'
        if sample_unit is None:
            question += ' Is there a separate Sample Unit for Density, or should this single image be used?'
        return ToolResult(structured_content=dict(status='ask_user_for_model_address', question=question))
    if scale is None:
        return ToolResult(structured_content=vision.check_scale(None))
    if sample_unit is None:
        return ToolResult(structured_content=dict(status="ask_user_for_sample_unit", question=vision.SAMPLE_UNIT_QUESTION))
    if sample_unit.mode == "single_image" and len(images) != 1:
        raise ValueError("Confirm these images form one SU using mode=mosaic or provide its boundary and area.")
    if sample_unit.mode == "roi" and not (roi or polygon or bounds or sample_unit.size_px or sample_unit.size_m):
        raise ValueError("Supply a boundary for the requested SU.")
    if polygon:
        ring = polygon[:-1] if polygon[0]==polygon[-1] else polygon
        if len(ring)<3 or detection.polygon_area(ring)<=0 or not detection.simple_polygon(ring):
            raise ValueError("The SU pavement polygon must be simple and have positive area.")
    vision.check_scale(scale)
    remote = '://' in address
    weights = None
    if remote:
        server_detector.endpoint(address)
    else:
        weights = Path(address).expanduser().resolve()
        if not weights.is_file() or weights.suffix.lower() != '.pt':
            raise ValueError('Supply an existing local Ultralytics .pt checkpoint or an HTTP(S) inference URL.')
    run = workflow.new_run()
    assembly = vision.worker("assemble",run,dict(images=[im.model_dump() for im in images],
                          bounds=bounds, roi=roi, polygon=polygon, size_px=sample_unit.size_px,
                          size_m=sample_unit.size_m, origin=sample_unit.origin_xy, scale=scale.model_dump()))
    workflow.write_json(run/"assembly.json", assembly)
    if assembly["status"] != "ready":
        return ToolResult(structured_content={**assembly,"run_id":run.name})
    # A declared pavement area calibrates the retained road, which is unknown at this point.
    # Calculate a gross metric area only when there is an independent geometric scale.
    source=Path(assembly['image_path']).resolve()
    stat=source.stat()
    snapshot=dict(path=str(source),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    gross_area=assembly['valid_area_px2'] if sample_unit.area_px2 else None
    if sample_unit.area_m2 and not sample_unit.area_px2 and any(
            (scale.m_per_px,scale.image_width_m,scale.image_height_m,scale.station_map_path)):
        geometry_scale=scale.model_copy(update={'pavement_area_m2':None})
        gross=vision.prepare_sample(run,assembly,geometry_scale,vision.SampleUnit(mode='roi'))
        gross_area=gross['scale']['pavement_area_m2']
    # An image extent smaller than the requested SU cannot grow during segmentation.
    target = sample_unit.area_px2 or sample_unit.area_m2
    boundary = bool(bounds or roi or polygon or sample_unit.size_px or sample_unit.size_m)
    if target and gross_area is None and not boundary:
        return ToolResult(structured_content=dict(status="needs_sample_unit_boundary",run_id=run.name,
            question="Provide the SU pixel boundary or calibration so its coverage can be checked."))
    if target and gross_area is not None and gross_area < target and not math.isclose(gross_area,target,rel_tol=1e-6):
        return ToolResult(structured_content=dict(status="needs_more_images",run_id=run.name,
            observed_area=gross_area, requested_area=target, question="Add adjoining photos to cover the SU."))
    context=dict(assembly=assembly, scale=scale.model_dump(), unit=sample_unit.model_dump(),
                 address=address, confidence=confidence, gross_area=gross_area,
                 original_source=snapshot)
    with open(run/'road-context.json','x',opener=lambda path,flags:os.open(path,flags,0o600)) as stream:
        json.dump(context,stream,allow_nan=False)
    view=vision.worker('road_view',run,dict(image=assembly['image_path']))
    return vision.image_result(dict(status='needs_carriageway_segmentation',run_id=run.name,
        width=assembly['width'],height=assembly['height'],view=view,model_instructions=ROAD_INSTRUCTIONS),[view])


def road_context(run_id):
    if not re.fullmatch('[0-9a-f]{32}',run_id):
        raise ValueError('Expected an SU segmentation run ID.')
    run=workflow.RUNS/run_id
    context=json.loads((run/'road-context.json').read_text())
    source=context['original_source']
    stat=Path(source['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(source['size'],source['mtime_ns']):
        raise ValueError('Assembled RGB changed. Prepare the SU again.')
    return run,context


def road_detail(run_id,bounds):
    run,context=road_context(run_id)
    view=vision.worker('road_view',run,dict(image=context['assembly']['image_path'],bounds=bounds))
    return vision.image_result(dict(run_id=run_id,view=view),[view])


def segment(run_id,extent,regions,evidence):
    import image_detection as detection
    parent,context=road_context(run_id)
    assembly=context['assembly']
    if not evidence.strip():
        raise ValueError('Describe the visible carriageway boundary or uncertainty.')
    if bool(regions) != (extent=='mixed'):
        raise ValueError('Submit positive road regions for mixed scenes only.')
    if extent=='uncertain':
        return ToolResult(structured_content=dict(status='needs_carriageway_boundary',run_id=run_id,
            evidence=evidence,next_step='Inspect detail windows or supply a verified travel-lane boundary.'))
    for region in regions:
        for ring in [region.boundary,*region.holes]:
            if any(x>assembly['width'] or y>assembly['height'] for x,y in ring):
                raise ValueError('Road outline is outside the cropped SU pixel grid.')
            if ring[-1]==ring[0]:
                ring.pop()
            if len(ring)<3 or detection.polygon_area(ring)<=0 or not detection.simple_polygon(ring):
                raise ValueError('Road outlines must be simple polygons with positive area.')
    # Each attempt gets its own mask and downstream inventory; the RGB source is shared read-only.
    run=workflow.new_run()
    outlines=[region.model_dump() for region in regions]
    roi=vision.worker('road_mask',run,dict(image=assembly['image_path'],regions=outlines,extent=extent))
    roi.update(source_run_id=parent.name,extent=extent,evidence=evidence,coverage_px2=assembly['valid_area_px2'],
               excluded_px2=assembly['valid_area_px2']-roi['retained_px2'])
    workflow.write_json(run/'carriageway.json',{**roi,'regions':outlines})
    overlay=[dict(path=roi['overlay_path'],role='Carriageway segmentation: red is excluded from PCI')]
    if not roi['retained_px2']:
        return vision.image_result(dict(status='no_assessable_carriageway',run_id=run.name,carriageway=roi),overlay)
    scale=vision.Scale(**context['scale'])
    unit=vision.SampleUnit(**context['unit'])
    # The SU extent was checked separately; the density denominator is the retained pavement.
    net_unit=vision.SampleUnit(mode='roi',name=unit.name)
    net_scale=scale.model_copy(update={'pavement_area_m2':None}) if scale.station_map_path else scale
    meta=vision.prepare_sample(run,assembly,net_scale,net_unit,road_mask=roi['mask_path'])
    meta['sample_unit'].update(requested_unit=context['unit'],denominator_source='segmented_carriageway')
    target=unit.area_px2 or unit.area_m2
    net_area=meta['pavement_area_px2'] if unit.area_px2 else meta['scale'].get('pavement_area_m2')
    if target and net_area is not None and not any(
            v is not None and math.isclose(v,target,rel_tol=1e-6) for v in [net_area,context['gross_area']]):
        short = net_area < target
        return vision.image_result(dict(status='needs_more_images' if short else 'needs_sample_unit_boundary',
            run_id=run.name,observed_area=net_area,requested_area=target,carriageway=roi,
            question='Verify SU coverage and the outlined carriageway against the specified area.'),overlay)
    meta.update(carriageway=roi,pavement_roi_source='vision_segmented_carriageway',pavement_roi_explicit=True)
    address=context['address']
    remote='://' in address
    weights=None if remote else Path(address).expanduser().resolve(strict=True)
    confidence=context['confidence']
    source = meta['original_source']['path']
    digest = None
    with TemporaryDirectory(dir=run, prefix='tiles-') as temporary:
        folder = Path(temporary)
        tiling = vision.worker('detector_tiles', run, dict(image=source, folder=temporary))
        if remote:
            manifest = server_detector.predict(folder, address, confidence)
        else:
            with weights.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            executable = os.environ.get('PCI_DETECTOR_PYTHON') or os.environ.get('PCI_IMAGE_PYTHON') or sys.executable
            try:
                result = subprocess.run([executable, '-B', str(workflow.ROOT/'local_detector.py'),
                                         temporary, str(weights), str(confidence)],
                                        env={**os.environ, 'YOLO_AUTOINSTALL':'false',
                                             'YOLO_CONFIG_DIR':str(run/'ultralytics')},
                                        capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired as exc:
                raise ValueError('Local inference timed out; no completed inventory.') from exc
            if result.returncode:
                raise ValueError('Local detector failed. Check PCI_DETECTOR_PYTHON and checkpoint compatibility. '
                                 + result.stderr[-1500:])
            manifest = json.loads((folder/'predictions.json').read_text())
            manifest.update(weights=str(weights), weights_sha256=digest)
        if manifest['tile_count'] != tiling['tile_count'] or not tiling['tile_count']:
            raise ValueError('Detector did not finish the pavement window plan.')
    detection.verify_original(meta)
    filtered=vision.worker('road_candidates',run,dict(candidates=manifest['candidates']))
    manifest.update(candidates=filtered['kept'],off_carriageway_candidates=filtered['excluded'])
    immutable(run/'detector.json', manifest)
    overview = vision.worker('detector_evidence', run, dict(overview=True, rows=manifest['candidates'],
                                                           width=meta['width'], height=meta['height']))
    meta.update(detector_manifest=str(run/'detector.json'), detector_weights_sha256=digest,
                detector_source='server' if remote else 'local', model_api_called=remote,
                candidate_count=len(manifest['candidates']), model_instructions=INSTRUCTIONS,
                status='needs_detector_candidate_review', detector_overview=overview[0]['path'])
    workflow.write_json(run/'prepared.json', meta)
    return vision.image_result(meta, overlay+overview)


def read_candidates(run_id, cursor, limit):
    import image_detection as detection
    run, meta = vision.load_run(run_id, 'detect')
    detection.verify_original(meta)
    if not meta.get('detector_manifest'):
        raise ValueError('Use a completed detector run.')
    rows = json.loads((run/'detector.json').read_text())['candidates']
    if cursor < 0 or cursor > len(rows) or (cursor == len(rows) and rows):
        raise ValueError('Cursor outside detector inventory.')
    selected = rows[cursor:cursor+limit]
    images = vision.worker('detector_evidence', run, dict(image=meta['original_source']['path'], rows=selected))
    receipts = run/'candidate_deliveries'
    receipts.mkdir(exist_ok=True)
    for row, im in zip(selected, images, strict=True):
        immutable(receipts/(row['instance_id']+'.json'), dict(row=row, image_sha256=detection.sha(im['path'])))
    end = cursor+len(selected)
    return vision.image_result(dict(run_id=run_id, candidates=selected, images=images, total=len(rows),
                                    next_cursor=end if end<len(rows) else None), images)


def check_inventory(run, annotations, dismissed):
    import image_detection as detection
    rows = json.loads((run/'detector.json').read_text())['candidates']
    candidates = {r['instance_id']: r for r in rows}
    kept = {a.instance_id for a in annotations}
    if kept & dismissed.keys() or kept | dismissed.keys() != candidates.keys():
        raise ValueError('Account for each detector ID exactly once as reviewed geometry or a reasoned dismissal.')
    if any(not reason.strip() for reason in dismissed.values()):
        raise ValueError('Record a dismissal reason for each rejected detector candidate.')
    for key, row in candidates.items():
        receipt = run/'candidate_deliveries'/(key+'.json')
        if not receipt.exists() or json.loads(receipt.read_text()) != dict(row=row, image_sha256=detection.sha(run/(key+'.png'))):
            raise ValueError('Read every detector candidate image before submitting geometry or dismissals.')
    for annotation in annotations:
        a,b,c,d = candidates[annotation.instance_id]['box']
        points = annotation.polygon or [p for line in annotation.polylines for p in line]
        xs, ys = zip(*points)
        if max(xs)<a or min(xs)>c or max(ys)<b or min(ys)>d:
            raise ValueError('Reviewed geometry must intersect its numbered detector location.')
    return candidates
