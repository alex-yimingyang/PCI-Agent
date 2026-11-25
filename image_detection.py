"""Measure, grade and score reviewed detections on the assembled sample-unit grid."""
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import random
from typing import Annotated, Literal
from pydantic import BaseModel, Field, model_validator
import detector_stage
import image_agents as vision
import pci_workflow as workflow

Coordinate=Annotated[float,Field(ge=0,allow_inf_nan=False)]
Point=Annotated[list[Coordinate],Field(min_length=2,max_length=2)]
GRADES=vision.GRADES
EXACT_ASSIGNMENTS=256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon_area(points):
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(points,points[1:]+points[:1])))/2


def simple_polygon(points):
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):return abs(cross(a,b,c))<1e-9 and min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
    def intersects(a,b,c,d):
        return (cross(a,b,c)*cross(a,b,d)<0 and cross(c,d,a)*cross(c,d,b)<0) or any((on(a,b,c),on(a,b,d),on(c,d,a),on(c,d,b)))
    n=len(points)
    for i in range(n):
        for j in range(i+1,n):
            if j==i+1 or (i==0 and j==n-1):continue
            if intersects(points[i],points[(i+1)%n],points[j],points[(j+1)%n]):return False
    return True


class ImageDetection(BaseModel):
    model_config={'extra':'forbid'}
    instance_id: Annotated[str,Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')]
    kind: vision.Kind | Literal['crack']
    preferred_crack_kind: vision.CrackKind | None=None
    class_candidates: list[vision.CrackKind]=Field(default_factory=list)
    class_rationale: Annotated[str,Field(min_length=1,max_length=1800)]
    polylines: list[Annotated[list[Point],Field(min_length=2,max_length=512)]]=Field(default_factory=list,
        description='Linear-crack centerlines in cropped SU pixels; no bounding boxes.')
    polygon: list[Point]=Field(default_factory=list,max_length=512,
        description='Actual affected-region boundary, not its localization bounding box; required for area distress.')
    evidence: Annotated[str,Field(min_length=1,max_length=1800)]
    geometry_limitations: Annotated[str,Field(min_length=1,max_length=1800)]
    source_view_ids: list[str]=Field(default_factory=list,
        description='Inspected RGB native windows supporting this detection; small images may use the overview.')

    @model_validator(mode='after')
    def validate_geometry(self):
        kind=self.kind
        if kind=='crack':
            if not self.class_candidates or self.preferred_crack_kind not in self.class_candidates or len(set(self.class_candidates))!=len(self.class_candidates):
                raise ValueError('Crack candidates require distinct per-instance candidates including a preferred subtype.')
            kind=self.preferred_crack_kind
        elif self.class_candidates or self.preferred_crack_kind:
            raise ValueError('Use kind=crack to record uncertain crack subtypes.')
        if kind=='linear_crack':
            if not self.polylines or self.polygon:raise ValueError('Linear cracks require centerline polylines only.')
            if any(all(p==line[0] for p in line) for line in self.polylines):raise ValueError('A centerline must have positive length.')
        else:
            if self.polylines or len(self.polygon)<3:raise ValueError('Area distress requires an affected-region polygon, not a bounding box or a length.')
            if self.polygon[0]==self.polygon[-1]:self.polygon=self.polygon[:-1]
            if len(set(map(tuple,self.polygon)))!=len(self.polygon) or polygon_area(self.polygon)<=0 or not simple_polygon(self.polygon):
                raise ValueError('Affected-region polygon must be simple, nondegenerate, and without repeated vertices.')
        return self


class DetectionGrade(vision.Judgment):
    geometry_review_evidence: Annotated[str,Field(min_length=1,max_length=1800)]


def verify_original(meta):
    for key in ('original_source','roi_snapshot'):
        item=meta[key];p=Path(item['path']);s=p.stat()
        if dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)!=item:
            raise workflow.WorkflowError('Original RGB or pavement ROI changed; prepare the detection route again.')


def measure_detections(run_id,detections,confirmed_no_distress=False,absence_evidence=None,dismissed=None):
    parent,meta=vision.load_run(run_id,'detect');verify_original(meta)
    if len({d.instance_id for d in detections})!=len(detections):raise workflow.WorkflowError('Detection instance IDs must be unique.')
    if not detections and not (confirmed_no_distress and absence_evidence and absence_evidence.strip()):
        raise workflow.WorkflowError('Empty detection inventory requires explicit no-distress confirmation and absence evidence; it does not imply PCI 100.')
    if detections and confirmed_no_distress:raise workflow.WorkflowError('No-distress confirmation conflicts with detected instances.')
    if meta.get('status') != 'needs_detector_candidate_review' or not meta.get('detector_manifest'):
        raise ValueError('A completed detector inventory for this SU is required.')
    dismissed = dismissed or {}
    candidates = detector_stage.check_inventory(parent, detections, dismissed)
    plan=vision.window_plan(parent)
    for d in detections:
        points=[p for line in d.polylines for p in line] or d.polygon
        a,b=min(p[0] for p in points),min(p[1] for p in points);r,t=max(p[0] for p in points),max(p[1] for p in points)
        for view in d.source_view_ids:
            if view not in plan or not (parent/'reviews'/f'{view}.json').is_file():
                raise workflow.WorkflowError('Detection evidence must refer to inspected and reviewed original RGB windows.')
            v=plan[view]
            if not (a<v['x']+v['width'] and r>=v['x'] and b<v['y']+v['height'] and t>=v['y']):
                raise workflow.WorkflowError('Cited RGB source window does not intersect this instance geometry.')
    scale=meta['scale'];metric=scale['status']=='known'
    scoring_scale=scale['m_per_px'] if metric else scale['assumed_m_per_px_for_pci_proxy']
    sample_unit=vision.sample_unit_for_run(meta)
    area=sample_unit['engine_area_m2']
    run=workflow.new_run();annotations=[d.model_dump() for d in detections]
    workflow.write_json(run/'detections.json',annotations)
    measured=vision.worker('detection_geometry',run,dict(original_meta=meta,detections=annotations,
        scoring_scale=scoring_scale,station_map=scale.get('station_map_path')))
    from engine.pci_core import density
    for row,d in zip(measured['rows'],annotations,strict=True):
        kind=row['kind'];unit='m' if kind=='linear_crack' else 'count' if kind=='pothole' else 'm2'
        px=row['length_px'] if unit=='m' else row['count'] if unit=='count' else row['measured_area_px2']
        q=px*(scoring_scale if unit=='m' else scoring_scale**2 if unit=='m2' else 1)
        if metric and scale.get('station_map_path'):q=row['mapped_length_m'] if unit=='m' else row['count'] if unit=='count' else row['mapped_area_m2']
        row.update(instance_id=d['instance_id'],
                   quantity_px_basis=px,quantity_px_unit='px' if unit=='m' else 'px2' if unit=='m2' else 'count',
                   density_px_basis=100*px/sample_unit['area_px2'] if sample_unit['area_px2'] is not None else None,density_px_unit='100 px/px2' if unit=='m' else 'percent' if unit=='m2' else 'count/100 px2',
                   quantity_for_engine=q,unit_for_engine=unit,density_for_engine=density(q,unit,area) if area is not None else None,
                   measured_length_m=(row['mapped_length_m'] if scale.get('station_map_path') else row['length_px']*scoring_scale) if metric else None,
                   measured_area_m2=(row['mapped_area_m2'] if scale.get('station_map_path') else row['measured_area_px2']*scoring_scale**2) if metric else None,
                   geometry_limitations=d['geometry_limitations'])
        row.update(vision.reported_measurement(px,q,row['quantity_px_unit'],metric,sample_unit))
        row.update(detector_candidate=candidates[d['instance_id']], class_source='LLM_review_of_detector_label',
                       geometry_source='LLM_trace_on_detector_candidate')
    for crop in measured['crops']:crop['sha256']=sha(crop['path'])
    result=dict(run_id=run.name,prepared_run_id=run_id,workflow='sample_unit_detector_pci',status='needs_geometry_and_instance_severity_review',
        scale=scale,assembly=meta['assembly'],carriageway=meta.get('carriageway'),sample_unit=sample_unit,pavement_area_px2=meta['pavement_area_px2'],engine_area_m2=area,assumed_full_canvas=meta['assumed_full_canvas'],
        measurements=measured['rows'],crops=measured['crops'],annotations=annotations,annotations_sha256=sha(run/'detections.json'),
        cross_type_overlap_px2=measured['cross_type_overlap_px2'],unassigned_pixels=measured['unassigned_pixels'],
        confirmed_no_distress=confirmed_no_distress,absence_evidence=absence_evidence,model_api_called=False,
        pavement_roi_source=meta.get('pavement_roi_source'),pavement_roi_explicit=meta.get('pavement_roi_explicit'),
        review_scheme='Original RGB native-window reviews plus explicit per-instance RGB/model-geometry crop review; numerical mask preparation is not a second visual review.')
    result.update(detector_manifest=meta['detector_manifest'], dismissed_candidates=dismissed,
                      detector_weights_sha256=meta['detector_weights_sha256'], detector_source=meta.get('detector_source','local'),
                      model_api_called=meta.get('model_api_called',False))
    workflow.write_json(run/'detected_measurements.json',result)
    public={k:v for k,v in result.items() if k not in ('annotations','crops')};public['crop_count']=len(result['crops'])
    public['measurements']=result['measurements'][:30];public['measurement_json']=str(run/'detected_measurements.json')
    public['next_step']='Call read_image_detection_crops until every crop is delivered; inspect geometry, then score_image_detected_pci with one independent grade judgment per instance.'
    return public


def load_measurement(run_id):
    if len(run_id)!=32 or any(c not in 'abcdef0123456789' for c in run_id):raise workflow.WorkflowError('Use the measurement run_id returned by this agent.')
    run=workflow.RUNS/run_id;data=json.loads((run/'detected_measurements.json').read_text())
    if sha(run/'detections.json')!=data['annotations_sha256']:raise workflow.WorkflowError('Detection annotations changed; measure again.')
    return run,data


def read_crops(run_id,cursor=0,limit=4):
    run,data=load_measurement(run_id);crops=data['crops']
    if not 0<=cursor<len(crops):raise workflow.WorkflowError('Crop cursor is past the end of the detection inventory.')
    selected=crops[cursor:cursor+limit];delivery=run/'crop_deliveries';delivery.mkdir(exist_ok=True)
    for row in selected:
        if sha(row['path'])!=row['sha256']:raise workflow.WorkflowError('Detection crop changed; measure again.')
        workflow.write_json(delivery/f"{row['instance_id']}.json",row)
    end=cursor+len(selected)
    return vision.image_result(dict(run_id=run_id,crops=selected,total_crops=len(crops),next_cursor=end if end<len(crops) else None),selected)


def assignment_plan(rows):
    options=[tuple(GRADES[GRADES.index(r['severity_lower']):GRADES.index(r['severity_upper'])+1]) for r in rows]
    count=math.prod(map(len,options))
    if count<=EXACT_ASSIGNMENTS:return list(product(*options)),count,True
    rng=random.Random(318);trials=[]
    for target in GRADES:trials.append(tuple(GRADES[max(GRADES.index(o[0]),min(GRADES.index(o[-1]),GRADES.index(target)))] for o in options))
    trials.extend(tuple(rng.choice(o) for o in options) for _ in range(24))
    return list(dict.fromkeys(trials)),count,False


def score_detections(run_id,judgments,inventory_review_evidence):
    measured_run,data=load_measurement(run_id);parent,meta=vision.load_run(data['prepared_run_id'],'detect')
    verify_original(meta);coverage=vision.check_window_coverage(parent,meta)
    if not inventory_review_evidence.strip():raise workflow.WorkflowError('Explain the final detection inventory review, including misses, duplicates and obscured regions.')
    for crop in data['crops']:
        receipt=measured_run/'crop_deliveries'/f"{crop['instance_id']}.json"
        if not receipt.is_file() or json.loads(receipt.read_text())!=crop or sha(crop['path'])!=crop['sha256']:
            raise workflow.WorkflowError('Inspect every RGB/model-geometry crop with read_image_detection_crops before scoring.')
    grades={g.mask_id:g for g in judgments};ids={d['instance_id'] for d in data['annotations']}
    if len(grades)!=len(judgments) or set(grades)!=ids:raise workflow.WorkflowError('Submit exactly one grade judgment for every detected instance, including zero-quantity duplicates.')
    inventory=[];audit=[];missing=[]
    for det,row in zip(data['annotations'],data['measurements'],strict=True):
        g=grades[det['instance_id']];kind=row['kind'];lo=g.severity_lower or g.severity;hi=g.severity_upper or g.severity
        if g.kind not in (None,kind):raise workflow.WorkflowError('Class changed after geometry measurement; submit revised detections and measure again.')
        if data.get('detector_manifest') and (lo, hi) == ('L', 'H'):
            raise ValueError('Provide a single grade or an evidence-supported adjacent range for each detector instance.')
        if lo is None:missing.append(g.mask_id)
        elif g.severity_basis!='rgb_crop' or not g.severity_rationale:raise workflow.WorkflowError('Each severity needs original RGB crop evidence and an individual rationale.')
        elif not g.forced_surface_estimate:
            excluded=set(GRADES)-set(GRADES[GRADES.index(lo):GRADES.index(hi)+1])
            if any(not g.excluded_grade_evidence.get(s,'').strip() for s in excluded):raise workflow.WorkflowError('Explain evidence excluding every grade outside this instance range, or explicitly record a forced surface estimate.')
        audit.append({**row,**g.model_dump(), 'kind':kind,'native_mask_id':row['mask_id'],
                      'instance_id':det['instance_id'],'class_decision':det,
                      'severity_lower':lo,'severity_upper':hi})
        if row['scored'] and lo is not None:inventory.append(dict(record_id=det['instance_id'],distress=kind,unit=row['unit_for_engine'],quantity=row['quantity_for_engine'],severity_lower=lo,severity_upper=hi))
    out=workflow.new_run();workflow.write_json(out/'instance_judgments_and_measurements.json',audit)
    result=dict(run_id=out.name,prepared_run_id=data['prepared_run_id'],measurement_run_id=run_id,workflow='sample_unit_detector_pci',
        model_api_called=False,official_pci=False,statistically_calibrated=False,scale=data['scale'],native_review=coverage,
        total_instances=len(ids),scored_instances=len(inventory),inventory_review_evidence=inventory_review_evidence,
        measurements_json=str(out/'instance_judgments_and_measurements.json'),pci_interval=None,pci_proxy_interval=None,
        interval_scope='Conditional on fixed RGB-model detections, preferred classes and drawn geometry. Search varies each instance severity only; missed detections, geometry boundaries and alternative classes are not covered.',
        overlap_policy='Same-class overlaps owned once in submitted annotation order; area distress suppresses covered linear traces. Cross-type area conflicts block final scoring.',
        assumed_full_canvas=data['assumed_full_canvas'],pavement_roi_source=data.get('pavement_roi_source'),pavement_roi_explicit=data.get('pavement_roi_explicit'))
    result['sample_unit'] = vision.sample_unit_for_run(data)
    result['assembly'] = data.get('assembly')
    result['carriageway'] = data.get('carriageway')
    if data.get('detector_manifest'):
        result.update(detector_manifest=data['detector_manifest'], detector_weights_sha256=data['detector_weights_sha256'],
                      dismissed_candidates=data['dismissed_candidates'], detector_source=data.get('detector_source','local'),
                      model_api_called=data.get('model_api_called',False),
                      interval_scope='Conditional on reviewed detector candidates, class and traced geometry; varies per-instance severity.')
    incomplete=missing or data['unassigned_pixels'] or data['cross_type_overlap_px2'] or any(r['withheld'] for r in data['measurements'])
    if data['engine_area_m2'] is None:
        result.update(status='needs_compatible_sample_unit_area', reason='Supply the SU area in the measurement units or calibrate the image.')
    elif incomplete:
        result.update(status='needs_instance_or_geometry_review',ungraded_instances=missing,cross_type_overlap_px2=data['cross_type_overlap_px2'],withheld_instances=[r['instance_id'] for r in data['measurements'] if r['withheld']])
    elif not inventory and not data['confirmed_no_distress']:
        result.update(status='no_scorable_distress',reason='Detected geometry has no positive owned quantity; empty scoring rows do not establish PCI 100.')
    else:
        assignments,count,exact=assignment_plan(inventory);area=data['engine_area_m2'];samples=[];flat=[];scenario_map={}
        for i,assignment in enumerate(assignments):
            sid=f'assignment_{i:04d}';scenario_map[sid]={r['record_id']:g for r,g in zip(inventory,assignment,strict=True)}
            samples.append(dict(sample_id=sid,area_m2=area,confirmed_no_distress=str(data['confirmed_no_distress']).lower()))
            flat.extend(dict(sample_id=sid,record_id=r['record_id'],distress=r['distress'],severity=g,quantity=r['quantity'],unit=r['unit'],severity_source='conversation_model_original_RGB_per_instance') for r,g in zip(inventory,assignment,strict=True))
        workflow.write_csv(out/'samples.csv',('sample_id','area_m2','confirmed_no_distress'),samples)
        workflow.write_csv(out/'inventory.csv',('sample_id','record_id','distress','severity','quantity','unit','severity_source'),flat)
        workflow.write_json(out/'severity_assignments.json',scenario_map)
        report,files=workflow.score(out,'error','approximate')  # actual unmodified road_pci.py CLI
        points={s['sample_id']:s['scenarios']['M']['pci'] for s in report['samples']}
        if exact:
            low=min(points,key=points.get);high=max(points,key=points.get)
            bounds=dict(lower=points[low],upper=points[high],method='exhaustive_road_pci_cli',converged=True,
                assignment_count=str(count),assignments_evaluated=len(assignments),endpoint_search_gap_pci=0,
                lower_witness=dict(sample_id=low,instance_grades=scenario_map[low]),upper_witness=dict(sample_id=high,instance_grades=scenario_map[high]))
        else:
            bounds=vision.worker('intervals',out,dict(pavement_area_m2=area,inventory=inventory))
            if any(not bounds['lower']-1e-6<=p<=bounds['upper']+1e-6 for p in points.values()):raise workflow.WorkflowError('Interval search disagrees with road_pci.py CLI; no PCI interval accepted.')
            bounds.update(assignment_count=str(count),road_cli_feasible_assignments=len(assignments),road_cli_witness_range=dict(lower=min(points.values()),upper=max(points.values())))
        workflow.write_json(out/'pci_bounds_details.json',bounds)
        interval=dict(lower=bounds['lower'],upper=bounds['upper']);field='pci_interval' if data['scale']['status']=='known' else 'pci_proxy_interval'
        ambiguous=[d['instance_id'] for d in data['annotations'] if d['kind']=='crack']
        result.update(status='preferred_class_estimate_only' if ambiguous else 'provisional',search=bounds,
            ambiguous_class_instance_ids=ambiguous,class_candidates_searched=False,
            engine='road_pci.py CLI: per-assignment fixed instance severities, unknown=error; no blanket L/M/H substitution',
            engine_sha256=sha(workflow.ENGINE),curve_sha256=report['curve_sha256'],audit_files={**files,'severity_assignments.json':str(out/'severity_assignments.json'),'pci_bounds_details.json':str(out/'pci_bounds_details.json')})
        result['preferred_class_'+field if ambiguous else field]=interval
    workflow.write_json(out/'detected_pci.json',result)
    result['result_json']=str(out/'detected_pci.json')
    return result
