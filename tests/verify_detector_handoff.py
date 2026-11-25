"""Regression checks for box merging, crop delivery and detector-to-LLM handoff."""
import json
import os
import socket
from http.server import BaseHTTPRequestHandler
from threading import Thread
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import detector_stage
import image_agents as vision
import image_detection as detection
from local_detector import merge_boxes
import pci_workflow as workflow


def expect_error(fragment, fn, *args):
    try:
        fn(*args)
    except ValueError as exc:
        assert fragment in str(exc), exc
    else:
        raise AssertionError(f'Expected failure containing {fragment!r}')


def verify(folder):
    base = dict(class_id=3, label='linear_crack', confidence=0.9, box=[2,2,12,12], view_id='tile_a')
    boxes = merge_boxes([base, {**base, 'view_id':'tile_b', 'confidence':0.7},
                         {**base, 'class_id':4}, {**base, 'box':[11,11,20,20]}])
    assert [r['instance_id'] for r in boxes] == ['det_00001','det_00002','det_00003']
    assert boxes[0]['confidence'] == 0.9
    photo = folder/'road.pgm'
    photo.write_bytes(b'P5\n24 24\n255\n' + bytes([90])*576)
    photo = [vision.SourceImage(path=str(photo))]
    scale = vision.Scale(status='known', m_per_px=0.002)
    single = vision.SampleUnit(mode='single_image')
    checkpoint = folder/'fixture.pt'
    checkpoint.write_bytes(b'fixture marker')
    with patch.dict(os.environ, {'PCI_DETECTOR_WEIGHTS':str(checkpoint)}):
        for selected_scale in (None, scale):
            answer = detector_stage.detect(photo,selected_scale,None,None,None,0.25)
            assert answer.structured_content['status'] == 'ask_user_for_model_address'
        assert not workflow.RUNS.exists()
    assert detector_stage.detect(photo,None,None,None,str(checkpoint),0.25).structured_content['status'] == 'ask_user_for_scale'
    expect_error('checkpoint', detector_stage.detect, photo,scale,None,None,str(folder/'missing.pt'),0.25,single)

    # Stub only inference. Use real tiled image preparation, evidence rendering and measurement.
    from subprocess import CompletedProcess, run as real_run
    def run(command, **opts):
        if str(ROOT/'local_detector.py') not in command:
            return real_run(command, **opts)
        tiles = Path(command[-3])
        plan = json.loads((tiles/'tiles.json').read_text())
        assert len(plan)==1 and Path(plan[0]['path']).exists()
        rows = [{**r, 'view_id':plan[0]['view_id']} for r in boxes[:2]]
        rows.append({**rows[0],'instance_id':'shoulder','box':[30,0,38,8]})
        (tiles/'predictions.json').write_text(json.dumps(dict(tile_count=len(plan),candidates=rows)))
        return CompletedProcess(command,0,stdout='',stderr='')

    other=folder/'adjoining.pgm'
    other.write_bytes(Path(photo[0].path).read_bytes())
    images=[vision.SourceImage(path=photo[0].path,offset_xy=(0,0)),
            vision.SourceImage(path=str(other),offset_xy=(16,0))]
    expected=vision.SampleUnit(mode='provided', area_px2=960)
    with patch('subprocess.run', side_effect=run):
        assert detector_stage.detect(photo,scale,None,None,str(checkpoint),0.25,expected).structured_content['status']=='needs_more_images'
        cropped=vision.SampleUnit(mode='provided',area_px2=300)
        first=detector_stage.detect(photo,scale,None,None,str(checkpoint),0.25,cropped)
        crop_status=detector_stage.segment(first.structured_content['run_id'],'all_road',[],'Fixture road throughout.')
        assert crop_status.structured_content['status']=='needs_sample_unit_boundary'
        scene=detector_stage.detect(images,scale,None,None,str(checkpoint),0.25,expected)
        sid=scene.structured_content['run_id']
        assert scene.structured_content['status']=='needs_carriageway_segmentation'
        detail=detector_stage.road_detail(sid,[5,0,25,16])
        assert detail.structured_content['view']['displayed_size_xy']==[20,16]
        assert detector_stage.segment(sid,'uncertain',[],'Unclear edge').structured_content['status']=='needs_carriageway_boundary'
        assert detector_stage.segment(sid,'no_road',[],'Background only').structured_content['status']=='no_assessable_carriageway'
        region=detector_stage.RoadRegion(boundary=[[0,0],[20,0],[20,24],[0,24]],
                                        holes=[[[14,12],[18,12],[18,16],[14,16]]])
        prepared=detector_stage.segment(sid,'mixed',[region],'Left carriageway; shoulder to right; internal island.')
    meta = prepared.structured_content
    assert meta['pavement_area_px2']==464 and meta['sample_unit']['area_px2']==464
    assert meta['carriageway']['excluded_px2']==496
    assert meta['candidate_count']==2 and any(c.type=='image' for c in prepared.content)
    # Exercise the HTTP exchange through paired sockets without uploading a tile to an external host.
    class Fixture(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == '/predict' and self.headers['Content-Type'] == 'image/png'
            assert self.headers['X-Confidence'] == '0.25'
            png = self.rfile.read(int(self.headers['Content-Length']))
            assert png.startswith(bytes.fromhex('89504e470d0a1a0a'))
            body = json.dumps({'detections': boxes[:2]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass

    client_socket, service_socket = socket.socketpair()
    def serve():
        with service_socket:
            Fixture(service_socket, ('test-client',0), None)
    thread = Thread(target=serve)
    thread.start()
    def connect(connection):
        connection.sock = client_socket
    with client_socket, patch('http.client.HTTPConnection.connect', connect):
        remote_scene=detector_stage.detect(photo,scale,None,None,'http://test.invalid/predict',0.25,single)
        remote=detector_stage.segment(remote_scene.structured_content['run_id'],'all_road',[],'Fixture full road.')
    thread.join()
    rm = remote.structured_content
    assert rm['detector_source']=='server' and rm['model_api_called'] and rm['detector_weights_sha256'] is None
    assert rm['candidate_count']==meta['candidate_count']
    assert detector_stage.read_candidates(rm['run_id'],0,1).structured_content['next_cursor']==1
    rid = meta['run_id']
    trace = detection.ImageDetection(instance_id='det_00001', kind='linear_crack',
        polylines=[[[3,3],[28,3]]], class_rationale='Fixture class', evidence='Fixture trace',
        geometry_limitations='Synthetic centerline.')
    dismissed = {'det_00002':'Overlapping fixture candidate, reviewed as a false positive.'}
    expect_error('Read every detector', detection.measure_detections, rid,[trace],False,None,dismissed)
    first = detector_stage.read_candidates(rid,0,1)
    assert first.structured_content['next_cursor']==1 and any(c.type=='image' for c in first.content)
    expect_error('Read every detector', detection.measure_detections, rid,[trace],False,None,dismissed)
    final = detector_stage.read_candidates(rid,1,1)
    assert final.structured_content['next_cursor'] is None
    expect_error('Account for each', detection.measure_detections, rid,[trace])
    expect_error('intersect', detection.measure_detections, rid,
                 [trace.model_copy(update={'polylines':[[[18,18],[23,23]]]})],False,None,dismissed)
    result = detection.measure_detections(rid,[trace],False,None,dismissed)
    assert result['measurements'][0]['detector_candidate']['instance_id']==trace.instance_id
    assert result['measurements'][0]['length_px']==16
    assert abs(result['engine_area_m2']-464*.002**2)<1e-12
    assert result['dismissed_candidates']==dismissed and result['crop_count']==1
    detection.read_crops(result['run_id'])
    grade=detection.DetectionGrade(mask_id='det_00001',severity='L',evidence='Fixture',
        geometry_review_evidence='Fixture trace',severity_basis='rgb_crop',severity_rationale='Fixture intact edges',
        excluded_grade_evidence={'M':'Fixture','H':'Fixture'})
    scored=detection.score_detections(result['run_id'],[grade],'Reviewed complete synthetic inventory.')
    assert scored['pci_interval'] and scored['assembly']['valid_area_px2']==960


if __name__ == '__main__':
    with TemporaryDirectory(prefix='pci detector handoff ') as tmp:
        with patch.object(workflow, 'RUNS', Path(tmp)/'runs'):
            verify(Path(tmp))
    print('Detector handoff regression passed.')
