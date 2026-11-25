"""Client for an explicitly selected pavement inference endpoint."""
import http.client
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from local_detector import merge_boxes


def endpoint(address):
    url = urlsplit(address)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username is not None or url.fragment:
        raise ValueError('Supply the full HTTP(S) inference endpoint, without credentials or a URL fragment.')
    return url


def parse_boxes(payload, tile, threshold):
    if not isinstance(payload, dict) or not isinstance(payload.get('detections'), list):
        raise ValueError('Detector response must contain a detections array.')
    rows = []
    for obj in payload['detections']:
        if not isinstance(obj, dict):
            raise ValueError('Expected a detection object.')
        box = obj.get('box')
        score, cls, label = (obj.get(k) for k in ('confidence', 'class_id', 'label'))
        if (not isinstance(box, list) or len(box) != 4 or
                not all(type(n) in (int,float) and math.isfinite(n) for n in [*box, score]) or
                not 0 <= score <= 1 or type(cls) is not int or cls < 0 or
                not isinstance(label, str) or not label.strip()):
            raise ValueError('Invalid detector box, class, label or confidence.')
        x0,y0,x1,y1 = box
        if not (0 <= x0 < x1 <= tile['width'] and 0 <= y0 < y1 <= tile['height']):
            raise ValueError('Expected xyxy pixel coordinates on the submitted tile.')
        if score >= threshold:
            rows.append(dict(class_id=cls, label=label, confidence=score, view_id=tile['view_id'],
                             box=[x0+tile['x'], y0+tile['y'], x1+tile['x'], y1+tile['y']]))
    return rows


def predict(folder, address, threshold):
    url = endpoint(address)
    connection = http.client.HTTPSConnection if url.scheme == 'https' else http.client.HTTPConnection
    target = url.path or '/'
    if url.query:
        target += '?' + url.query
    tiles = json.loads((folder/'tiles.json').read_text())
    rows = []
    for tile in tiles:
        client = connection(url.hostname, url.port, timeout=60)
        try:
            headers = {'Content-Type':'image/png', 'Accept':'application/json', 'X-Confidence':str(threshold)}
            client.request('POST', target, body=Path(tile['path']).read_bytes(), headers=headers)
            response = client.getresponse()
            if response.status != 200:
                raise ValueError(f'Detector returned HTTP {response.status} for {tile["view_id"]}.')
            payload = response.read(4*1024*1024+1)
            if len(payload) > 4*1024*1024:
                raise ValueError('Detector response exceeds 4 MiB per tile.')
            rows.extend(parse_boxes(json.loads(payload), tile, threshold))
        except (OSError, http.client.HTTPException) as exc:
            raise ValueError(f'Detector connection failed for {tile["view_id"]}: {exc}') from exc
        finally:
            client.close()
    return dict(backend='http', endpoint=url._replace(query='').geturl(), threshold=threshold,
                cross_window_nms_iou=0.5, tile_count=len(tiles), raw_box_count=len(rows),
                candidates=merge_boxes(rows))
