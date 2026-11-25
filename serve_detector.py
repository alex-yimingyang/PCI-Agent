"""Serve one explicitly selected Ultralytics checkpoint via POST /predict."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from wsgiref.simple_server import make_server

from local_detector import predict_boxes


def application(model):
    from PIL import Image
    def predict(env, start_response):
        status = '200 OK'
        if env['PATH_INFO'] != '/predict':
            status, data = '404 Not Found', {'error':'Unknown endpoint.'}
        elif env['REQUEST_METHOD'] != 'POST':
            status, data = '405 Method Not Allowed', {'error':'Use POST.'}
        elif env.get('CONTENT_TYPE') != 'image/png':
            status, data = '415 Unsupported Media Type', {'error':'Send a PNG tile.'}
        else:
            try:
                length = int(env.get('CONTENT_LENGTH') or '0')
                confidence = float(env.get('HTTP_X_CONFIDENCE', '0.25'))
                if not 0 < confidence < 1 or not 0 < length <= 16*1024*1024:
                    raise ValueError('Invalid confidence or tile size.')
                with Image.open(BytesIO(env['wsgi.input'].read(length))) as im:
                    if im.format != 'PNG' or max(im.size) > 1536:
                        raise ValueError('Expected a PNG tile up to 1536 pixels per side.')
                    rgb = im.convert('RGB')
            except (ValueError, OSError, Image.DecompressionBombError):
                status, data = '400 Bad Request', {'error':'Invalid PNG tile or confidence.'}
            else:
                data = {'detections':predict_boxes(model, rgb, rgb.width, rgb.height, confidence)}
        body = json.dumps(data, allow_nan=False).encode('utf-8')
        start_response(status, [('Content-Type','application/json'), ('Content-Length',str(len(body)))])
        return [body]
    return predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    path = args.weights.expanduser()
    if path.suffix.lower() != '.pt' or not path.is_file():
        parser.error('--weights requires an existing local .pt file')
    from ultralytics import YOLO
    model = YOLO(str(path))
    if model.task not in ('detect', 'segment'):
        parser.error('Expected a detection or segmentation checkpoint.')
    with make_server(args.host, args.port, application(model)) as server:
        print(f'Detector listening at http://{args.host}:{server.server_port}/predict', flush=True)
        server.serve_forever()
