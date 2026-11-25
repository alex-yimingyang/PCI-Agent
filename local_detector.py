"""Local Ultralytics inference. This worker may use a separate Python environment."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection)


def merge_boxes(rows):
    # Cross-window NMS by class. Keep the confidence and source of the winning box.
    grid = defaultdict(list)
    kept = []
    for row in sorted(rows, key=lambda r: -r['confidence']):
        x0, y0, x1, y1 = row['box']
        cells = [(row['class_id'], x, y)
                 for x in range(int(x0//768), int(x1//768)+1)
                 for y in range(int(y0//768), int(y1//768)+1)]
        neighbors = {i for cell in cells for i in grid[cell]}
        if any(kept[i]['view_id'] != row['view_id'] and iou(kept[i]['box'], row['box']) >= 0.5
               for i in neighbors):
            continue
        for cell in cells:
            grid[cell].append(len(kept))
        kept.append({**row, 'instance_id': f'det_{len(kept)+1:05d}'})
    return kept


def predict_boxes(model, source, width, height, confidence):
    rows = []
    results = model.predict(source=source, conf=confidence, verbose=False, save=False)
    result, = results
    if result.orig_shape != (height, width):
        raise ValueError('Detector returned a different source grid.')
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        score = float(box.conf.item())
        if not all(math.isfinite(n) for n in [*xyxy, score]):
            raise ValueError('Detector returned non-finite coordinates or confidence.')
        x0, y0, x1, y1 = xyxy
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        if x0 >= x1 or y0 >= y1:
            continue
        cls = int(box.cls.item())
        rows.append(dict(class_id=cls, label=str(result.names[cls]), confidence=score,
                         box=[x0, y0, x1, y1]))
    return rows


def predict(folder, weights, confidence):
    import ultralytics
    model = ultralytics.YOLO(str(weights))
    if model.task not in ('detect', 'segment'):
        raise ValueError('Use a pavement detection or instance-segmentation checkpoint.')
    tiles = json.loads((folder / 'tiles.json').read_text())
    rows = []
    for tile in tiles:
        for row in predict_boxes(model, tile['path'], tile['width'], tile['height'], confidence):
            x0,y0,x1,y1 = row['box']
            rows.append({**row, 'view_id':tile['view_id'], 'box':[x0+tile['x'], y0+tile['y'],
                                                                  x1+tile['x'], y1+tile['y']]})
    return dict(backend='ultralytics', version=ultralytics.__version__, task=model.task,
                threshold=confidence, cross_window_nms_iou=0.5, tile_count=len(tiles),
                raw_box_count=len(rows), candidates=merge_boxes(rows))


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    data = predict(directory, Path(sys.argv[2]), float(sys.argv[3]))
    (directory / 'predictions.json').write_text(json.dumps(data, allow_nan=False))
