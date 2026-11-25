"""Windowed RGB input and paired annotated evidence for the local detector."""
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw
import rasterio
from rasterio.windows import Window
import native_windows as native


def tiles(run, payload):
    folder = Path(payload['folder'])
    seen = set()
    result = []
    with rasterio.open(payload['image']) as src, rasterio.open(run/'roi.tif') as roi:
        for line in (run/'windows.jsonl').read_text().splitlines():
            tile = json.loads(line)
            box = tuple(tile[k] for k in ('x', 'y', 'width', 'height'))
            if box in seen:
                continue
            seen.add(box)
            rgb = native.rgb(src, box)
            rgb[roi.read(1, window=Window(*box)) == 0] = 190
            path = folder / (tile['view_id'] + '.png')
            Image.fromarray(rgb).save(path)
            result.append({**tile, 'path': str(path)})
    (folder/'tiles.json').write_text(json.dumps(result))
    return {'tile_count': len(result)}


def outline(image, row, origin, scale):
    a, b, c, d = row['box']
    box = ((a-origin[0])*scale[0], (b-origin[1])*scale[1],
           (c-origin[0])*scale[0], (d-origin[1])*scale[1])
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline='#ffba21', width=2)
    draw.text((box[0]+2, box[1]+1), row['instance_id'], fill='white',
              stroke_width=1, stroke_fill='black')


def evidence(run, payload):
    rows = payload['rows']
    if payload.get('overview'):
        with Image.open(run/'original.jpg') as original:
            im = original.convert('RGB')
        scale = (im.width/payload['width'], im.height/payload['height'])
        for row in rows:
            outline(im, row, (0, 0), scale)
        path = run/'detector_overview.jpg'
        native.save_image(im, path)
        return [dict(path=str(path), role='Detector | numbered whole-image overview')]

    images = []
    with rasterio.open(payload['image']) as src:
        for row in rows:
            a, b, c, d = row['box']
            x, y = max(0, math.floor(a)-32), max(0, math.floor(b)-32)
            w, h = min(src.width, math.ceil(c)+32)-x, min(src.height, math.ceil(d)+32)-y
            f = min(1, 512/max(w,h))
            shape = (max(1, round(h*f)), max(1, round(w*f)))
            raw = Image.fromarray(native.rgb(src, (x,y,w,h), shape))
            marked = raw.copy()
            outline(marked, row, (x,y), (raw.width/w, raw.height/h))
            sheet = Image.new('RGB', (raw.width*2, raw.height+22), 'white')
            sheet.paste(raw, (0,22))
            sheet.paste(marked, (raw.width,22))
            ImageDraw.Draw(sheet).text((2,3), row['instance_id']+' | RGB / detection', fill='black')
            path = run/(row['instance_id']+'.png')
            native.save_image(sheet, path)
            images.append(dict(path=str(path), role=row['instance_id']+' | RGB and detector box',
                               bounds_xyxy=[x,y,x+w,y+h]))
    return images
