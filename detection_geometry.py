"""Rasterize conversational RGB annotations and measure them; no detector/model API."""
from pathlib import Path
import math
import cv2
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from affine import Affine
from PIL import Image, ImageDraw
import native_windows as native


def shape_of(row):
    if row['polylines']:
        return {'type':'MultiLineString', 'coordinates':row['polylines']}
    ring=row['polygon'];rings=[ring+[ring[0]]] if ring[0]!=ring[-1] else [ring]
    return {'type':'Polygon','coordinates':rings}


def detail_boxes(mask,extent,width,height):
    x,y,r,b=extent;factor=min(1.,384/(r-x),384/(b-y));h=max(1,round((b-y)*factor));w=max(1,round((r-x)*factor))
    preview=native.preview_binary(mask,(x,y,r-x,b-y),(h,w));yy,xx=np.nonzero(preview);result=[]
    if not len(xx):
        # A one-pixel centreline can disappear in a nearest-neighbour preview.
        # Select actual foreground pixels instead of inventing a detail location.
        yy,xx=np.nonzero(mask.read(1,window=Window(x,y,r-x,b-y))>0)
        if not len(xx):
            raise ValueError('Detection geometry has no foreground inside the pavement ROI.')
        w,h=r-x,b-y
    for fraction in (.25,.75):
        idx=min(len(xx)-1,int(len(xx)*fraction));a=x+math.floor(xx[idx]*(r-x)/w);t=y+math.floor(yy[idx]*(b-y)/h)
        c=min(r,x+math.ceil((xx[idx]+1)*(r-x)/w));d=min(b,y+math.ceil((yy[idx]+1)*(b-y)/h))
        local=mask.read(1,window=Window(a,t,c-a,d-t))>0;ly,lx=np.nonzero(local)
        if len(lx):cx,cy=a+int(lx[len(lx)//2]),t+int(ly[len(ly)//2])
        else:cx,cy=a,t
        bw,bh=min(384,width),min(384,height);left=max(0,min(width-bw,cx-bw//2));top=max(0,min(height-bh,cy-bh//2))
        result.append([left,top,left+bw,top+bh])
    return result


def draw_geometry(image,row,bounds):
    x,y,r,b=bounds;w,h=image.size;draw=ImageDraw.Draw(image)
    convert=lambda ps:[((a-x)*w/(r-x),(c-y)*h/(b-y)) for a,c in ps]
    if row['polylines']:
        for line in row['polylines']:draw.line(convert(line),fill='#FF4B48',width=2)
    else:draw.line(convert(row['polygon']+[row['polygon'][0]]),fill='#FF4B48',width=2)


def measure(run,payload):
    cv2.setNumThreads(1)
    source=payload['original_meta'];rows=payload['detections']
    native.verify_source(source['original_source']);native.verify_source(source['roi_snapshot'])
    width,height=source['width'],source['height'];specs=[];crops=[]
    for i,row in enumerate(rows,1):
        points=[p for line in row['polylines'] for p in line] or row['polygon']
        inside=(lambda x,y:0<=x<width and 0<=y<height) if row['polylines'] else (lambda x,y:0<=x<=width and 0<=y<=height)
        if any(not inside(x,y) for x,y in points):
            raise ValueError(f"{row['instance_id']}: coordinates must be within the original image pixel grid.")
        x0=max(0,math.floor(min(p[0] for p in points)));y0=max(0,math.floor(min(p[1] for p in points)))
        x1=min(width,math.ceil(max(p[0] for p in points))+1);y1=min(height,math.ceil(max(p[1] for p in points))+1)
        path=run/f'annotation_{i}.tif';geometry=shape_of(row)
        with rasterio.open(path,'w',**native.profile(x1-x0,y1-y0)) as dst:
            for lx,ly,w,h in native.rects(x1-x0,y1-y0):
                data=rasterize([(geometry,1)],out_shape=(h,w),transform=Affine.translation(x0+lx,y0+ly),
                               fill=0,all_touched=False,dtype='uint8')
                if data.any():dst.write(data,1,window=Window(lx,ly,w,h))
        kind=row['preferred_crack_kind'] if row['kind']=='crack' else row['kind']
        specs.append(dict(path=str(path),bounds_xyxy=[x0,y0,x1,y1],kind=kind,
                          semantics='crack_trace' if kind=='linear_crack' else 'affected_region'))
        crops.append([max(0,x0-48),max(0,y0-48),min(width,x1+48),min(height,y1+48)])
    meta=native.prepare(run,dict(image_path=source['original_source']['path'],masks=specs,
                                roi_path=source['roi_snapshot']['path'],station_map=payload.get('station_map')))
    judgments=[dict(mask_id=f'mask_{i}',kind=spec['kind'],semantics=spec['semantics'],box=None,
                    severity_lower=None,severity_upper=None,severity_source='pending_per_instance_RGB_grade',
                    evidence=row['evidence']) for i,(row,spec) in enumerate(zip(rows,specs,strict=True),1)]
    measured=native.measure(run,dict(meta=meta,judgments=judgments,scoring_scale=payload['scoring_scale'],station_map=payload.get('station_map')))
    # These are numerical geometry snapshots, not a second completed native visual review.
    measured['geometry_prepared']=meta;images=[]
    with rasterio.open(source['original_source']['path']) as src:
        for i,(row,bounds) in enumerate(zip(rows,crops,strict=True),1):
            x,y,r,b=bounds;factor=min(1.,384/(r-x),384/(b-y));h=max(1,round((b-y)*factor));w=max(1,round((r-x)*factor))
            raw=Image.fromarray(native.rgb(src,(x,y,r-x,b-y),(h,w)));overlay=raw.copy();draw_geometry(overlay,row,bounds)
            with rasterio.open(run/f'mask_{i}.tif') as mask:details=detail_boxes(mask,meta['masks'][i-1]['bounds_xyxy'],width,height)
            canvas=Image.new('RGB',(768,h+238),'#e9eff3');canvas.paste(raw,(0,22));canvas.paste(overlay,(384,22))
            ImageDraw.Draw(canvas).text((4,4),row['instance_id']+' | '+specs[i-1]['kind']+' | RGB / model geometry',fill='#102b3b')
            for j,box in enumerate(details):
                a,t,c,d=box;im=Image.fromarray(native.rgb(src,(a,t,c-a,d-t)));im.thumbnail((192,192),Image.Resampling.LANCZOS)
                marked=im.copy();draw_geometry(marked,row,box);canvas.paste(im,(j*384,h+46));canvas.paste(marked,(j*384+192,h+46))
                ImageDraw.Draw(canvas).text((j*384+4,h+27),f'Detail {j+1}: {c-a} x {d-t} source px',fill='#102b3b')
            p=run/f'detection_crop_{i}.png';canvas.save(p)
            images.append(dict(instance_id=row['instance_id'],path=str(p),role='RGB_and_model_annotation',bounds_xyxy=bounds,detail_bounds_xyxy=details))
    measured['crops']=images
    return measured
