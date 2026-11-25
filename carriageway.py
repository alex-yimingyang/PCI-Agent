"""Render RGB evidence and rasterize a vision-model carriageway delineation on the SU grid."""
import numpy as np
from PIL import Image
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from affine import Affine
import native_windows as native


def view(run, payload):
    with rasterio.open(payload['image']) as src:
        x,y,r,b=payload.get('bounds') or (0,0,src.width,src.height)
        if not (0<=x<r<=src.width and 0<=y<b<=src.height):
            raise ValueError('View must be inside the cropped SU.')
        w,h=r-x,b-y
        factor=min(1.,1600/max(w,h))
        shape=(max(1,round(h*factor)),max(1,round(w*factor)))
        rgb=native.rgb(src,(x,y,w,h),shape)
        valid=src.dataset_mask(window=Window(x,y,w,h),out_shape=shape)>0
        rgb[~valid]=190
        name=run/f'carriageway-{x}-{y}-{r}-{b}.jpg'
        native.save_image(Image.fromarray(rgb),name)
    return dict(path=str(name),role='Original RGB for carriageway segmentation',bounds_xyxy=[x,y,r,b],
                displayed_size_xy=[shape[1],shape[0]])


def mask(run, payload):
    def burn(rings,shape,transform):
        shapes=(({'type':'Polygon','coordinates':[ring]},1) for ring in rings)
        return rasterize(shapes,out_shape=shape,transform=transform,dtype='uint8')
    path=run/'carriageway.tif'
    retained=0
    with rasterio.open(payload['image']) as src:
        profile=native.profile(src.width,src.height)
        profile.update(transform=src.transform,crs=src.crs,BIGTIFF='IF_SAFER')
        with rasterio.open(path,'w',**profile) as dst:
            for box in native.rects(src.width,src.height):
                x,y,w,h=box
                if payload['extent']=='mixed':
                    region=np.zeros((h,w),dtype='uint8')
                    transform=Affine.translation(x,y)
                    for polygon in payload['regions']:
                        lane=burn([polygon['boundary']],(h,w),transform)
                        if polygon['holes'] and lane.any():
                            lane[burn(polygon['holes'],(h,w),transform)>0]=0
                        region |= lane
                else:
                    region=np.full((h,w),payload['extent']=='all_road',np.uint8)
                region &= native.alpha(src,box)
                retained+=int(region.sum())
                dst.write(region,1,window=Window(*box))
        shape=(max(1,round(src.height*min(1,1600/max(src.shape)))),
               max(1,round(src.width*min(1,1600/max(src.shape)))))
        box=(0,0,src.width,src.height)
        rgb=native.rgb(src,box,shape)
        with rasterio.open(path) as ds:
            inside=native.preview_binary(ds,box,shape)
        rgb[~inside]=(rgb[~inside].astype(float)*.35 + np.array([255,56,40])*.65).astype(np.uint8)
        preview=run/'carriageway-overlay.jpg'
        native.save_image(Image.fromarray(rgb),preview)
    return dict(mask_path=str(path), overlay_path=str(preview), retained_px2=retained)


def filter_candidates(run,payload):
    kept=[]; excluded=[]
    with rasterio.open(run/'roi.tif') as src:
        for row in payload['candidates']:
            a,b,c,d=row['box']
            left,top=max(0,int(np.floor(a))),max(0,int(np.floor(b)))
            right,bottom=min(src.width,int(np.ceil(c))),min(src.height,int(np.ceil(d)))
            if right>left and bottom>top and src.read(1,window=Window(left,top,right-left,bottom-top)).any():
                kept.append(row)
            else:
                excluded.append(row)
    return dict(kept=kept,excluded=excluded)
