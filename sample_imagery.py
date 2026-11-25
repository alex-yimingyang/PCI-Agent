"""Assemble same-resolution orthophotos on one pixel grid, then crop to the requested SU."""
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
import math
import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from rasterio.features import rasterize
import native_windows as native


@dataclass
class Source:
    path: str
    width: int
    height: int
    crs: object
    transform: Affine


def features(ds):
    factor = min(1., 1800/max(ds.width, ds.height))
    h, w = max(1, round(ds.height*factor)), max(1, round(ds.width*factor))
    rgb = native.rgb(ds, (0,0,ds.width,ds.height), (h,w))
    valid = ds.dataset_mask(out_shape=(h,w))
    points, desc = cv2.SIFT_create(nfeatures=3500).detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),valid)
    xy = np.array([p.pt for p in points]).reshape(-1,2) * [ds.width/w,ds.height/h]
    return xy, desc


def translation(left, right):
    a, da = left
    b, db = right
    if min(len(a),len(b)) < 12:
        return None
    matcher = cv2.BFMatcher()
    def pairs(first, second):
        return {(p.queryIdx,p.trainIdx) for p,q in matcher.knnMatch(first,second,k=2)
                if p.distance < .72*q.distance}
    matches = sorted(pairs(da,db) & {(j,i) for i,j in pairs(db,da)})
    if len(matches)<12:
        return None
    i,j = np.array(matches).T
    delta = a[i]-b[j]
    offset = np.median(delta,axis=0)
    good = np.linalg.norm(delta-offset,axis=1) <= 2
    if good.sum()<12 or good.mean()<.75 or min(np.ptp(a[i][good],axis=0))<16:
        return None
    return np.rint(np.median(delta[good],axis=0)).astype(int)


def placements(datasets, inputs):
    offsets = [s.get('offset_xy') for s in inputs]
    if any(v is not None for v in offsets):
        if any(v is None for v in offsets):
            raise ValueError('Supply integer pixel offsets for every source on the same grid.')
        return offsets, 'supplied_offsets'
    if len(inputs)==1:
        return [[0,0]], 'single_source'
    first = datasets[0]
    if any(d.crs is not None for d in datasets):
        offsets = []
        for d in datasets:
            rel = ~first.transform * d.transform
            shift = np.rint([rel.c,rel.f]).astype(int)
            if first.crs is None or d.crs!=first.crs or not np.allclose(
                    tuple(rel)[:6],(1,0,shift[0],0,1,shift[1]),rtol=0,atol=1e-4):
                raise ValueError('Orthorectify georeferenced inputs onto the same pixel grid before assembly.')
            offsets.append(shift.tolist())
        return offsets, 'georeferenced_grid'
    descriptors = []
    for source in datasets:
        with rasterio.open(source.path) as ds:
            descriptors.append(features(ds))
    positions = {0:np.zeros(2,dtype=int)}
    todo = [0]
    while todo:
        i = todo.pop()
        for j in range(len(inputs)):
            if j==i or (j in positions and j<i):
                continue
            delta = translation(descriptors[i], descriptors[j])
            if delta is None:
                continue
            pos = positions[i]+delta
            if j not in positions:
                positions[j]=pos
                todo.append(j)
            elif np.max(np.abs(positions[j]-pos))>2:
                raise ValueError('Conflicting image registrations. Supply surveyed pixel offsets or georeferenced sources.')
    if len(positions)!=len(inputs):
        raise ValueError('Insufficient consistent overlap. Supply overlapping orthophotos, georeferencing or pixel offsets.')
    return [positions[i].tolist() for i in range(len(inputs))], 'overlap_translation'


def assemble(run, payload):
    inputs = payload['images']
    bounds, polygon, roi_path = (payload.get(k) for k in ('bounds','polygon','roi'))
    with ExitStack() as stack:
        datasets = []
        for source in inputs:
            with rasterio.open(source['path']) as ds:
                datasets.append(Source(source['path'],ds.width,ds.height,ds.crs,ds.transform))
        readers = OrderedDict()
        def reader(path):
            if path not in readers:
                if len(readers) == 16:
                    readers.popitem(last=False)[1].close()
                readers[path] = rasterio.open(path)
            readers.move_to_end(path)
            return readers[path]
        def close_readers():
            for ds in readers.values():
                ds.close()
        stack.callback(close_readers)
        offsets, method = placements(datasets,inputs)
        spans = [(x,y,x+d.width,y+d.height) for (x,y),d in zip(offsets,datasets)]
        union = [min(a[0] for a in spans), min(a[1] for a in spans),
                 max(a[2] for a in spans), max(a[3] for a in spans)]
        size = payload.get('size_px')
        if payload.get('size_m'):
            scale = payload['scale']
            gsd = scale.get('m_per_px')
            if gsd is None and scale.get('image_width_m'):
                gsd = scale['image_width_m']/datasets[0].width
            if gsd is None and scale.get('image_height_m'):
                gsd = scale['image_height_m']/datasets[0].height
            if not gsd or scale.get('station_map_path'):
                return dict(status='needs_sample_unit_boundary', question='Supply the SU size in pixels, a pixel boundary or a uniform meters-per-pixel scale.')
            size = [round(length/gsd) for length in payload['size_m']]
        if size:
            if bounds:
                raise ValueError('Use either crop bounds or a rectangular SU size.')
            left,top = payload.get('origin') or union[:2]
            bounds = [left,top,left+size[0],top+size[1]]
        roi = stack.enter_context(rasterio.open(roi_path)) if roi_path else None
        if roi and polygon:
            raise ValueError('Use one pavement ROI mask or one polygon.')
        if roi and (roi.width,roi.height)!=(union[2]-union[0],union[3]-union[1]):
            raise ValueError('The ROI raster must match the uncropped mosaic grid.')
        if bounds is None and polygon:
            xx, yy = zip(*polygon)
            bounds = [math.floor(min(xx)),math.floor(min(yy)),math.ceil(max(xx)),math.ceil(max(yy))]
        if bounds is None and roi:
            extent = [roi.width,roi.height,0,0]
            for x,y,w,h in native.rects(roi.width,roi.height):
                binary,_ = native.read_binary(roi,(x,y,w,h))
                ys,xs = np.nonzero(binary)
                if xs.size:
                    extent=[min(extent[0],x+int(xs.min())),min(extent[1],y+int(ys.min())),
                            max(extent[2],x+int(xs.max())+1),max(extent[3],y+int(ys.max())+1)]
            bounds=[extent[i]+union[i%2] for i in range(4)]
        x0,y0,x1,y1 = bounds or union
        if x1<=x0 or y1<=y0:
            raise ValueError('Empty Sample Unit boundary.')
        if x0<union[0] or y0<union[1] or x1>union[2] or y1>union[3]:
            return dict(status='needs_more_images',question='The requested SU extends beyond the supplied image extent. Add adjoining source images with their placement.')
        first = datasets[0]
        geometry = {'type':'Polygon','coordinates':[polygon]} if polygon else None
        info = dict(registration=method, images=[{**native.fingerprint(s['path']),'offset_xy':v}
                    for s,v in zip(inputs,offsets)], source_extent_xyxy=union, bounds_xyxy=[x0,y0,x1,y1],
                    width=x1-x0,height=y1-y0,first_source_size=[first.width,first.height])
        strict = bool(bounds or len(inputs)>1 or roi)
        added = repeated = missing = 0
        path = run/'sample_unit.tif'
        profile = native.profile(x1-x0,y1-y0)
        profile.update(count=4,photometric='RGB',BIGTIFF='IF_SAFER',crs=first.crs,
                       transform=first.transform*Affine.translation(x0-offsets[0][0],y0-offsets[0][1]))
        with rasterio.open(path,'w',**profile) as out:
            out.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
            for lx,ly,w,h in native.rects(out.width,out.height):
                x,y=x0+lx,y0+ly
                want = np.ones((h,w),bool)
                if roi:
                    want,_=native.read_binary(roi,(x-union[0],y-union[1],w,h))
                elif geometry:
                    want=rasterize([(geometry,1)],out_shape=(h,w),transform=Affine.translation(x,y),dtype='uint8').astype(bool)
                occupied=np.zeros((h,w),bool)
                canvas=np.zeros((h,w,4),np.uint8)
                for d,(a,b,c,e) in zip(datasets,spans):
                    l,t,r,f=max(x,a),max(y,b),min(x+w,c),min(y+h,e)
                    if l>=r or t>=f:
                        continue
                    d=reader(d.path)
                    block=(l-a,t-b,r-l,f-t)
                    dest=np.s_[t-y:f-y,l-x:r-x]
                    valid=native.alpha(d,block) & want[dest]
                    repeated+=int((valid & occupied[dest]).sum())
                    take=valid & ~occupied[dest]
                    if take.any():
                        canvas[dest][:,:,:3][take]=native.rgb(d,block)[take]
                        occupied[dest] |= take
                added+=int(occupied.sum())
                missing+=int((want & ~occupied).sum())
                canvas[:,:,3]=occupied.astype('uint8')*255
                out.write(np.moveaxis(canvas,-1,0),window=Window(lx,ly,w,h))
        info.update(valid_area_px2=added,overlap_pixels_skipped=repeated,uncovered_px2=missing)
        if not added or (strict and missing):
            return dict(**info,status='needs_more_images',question='Add imagery covering the missing SU pixels, or provide the actual pavement boundary.')
        return dict(**info,status='ready',image_path=str(path))
