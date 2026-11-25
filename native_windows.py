"""Bounded native-raster reads, dual-scale evidence, and whole-sample geometry."""
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import warnings
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

from fusion_source.geometry import Mapping, Patch, thin
from fusion_source.line_evidence import trace_visible

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
STRIDE = 384
SIZES = (768, 1536)
BLOCK = 1024
MAX_OVERVIEW = 1600


def fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_image(image,path):
    temporary=path.with_name(path.stem+"."+uuid4().hex+path.suffix)
    image.save(temporary,**({"quality":92} if path.suffix==".jpg" else {}))
    temporary.replace(path)


def verify_source(item):
    if fingerprint(item["path"]) != item:
        raise ValueError("Source raster changed. Prepare a new run.")


def rects(width, height, size=BLOCK):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield x, y, min(size, width-x), min(size, height-y)


def read_binary(ds, box):
    x,y,w,h = box
    if ds.count == 1:
        data = ds.read(1,window=Window(x,y,w,h))
    elif ds.count in (3,4):
        channels=ds.read(window=Window(x,y,w,h))
        if not np.array_equal(channels[0],channels[1]) or not np.array_equal(channels[0],channels[2]):
            raise ValueError("Use grayscale binary masks; color overlays are not masks.")
        if ds.count==4 and (channels[3]!=255).any():
            raise ValueError("Binary mask alpha must be opaque.")
        data=channels[0]
    else:
        raise ValueError("Masks must be grayscale binary rasters with zero background.")
    values = np.unique(data)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError("Masks need finite nonnegative pixel values.")
    return data > 0, {int(n) for n in values if n != 0}


def alpha(ds, box):
    x,y,w,h = box
    if ColorInterp.alpha in ds.colorinterp:
        band = ds.colorinterp.index(ColorInterp.alpha)+1
        return ds.read(band,window=Window(x,y,w,h)) > 0
    return ds.dataset_mask(window=Window(x,y,w,h)) > 0


def profile(width,height):
    return dict(driver="GTiff",width=width,height=height,count=1,dtype="uint8",
                tiled=True,blockxsize=256,blockysize=256,compress="deflate",SPARSE_OK="TRUE")


def rgb(ds, box, out_shape=None):
    x,y,w,h=box
    indexes=(1,2,3) if ds.count>=3 else (1,)
    if len(indexes)==1:
        raw=ds.read(1,window=Window(x,y,w,h),out_shape=out_shape,resampling=Resampling.average)
        result=np.stack([raw,raw,raw],axis=-1)
    else:
        shape=(3,*out_shape) if out_shape else None
        raw=ds.read(indexes,window=Window(x,y,w,h),out_shape=shape,resampling=Resampling.average)
        result=np.moveaxis(raw,0,-1)
    if result.dtype != np.uint8:
        raise ValueError("Original images must have 8-bit RGB or grayscale pixels.")
    return result


def preview_binary(ds,box,shape):
    return ds.read(1,window=Window(*box),out_shape=shape,out_dtype="float32",
                   resampling=Resampling.average)>0


def prepare(run,payload):
    """Snapshot masks blockwise and create a lazy view plan without large RGB arrays."""
    masks=payload["masks"]
    raw_path=payload["image_path"]
    roi_path=payload["roi_path"]
    with ExitStack() as stack:
        raw=stack.enter_context(rasterio.open(raw_path)) if raw_path else None
        roi=stack.enter_context(rasterio.open(roi_path)) if roi_path else None
        if raw is not None:
            width,height=raw.width,raw.height
        elif roi is not None:
            width,height=roi.width,roi.height
        else:
            first=stack.enter_context(rasterio.open(masks[0]["path"]))
            if masks[0].get("bounds_xyxy"):
                raise ValueError("Cropped instance masks require a full pavement ROI or aligned original image.")
            width,height=first.width,first.height
        for ds in (raw,roi):
            if ds is not None and (ds.width,ds.height)!=(width,height):
                raise ValueError("Original and ROI must share the full pixel grid.")
        meta=dict(width=width,height=height,images=[],masks=[],source_sha256={},
                  pavement_area_px2=0,assumed_full_canvas=raw is None and roi is None)
        grid=raw if raw is not None else roi
        if grid is not None:
            meta["source_transform"]=list(grid.transform)
            meta["source_crs"]=str(grid.crs) if grid.crs else None
        if raw is not None:
            meta["original_source"]=fingerprint(raw_path)
        station_map=payload.get("station_map")
        mapper=Mapping(Path(station_map),grid.transform,width,height) if station_map and grid else None
        if station_map and not mapper:
            raise ValueError("Station mapping requires an aligned original or georeferenced pavement ROI TIFF.")
        meta["mapped_pavement_area_m2"]=0.0 if mapper else None
        roi_out=run/"roi.tif"
        roi_profile=profile(width,height)
        if grid:
            roi_profile.update(transform=grid.transform,crs=grid.crs)
        with rasterio.open(roi_out,"w",**roi_profile) as dst:
            for box in rects(width,height):
                x,y,w,h=box
                valid=alpha(raw,box) if raw else np.ones((h,w),bool)
                if roi:
                    binary,_=read_binary(roi,box)
                    valid &= binary
                meta["pavement_area_px2"]+=int(valid.sum())
                if mapper:
                    meta["mapped_pavement_area_m2"]+=mapper.area(Patch(x,y,valid,"pavement",1,[]))[0]
                dst.write(valid.astype(np.uint8),1,window=Window(*box))
        if not meta["pavement_area_px2"]:
            raise ValueError("No valid pavement pixels.")
        roi_ds=stack.enter_context(rasterio.open(roi_out))
        for i,spec in enumerate(masks,1):
            source=stack.enter_context(rasterio.open(spec["path"]))
            bounds=spec.get("bounds_xyxy") or [0,0,width,height]
            x0,y0,x1,y1=bounds
            if not (0<=x0<x1<=width and 0<=y0<y1<=height):
                raise ValueError("Instance mask bounds are outside the sample.")
            if (source.width,source.height)!=(x1-x0,y1-y0):
                raise ValueError("Mask dimensions must match bounds_xyxy or the full pixel grid.")
            mask_id=f"mask_{i}"
            path=run/f"{mask_id}.tif"
            values=set()
            count=0
            extent=[width,height,0,0]
            with rasterio.open(path,"w",**profile(width,height)) as dst:
                for local in rects(source.width,source.height):
                    lx,ly,w,h=local
                    data,nonzero=read_binary(source,local)
                    values |= nonzero
                    if len(values)>1:
                        raise ValueError("Mask must have a single nonzero foreground value.")
                    x,y=x0+lx,y0+ly
                    data &= roi_ds.read(1,window=Window(x,y,w,h))>0
                    n=int(data.sum())
                    if n:
                        yy,xx=np.nonzero(data)
                        extent=[min(extent[0],x+int(xx.min())),min(extent[1],y+int(yy.min())),
                                max(extent[2],x+int(xx.max())+1),max(extent[3],y+int(yy.max())+1)]
                        dst.write(data.astype(np.uint8),1,window=Window(x,y,w,h))
                        count+=n
            if not count:
                raise ValueError(f"{mask_id}: no mask foreground inside the pavement.")
            item=dict(mask_id=mask_id,kind=spec["kind"],semantics=spec["semantics"],
                      foreground_px2=count,bounds_xyxy=extent,source=fingerprint(spec["path"]),
                      snapshot=fingerprint(path))
            meta["masks"].append(item)
            source.close()
        scale=min(1.,MAX_OVERVIEW/width,MAX_OVERVIEW/height)
        target=(max(1,round(height*scale)),max(1,round(width*scale)))
        if raw:
            preview=rgb(raw,(0,0,width,height),target)
            valid=preview_binary(roi_ds,(0,0,width,height),target)
            preview[~valid]=190
            Image.fromarray(preview).save(run/"original.jpg",quality=88)
            meta["images"].append(dict(role="original" if not masks else "original_aligned",path="original.jpg"))
        for item in meta["masks"]:
            with rasterio.open(run/f"{item['mask_id']}.tif") as ds:
                data=preview_binary(ds,(0,0,width,height),target)
            name=f"{item['mask_id']}_preview.png"
            Image.fromarray(data.astype(np.uint8)*255).save(run/name)
            meta["images"].append(dict(role=item["mask_id"],path=name))
        plan=run/"windows.jsonl"
        counts={str(size):0 for size in SIZES}
        skipped=0
        # Same source origins as CAE_Current_Detection_Fusion.infer_sections.
        with plan.open("w") as output:
            for size in SIZES:
                for y in range(0, math.ceil(max(0,height-size)/STRIDE)*STRIDE+1, STRIDE):
                    for x in range(0, math.ceil(max(0,width-size)/STRIDE)*STRIDE+1, STRIDE):
                        w,h=min(size,width-x),min(size,height-y)
                        window=Window(x,y,w,h)
                        if not np.any(roi_ds.read(1,window=window)):
                            skipped+=1
                            continue
                        relevant=[]
                        for item in meta["masks"]:
                            a,b,c,d=item["bounds_xyxy"]
                            if x<c and x+w>a and y<d and y+h>b:
                                with rasterio.open(run/f"{item['mask_id']}.tif") as ds:
                                    if ds.read(1,window=window).any():
                                        relevant.append(item["mask_id"])
                        if masks and not relevant:
                            skipped+=1
                            continue
                        groups=[relevant[j:j+3] for j in range(0,len(relevant),3)] or [[]]
                        for group,ids in enumerate(groups):
                            row=dict(view_id=f"w{size}_{x}_{y}_g{group}",size=size,x=x,y=y,width=w,height=h,
                                     mask_ids=ids)
                            output.write(json.dumps(row)+"\n")
                            counts[str(size)]+=1
        meta["native_views"]=dict(plan=str(plan),counts=counts,total=sum(counts.values()),
                                  skipped_empty=skipped,stride=STRIDE,large_display_px=768)
        meta["storage"]="windowed_tiff"
        meta["roi_snapshot"]=fingerprint(roi_out)
        meta["strategy"]="CAE_Current_Detection_Fusion_dual_window"
        return meta


def views(run,payload):
    meta=payload["meta"]
    verify_source(meta["roi_snapshot"])
    for item in meta["masks"]:
        verify_source(item["snapshot"])
    if meta.get("original_source"):
        verify_source(meta["original_source"])
    start,limit=payload["cursor"],payload["limit"]
    rows=[]
    cursor=0
    with (run/"windows.jsonl").open() as stream:
        for cursor,line in enumerate(stream):
            if cursor<start:
                continue
            if len(rows)==limit:
                break
            rows.append(json.loads(line))
    images=[]
    for row in rows:
        x,y,w,h=(row[k] for k in ("x","y","width","height"))
        shape=(max(1,round(h*min(1,768/row["size"]))),max(1,round(w*min(1,768/row["size"]))))
        box=(x,y,w,h)
        row["mask_weights"]={}
        if meta.get("original_source"):
            with rasterio.open(meta["original_source"]["path"]) as ds:
                data=rgb(ds,box,shape)
            with rasterio.open(run/"roi.tif") as roi:
                valid=preview_binary(roi,box,shape)
            data[~valid]=190
            path=run/f"{row['view_id']}_rgb.jpg"
            save_image(Image.fromarray(data),path)
            images.append(dict(role="original",view_id=row["view_id"],path=str(path)))
        for mask_id in row["mask_ids"]:
            with rasterio.open(run/f"{mask_id}.tif") as ds:
                native=ds.read(1,window=Window(*box))>0
                data=cv2.resize(native.astype(np.float32),(shape[1],shape[0]),interpolation=cv2.INTER_AREA)>0
            xx=np.arange(w)
            yy=np.arange(h)
            size=row["size"]
            dist=np.minimum(np.minimum(xx,size-1-xx)[None,:],np.minimum(yy,size-1-yy)[:,None])
            if size==1536:
                weight=(.15+.85*np.clip((dist-96)/288,0,1))
                weight[dist<96]=0
            else:
                left=xx if x else xx+size
                top=yy if y else yy+size
                right=size-1-xx if x+w<meta["width"] else size-1-xx+size
                bottom=size-1-yy if y+h<meta["height"] else size-1-yy+size
                dist=np.minimum(np.minimum(left,right)[None,:],np.minimum(top,bottom)[:,None])
                weight=.06+.94*np.clip(dist/(size/4),0,1)
            row["mask_weights"][mask_id]=float(weight[native].sum())
            path=run/f"{row['view_id']}_{mask_id}.png"
            save_image(Image.fromarray(data.astype(np.uint8)*255),path)
            images.append(dict(role=mask_id,view_id=row["view_id"],path=str(path)))
    next_cursor=start+len(rows)
    return dict(windows=rows,images=images,total_windows=meta["native_views"]["total"],
                next_cursor=next_cursor if next_cursor<meta["native_views"]["total"] else None)


def intersect(a,b):
    x,y=max(a[0],b[0]),max(a[1],b[1])
    r,d=min(a[2],b[2]),min(a[3],b[3])
    return (x,y,r,d) if x<r and y<d else None


def region(row,meta):
    if row["box"] is None:
        return (0,0,meta["width"],meta["height"])
    a,b,c,d=row["box"]
    if not 0<=a<c<=1000 or not 0<=b<d<=1000:
        raise ValueError("Judgment box must use ordered normalized 0–1000 coordinates.")
    return (math.floor(a*meta["width"]/1000),math.floor(b*meta["height"]/1000),
            math.ceil(c*meta["width"]/1000),math.ceil(d*meta["height"]/1000))


def edge_length(skeleton,x,y,core,mapper=None):
    """Count each forward skeleton edge at its original full-grid origin once."""
    padded=np.pad(skeleton,1)
    yy,xx=np.nonzero(skeleton)
    gx,gy=xx+x,yy+y
    a,b,c,d=core
    eligible=(gx>=a)&(gx<c)&(gy>=b)&(gy<d)
    length=0.
    for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
        hit=eligible & padded[yy+1+dy,xx+1+dx]
        if dx and dy:
            hit &= ~(padded[yy+1,xx+1+dx]|padded[yy+1+dy,xx+1])
        if mapper:
            e,n=mapper.xy(gx[hit]+.5,gy[hit]+.5)
            ee,nn=mapper.xy(gx[hit]+.5+dx,gy[hit]+.5+dy)
            length+=float(np.hypot(ee-e,nn-n).sum())
        else:
            length+=float(hit.sum())*math.hypot(dx,dy)
    return length


def measure(run,payload):
    """Measure disjoint whole-grid instance ownership in bounded tiles with halos."""
    from image_worker import footprint
    meta=payload["meta"]
    verify_source(meta["roi_snapshot"])
    for item in meta["masks"]:
        verify_source(item["snapshot"])
    width,height=meta["width"],meta["height"]
    known={m["mask_id"]:m for m in meta["masks"]}
    decisions=payload["judgments"]
    configs=[]
    covered={key:0 for key in known}
    for row in decisions:
        item=known.get(row["mask_id"])
        if item is None:
            raise ValueError("Unknown mask ID.")
        kind=item["kind"] or row["kind"]
        if item["kind"] and row["kind"] not in (None,item["kind"]):
            raise ValueError(f"keep the supplied class {item['kind']}")
        if not kind:
            raise ValueError("Class decision is required.")
        semantics=item["semantics"] if item["semantics"]!="unspecified" else row["semantics"]
        if kind in ("alligator","block_crack","patch","pothole") and semantics=="unspecified":
            raise ValueError("Specify crack_trace or affected_region semantics.")
        cfg=dict(row,kind=kind,semantics=semantics,region=region(row,meta),
                 qualified=kind!="unknown" and not(kind in ("patch","pothole") and semantics=="crack_trace"))
        configs.append(cfg)
    if set(known)!={r["mask_id"] for r in configs}:
        raise ValueError("Every supplied mask requires a judgment.")
    rows=[dict(mask_id=c["mask_id"],kind=c["kind"],severity_lower=c["severity_lower"],
               severity_upper=c["severity_upper"],severity_source=c["severity_source"],
               class_source="provided" if known[c["mask_id"]]["kind"] else "vision_agent",
               evidence=c["evidence"],box=c["box"],semantics=c["semantics"],
               trace_or_mask_area_px2=0,measured_area_px2=0,added_affected_area_px2=0,
               length_px=0.,count=0,overlap_removed_px2=0,scored=False,withheld=not c["qualified"],
               mapped_area_m2=0.,mapped_length_m=0.,rgb_refined_pixels=0)
          for c in configs]
    scale=payload["scoring_scale"]
    halo=max(64,min(1024,math.ceil(1.7/scale))) if any(c["semantics"]=="crack_trace" and c["kind"] in ("alligator","block_crack") for c in configs) else 64
    cross_overlap=0
    with ExitStack() as stack:
        masks={key:run/f"{key}.tif" for key in known}
        roi=stack.enter_context(rasterio.open(run/"roi.tif"))
        raw=stack.enter_context(rasterio.open(meta["original_source"]["path"])) if meta.get("original_source") else None
        if raw:
            verify_source(meta["original_source"])
        station=payload.get("station_map")
        mapper=Mapping(Path(station),roi.transform,width,height) if station else None
        for core_box in rects(width,height):
            cx,cy,cw,ch=core_box
            core=(cx,cy,cx+cw,cy+ch)
            x,y=max(0,cx-halo),max(0,cy-halo)
            r,b=min(width,cx+cw+halo),min(height,cy+ch+halo)
            box=(x,y,r-x,b-y)
            window=Window(*box)
            valid=roi.read(1,window=window)>0
            if not valid.any():
                continue
            active=[]
            raw_cache={}
            for index,cfg in enumerate(configs):
                mask_id=cfg["mask_id"]
                if not intersect(known[mask_id]["bounds_xyxy"],(x,y,r,b)):
                    continue
                bound=intersect(cfg["region"],(x,y,r,b))
                if bound is None:
                    continue
                if mask_id not in raw_cache:
                    with rasterio.open(masks[mask_id]) as source:
                        raw_cache[mask_id]=source.read(1,window=window)>0
                selected=np.zeros(valid.shape,bool)
                a,t,c,d=bound
                selected[t-y:d-y,a-x:c-x]=True
                trace=raw_cache[mask_id]&valid&selected
                active.append([index,cfg,trace,trace.copy(),selected])
            cs=np.s_[cy-y:cy-y+ch,cx-x:cx-x+cw]
            claimed={}
            for index,cfg,trace,final,selected in active:
                key=cfg["mask_id"]
                previous=claimed.setdefault(key,np.zeros(valid.shape,bool))
                if (previous[cs]&trace[cs]).any():
                    raise ValueError(f"{key}: boxes overlap on foreground pixels.")
                previous|=trace
                raw_count=int(trace[cs].sum())
                covered[key]+=raw_count
                rows[index]["trace_or_mask_area_px2"]+=raw_count
                if cfg["kind"] in ("alligator","block_crack") and cfg["semantics"]=="crack_trace":
                    # Existing mask is the anchor. Filled regions are generated only
                    # from supplied traces, never from neighboring unrelated detections.
                    final[:]=footprint(trace,valid&selected,scale,.5 if cfg["kind"]=="alligator" else 1.5)
            area=np.zeros(valid.shape,bool)
            for _,cfg,_,final,_ in active:
                if cfg["qualified"] and cfg["kind"] in ("alligator","block_crack","patch"):
                    area|=final
            seen={}
            areas_seen=np.zeros(valid.shape,bool)
            collided_areas=np.zeros(valid.shape,bool)
            rgb_data=None
            for index,cfg,trace,final,_ in active:
                row=rows[index]
                used=final.copy()
                if cfg["qualified"]:
                    prior=seen.setdefault(cfg["kind"],np.zeros(valid.shape,bool))
                    blocked=prior|area if cfg["kind"]=="linear_crack" else prior
                    row["overlap_removed_px2"]+=int((used[cs]&blocked[cs]).sum())
                    used &= ~blocked
                    prior|=used
                    if cfg["kind"]!="linear_crack":
                        collided_areas|=used&areas_seen
                        used &= ~areas_seen
                        areas_seen|=used
                count=int(used[cs].sum())
                row["measured_area_px2"]+=count
                row["added_affected_area_px2"]+=int((used[cs]&~trace[cs]).sum())
                if mapper and count:
                    row["mapped_area_m2"]+=mapper.area(Patch(cx,cy,used[cs],cfg["kind"],1,[]))[0]
                if cfg["qualified"] and cfg["kind"]=="linear_crack" and used.any():
                    native_two_mm=mapper is not None or math.isclose(scale,.002,rel_tol=1e-6)
                    if raw and native_two_mm and cv2.countNonZero((cv2.distanceTransform(used.astype(np.uint8),cv2.DIST_L2,5)>=14).astype(np.uint8))>=1000:
                        if rgb_data is None:
                            rgb_data=rgb(raw,box)
                        line=trace_visible(np.ascontiguousarray(rgb_data),used)
                        row["rgb_refined_pixels"]+=int((used[cs]&~line[cs]).sum())
                    else:
                        line=thin(used)
                    row["length_px"]+=edge_length(line,x,y,core)
                    if mapper:
                        row["mapped_length_m"]+=edge_length(line,x,y,core,mapper)
            cross_overlap+=int(collided_areas[cs].sum())
    for row,cfg in zip(rows,configs,strict=True):
        if cfg["qualified"] and cfg["kind"]=="pothole" and row["measured_area_px2"]:
            row["count"]=1
        q=row["length_px"] if cfg["kind"]=="linear_crack" else row["count"] if cfg["kind"]=="pothole" else row["measured_area_px2"]
        row["scored"]=bool(cfg["qualified"] and q>0)
        row["withheld"] |= bool(cfg["kind"]=="linear_crack" and row["measured_area_px2"] and not q)
    unassigned={key:known[key]["foreground_px2"]-n for key,n in covered.items() if known[key]["foreground_px2"]>n}
    return dict(rows=rows,unassigned_pixels=unassigned,cross_type_overlap_px2=cross_overlap,
                measurement=dict(block_px=BLOCK,halo_px=halo,coordinate_basis="station_to_UTM" if mapper else "uniform_pixel_grid"))
