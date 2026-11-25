"""Verify sample assembly and clipping against a known full-resolution surface."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from affine import Affine

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sample_imagery import assemble
from native_windows import prepare
from detector_views import tiles
from carriageway import mask as lane_mask


def verify(folder):
    rng=np.random.default_rng(851)
    surface=rng.integers(0,256,(320,2200,3),dtype=np.uint8)
    inputs=[]
    for i,x in enumerate((0,630,1240)):
        path=folder/f'source-{i}.png'
        Image.fromarray(surface[:,x:min(x+1000,2200)]).save(path)
        inputs.append({'path':str(path),'offset_xy':[x,0]})
    def check(name, images, **kwargs):
        run=folder/name
        run.mkdir()
        return assemble(run,{'images':images,**kwargs})
    def same(result,expected):
        assert result['status']=='ready',result
        with rasterio.open(result['image_path']) as d:
            assert np.array_equal(d.read([1,2,3]).transpose(1,2,0),expected)
            assert np.all(d.read(4)==255)
    result=check('placed',inputs)
    same(result,surface)
    assert result['valid_area_px2']==2200*320 and result['overlap_pixels_skipped']==760*320
    # Oversized SU canvases are tiled on the final grid. There is one inventory and denominator.
    run=folder/'placed'
    meta=prepare(run,dict(image_path=result['image_path'],masks=[],roi_path=None))
    assert meta['pavement_area_px2']==2200*320
    tiles(run,dict(image=result['image_path'],folder=str(run)))
    import json
    plan=json.loads((run/'tiles.json').read_text())
    cover=np.zeros(surface.shape[:2],bool)
    for t in plan:
        assert max(t['width'],t['height'])<=1536
        cover[t['y']:t['y']+t['height'],t['x']:t['x']+t['width']]=True
    assert cover.all() and len(plan)>1
    # Check actual detector pixels across raster blocks and overlapping windows.
    lanes=folder/'road-windows'
    lanes.mkdir()
    outline=[[0,90],[2200,90],[2200,260],[0,260]]
    cutout=[[1010,120],[1050,120],[1050,150],[1010,150]]
    road=lane_mask(lanes,dict(image=result['image_path'],extent='mixed',regions=[dict(boundary=outline,holes=[cutout])]))
    valid=np.zeros(surface.shape[:2],bool)
    valid[90:260,:]=True
    valid[120:150,1010:1050]=False
    assert road['retained_px2']==int(valid.sum())
    prepare(lanes,dict(image_path=result['image_path'],masks=[],roi_path=road['mask_path']))
    tiles(lanes,dict(image=result['image_path'],folder=str(lanes)))
    for tile in json.loads((lanes/'tiles.json').read_text()):
        x,y,w,h=[tile[key] for key in ('x','y','width','height')]
        area=valid[y:y+h,x:x+w]
        pixels=np.asarray(Image.open(tile['path']))
        assert np.all(pixels[~area]==190)
        assert np.array_equal(pixels[area],surface[y:y+h,x:x+w][area])

    same(check('cropped',inputs,bounds=[500,32,1400,210]),surface[32:210,500:1400])
    same(check('metric_crop',inputs,size_m=[.5,.46],scale={'m_per_px':.002},origin=[10,15]),surface[15:245,10:260])
    same(check('auto_crop',inputs,size_px=[250,230]),surface[:230,:250])
    same(check('single',inputs[:1],bounds=[10,15,260,245]),surface[15:245,10:260])
    unordered=[{'path':i['path']} for i in (inputs[1],inputs[2],inputs[0])]
    auto=check('registered',unordered)
    same(auto,surface)
    assert auto['registration']=='overlap_translation' and auto['bounds_xyxy']==[-630,0,1570,320]
    assert check('missing',inputs[:1],bounds=[0,0,1500,320])['status']=='needs_more_images'
    assert check('gap',[inputs[0],inputs[2]])['uncovered_px2']==240*320
    roi=np.zeros(surface.shape[:2],np.uint8)
    roi[15:245,10:260]=255
    Image.fromarray(roi).save(folder/'roi.png')
    same(check('roi',inputs,roi=str(folder/'roi.png')),surface[15:245,10:260])
    geo=[]
    for i,source in enumerate(inputs):
        x,_=source['offset_xy']
        path=folder/f'geo-{i}.tif'
        arr=np.array(Image.open(source['path']))
        transform=from_origin(360000,3800000,.002,.002)*Affine.translation(x,0)
        with rasterio.open(path,'w',driver='GTiff',width=arr.shape[1],height=320,count=3,dtype='uint8',
                           transform=transform,crs='EPSG:32617') as dst:
            dst.write(arr.transpose(2,0,1))
        geo.append({'path':str(path)})
    geo_result=check('geo',geo,bounds=[500,32,1400,210])
    same(geo_result,surface[32:210,500:1400])
    assert geo_result['registration']=='georeferenced_grid'
    print('Single-image crop, offset/visual/geospatial mosaics, gaps, overlap and inference-window coverage passed.')


if __name__=='__main__':
    with TemporaryDirectory() as tmp:
        verify(Path(tmp))
