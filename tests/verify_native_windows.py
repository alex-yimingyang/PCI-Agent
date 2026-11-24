"""Verify native plans, review checkpoints, seam-safe quantities and physical mapping."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_image_agents import connect,call,MODES

ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get("PCI_IMAGE_PYTHON",sys.executable)


def fixture(folder):
    script = r"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
p=Path(sys.argv[1]); h,w=900,2400
rgb=np.full((4,h,w),130,np.uint8); rgb[3]=255; rgb[3,:,2350:]=0
rgb[:3,200,20:2380]=20
profile=dict(driver='GTiff',width=w,height=h,count=4,dtype='uint8',
             transform=from_origin(0,.9,.002,.002),crs='EPSG:32617',
             tiled=True,blockxsize=256,blockysize=256,compress='deflate')
with rasterio.open(p/'raw.tif','w',**profile) as ds:
 ds.write(rgb); ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
with rasterio.open(p/'roi.tif','w',**{**profile,'count':1}) as ds:
 ds.write(rgb[3],1)
a=np.zeros((h,w),np.uint8); a[200,20:2380]=255; Image.fromarray(a).save(p/'line.png')
Image.fromarray(rgb[3]).save(p/'roi.png')
Image.fromarray(a[199:202,20:2380]).save(p/'line-crop.png')
a.fill(0); a[650:750,1050:1250]=255; Image.fromarray(a).save(p/'area.png')
(p/'station.csv').write_text('station_m,center_easting_m,center_northing_m,left_unit_easting,left_unit_northing\n0,500000,4400000,0,1\n5,500005,4400000,0,1\n')
"""
    subprocess.run([PYTHON,"-B","-c",script,str(folder)],check=True)


async def inspect(client,prepared):
    cursor=0
    windows=[]
    while cursor is not None:
        batch=await call(client,"read_pavement_windows",{"run_id":prepared["run_id"],"cursor":cursor,"limit":4})
        reviews=[]
        for view in batch["windows"]:
            assert view["size"] in (768,1536)
            assert view["x"]%384==view["y"]%384==0
            windows.append(view)
            reviews.append({"view_id":view["view_id"],"evidence":"Synthetic geometry test view",
                            "judgments":[{"mask_id":m,"kind":"linear_crack" if m!="mask_3" else "alligator",
                                          "severity_lower":"L","severity_upper":"M","evidence":"Synthetic view"}
                                         for m in view["mask_ids"]]})
        saved=await call(client,"save_pavement_window_reviews",{"run_id":prepared["run_id"],"reviews":reviews})
        cursor=batch["next_cursor"]
    assert saved["coverage"]["complete"] and saved["coverage"]["reviewed"]==len(windows)
    # Repeated delivery/review IDs replace an entry, never increase the coverage.
    same=await call(client,"save_pavement_window_reviews",{"run_id":prepared["run_id"],"reviews":[reviews[-1]]})
    assert same["coverage"]==saved["coverage"]
    return windows,saved


async def verify(folder):
    fixture(folder)
    runs=folder/"runs"
    known={"status":"known","m_per_px":.002,"station_map_path":str(folder/"station.csv")}
    masks=[{"path":str(folder/"line.png"),"kind":"linear_crack","semantics":"crack_trace"},
           {"path":str(folder/"line-crop.png"),"bounds_xyxy":[20,199,2380,202],
            "kind":"linear_crack","semantics":"crack_trace"},
           {"path":str(folder/"area.png"),"kind":"alligator","semantics":"affected_region"}]
    judgments=[{"mask_id":"mask_1","severity_lower":"L","severity_upper":"M","evidence":"Trace"},
               {"mask_id":"mask_2","severity":"L","evidence":"Duplicate trace"},
               {"mask_id":"mask_3","severity_lower":"L","severity_upper":"M","evidence":"Area"}]
    for j in judgments:
        j.update(severity_basis="rgb_crop",severity_rationale="Synthetic inspected instance remains intact.",
                 excluded_grade_evidence={"M":"Fine intact duplicate trace.","H":"No severe breakup in synthetic crop."})
    async with connect("labeled",runs) as client:
        prepared=await call(client,MODES["labeled"][0],{"masks":masks,"physical_scale":known,
                                                    "original_image_path":str(folder/"raw.tif")})
        assert prepared["pavement_area_px2"]==2350*900
        assert prepared["native_review_required"]
        await call(client,MODES["labeled"][1],{"run_id":prepared["run_id"],"judgments":judgments},
                   error="Native-window review incomplete")
        measuring=await call(client,MODES["labeled"][1],{"run_id":prepared["run_id"],"judgments":judgments,"measure_only":True})
        assert measuring["status"]=="needs_visual_severity_assessment"
        windows,saved=await inspect(client,prepared)
        assert any(view["size"]==768 for view in windows) and any(view["size"]==1536 for view in windows)
        final=await call(client,MODES["labeled"][1],{"run_id":prepared["run_id"],"judgments":judgments})
        measured=final["measurements"]
        assert measured[0]["length_px"]==2329,measured
        assert measured[1]["length_px"]==0 and not measured[1]["scored"]
        assert measured[1]["overlap_removed_px2"]==2330
        assert measured[2]["measured_area_px2"]==20000
        assert abs(measured[0]["measured_length_m"]-2329*.002)<1e-5
        assert abs(measured[2]["measured_area_m2"]-.08)<1e-6
        assert final["measurement"]["coordinate_basis"]=="station_to_UTM"
        assert final["native_review"]["complete"] and final["pci_interval"]
        assert saved["instance_consensus"][2]["status"]=="protected_supplied_anchor"
        labeled_interval=final["pci_interval"]
    async with connect("labeled",runs) as client:
        # A georeferenced ROI supports mapping without disclosing RGB. The surveyed
        # denominator may differ from unfolded pixel area and must be retained.
        prepared=await call(client,MODES["labeled"][0],{"masks":masks,
            "pavement_roi_mask":str(folder/"roi.tif"),
            "physical_scale":{**known,"pavement_area_m2":10}})
        assert prepared["scale"]["pavement_area_m2"]==10
        assert not prepared.get("original_source")
        mapped=await call(client,MODES["labeled"][1],{"run_id":prepared["run_id"],
            "judgments":judgments,"measure_only":True})
        assert abs(mapped["measurements"][0]["measured_length_m"]-2329*.002)<1e-5
        assert abs(mapped["measurements"][2]["measured_area_m2"]-.08)<1e-6
        assert abs(mapped["measurements"][2]["density_metric"]-.8)<1e-6
    async with connect("image",runs) as client:
        prepared=await call(client,MODES["image"][0],{"image_path":str(folder/"raw.tif"),
                                                  "physical_scale":{"status":"unavailable"}})
        visual=dict(pci=65,pci_lower=55,pci_upper=74,evidence="Synthetic visual interval",lower_evidence="Possible fine cracks",upper_evidence="Intact visible surface")
        await call(client,MODES["image"][1],{"run_id":prepared["run_id"],**visual},
                   error="Native-window review incomplete")
        await inspect(client,prepared)
        final=await call(client,MODES["image"][1],{"run_id":prepared["run_id"],**visual})
        assert final["pci"]==65 and final["native_review"]["complete"]
    async with connect("mask",runs) as client:
        unlabeled=[{key:value for key,value in mask.items() if key!="kind"} for mask in masks]
        prepared=await call(client,MODES["mask"][0],{"masks":unlabeled,
                         "pavement_roi_mask":str(folder/"roi.png"),"physical_scale":{"status":"unavailable"}})
        windows,saved=await inspect(client,prepared)
        assert saved["instance_consensus"][2]["status"]=="consensus",saved
        inferred=await call(client,MODES["mask"][1],{"run_id":prepared["run_id"],"judgments":judgments})
        assert inferred["measurements"][0]["kind"]=="linear_crack"
        assert inferred["measurements"][0]["length_px"]==2329
        assert inferred["pci_proxy_interval"] and inferred["native_review"]["complete"]
        family=[judgments[0],judgments[1],{**judgments[2],"kind":"crack",
            "preferred_crack_kind":"alligator","class_candidates":["alligator","block_crack"],
            "class_rationale":"Synthetic filled region supports an area-cracking family; subtype is intentionally ambiguous.",
            "semantics":"affected_region"}]
        bounded=await call(client,MODES['mask'][1],{'run_id':prepared['run_id'],'judgments':family})
        assert bounded['native_review']['complete'] and bounded['class_search']['complete']
        assert bounded['class_search']['evaluated_assignments']==2
        assert bounded['measurements'][2]['class_family']=='crack'
        assert bounded['measurements'][2]['measured_area_px2']==20000
        assert bounded['pci_proxy_interval']['lower']<=inferred['pci_proxy_interval']['lower']
        assert bounded['pci_proxy_interval']['upper']>=inferred['pci_proxy_interval']['upper']
    print(json.dumps({"native_windows":len(windows),"seam_safe_length_px":2329,
                      "duplicate_pixels_removed":2330,"area_px2":20000,
                      "mapped_line_m":measured[0]["measured_length_m"],
                      "mapped_area_m2":measured[2]["measured_area_m2"],
                      "labeled_pci_interval":labeled_interval,
                      "georeferenced_roi_without_rgb_verified":True,
                      "review_checkpoint_and_guard_verified":True},indent=2))


if __name__=="__main__":
    with TemporaryDirectory(prefix="native PCI ") as folder:
        asyncio.run(verify(Path(folder).resolve()))
