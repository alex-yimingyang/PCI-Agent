"""Stage evidence for the conversational model, then score validated judgments."""

import base64
import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import select
import subprocess
import sys
import threading
from typing import Annotated, Literal
from uuid import uuid4

from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
from pydantic import BaseModel, Field, model_validator

import pci_workflow as workflow

Kind = Literal["linear_crack", "alligator", "block_crack", "patch", "pothole"]
CrackKind = Literal["linear_crack", "alligator", "block_crack"]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Grade = Literal["L", "M", "H"]
GRADES = ("L", "M", "H")
REFERENCE_AREA_M2 = 5000 * .3048**2
SCALE_QUESTION = "Is a physical scale available? Supply pixel scale, the physical width or height of the full image, or its valid pavement area. With no scale, I will use pixel units."
SAMPLE_UNIT_QUESTION = 'For Density, do you have a Sample Unit boundary or a separate area (m² or px²)? If none, use this single image’s valid pavement area.'


class SourceImage(BaseModel):
    path: str
    offset_xy: Annotated[list[int],Field(min_length=2,max_length=2)] | None = Field(default=None, description="Top-left on the common pixel grid; supply offsets for all sources or none.")


class Scale(BaseModel):
    status: Literal["known", "unavailable"]
    m_per_px: Positive | None = None
    pavement_area_m2: Positive | None = None
    image_width_m: Positive | None = Field(default=None, description="Ground width of the full image, assuming uniform square pixels.")
    image_height_m: Positive | None = Field(default=None, description="Ground height of the full image, assuming uniform square pixels.")
    station_map_path: str | None = Field(default=None, description="Optional CAE station-to-UTM CSV for the aligned TIFF.")


class SampleUnit(BaseModel):
    """Select single_image, the supplied pavement ROI, or a user-specified total area."""
    model_config = {"extra": "forbid"}
    mode: Literal["single_image", "roi", "provided", "mosaic"]
    name: str | None = None
    size_px: Annotated[list[Annotated[int,Field(gt=0)]],Field(min_length=2,max_length=2)] | None = None
    size_m: Annotated[list[Positive],Field(min_length=2,max_length=2)] | None = None
    origin_xy: Annotated[list[int],Field(min_length=2,max_length=2)] | None = None
    area_px2: Positive | None = None
    area_m2: Positive | None = None

    @model_validator(mode="after")
    def validate_area(self):
        if self.size_px and self.size_m:
            raise ValueError("Use one rectangular SU size in pixels or meters.")
        if self.origin_xy and not (self.size_px or self.size_m):
            raise ValueError("Supply a rectangular SU size with its origin.")
        supplied = self.area_px2 is not None or self.area_m2 is not None
        if supplied != (self.mode == "provided"):
            raise ValueError("Provide an area in px² or m² for a separate SU; single_image and roi use the valid pavement in the image.")
        return self


def resolve_sample_unit(unit, scale, image_px):
    metric = scale["status"] == "known"
    image_m2 = scale["pavement_area_m2"] if metric else None
    px, m2 = unit.area_px2, unit.area_m2
    if unit.mode != "provided":
        px, m2 = image_px, image_m2
    elif metric:
        # For a nonuniform station map only the mapped image footprint has a known conversion.
        if scale.get("station_map_path"):
            if px == image_px and m2 is None:
                m2 = image_m2
            if m2 == image_m2 and px is None:
                px = image_px
        else:
            cell = scale["m_per_px"] ** 2
            if px is not None and m2 is not None and not math.isclose(px*cell, m2, rel_tol=.02):
                raise ValueError("Sample Unit areas in m² and px² disagree with the supplied scale.")
            px = px if px is not None else m2 / cell
            m2 = m2 if m2 is not None else px * cell
    engine_area = m2 if metric else None
    if not metric and px is not None:
        engine_area = px * scale["assumed_m_per_px_for_pci_proxy"]**2
    return {"mode":unit.mode, "name":unit.name, "area_px2":px, "area_m2":m2,
            "engine_area_m2":engine_area, "engine_area_is_proxy":not metric,
            "image_area_px2":image_px, "image_area_m2":image_m2,
            "inventory_scope":"Distresses measured on the supplied image within its pavement ROI."}


def sample_unit_for_run(meta):
    # Older preparations used the valid image area as their Sample Unit.
    if "sample_unit" in meta:
        return meta["sample_unit"]
    return resolve_sample_unit(SampleUnit(mode="single_image"), meta["scale"], meta["pavement_area_px2"])


def reported_measurement(pixel_qty, metric_qty, pixel_unit, metric, unit):
    units = {"px":"m", "px2":"m2", "count":"count"}
    qty, qty_unit = (metric_qty, units[pixel_unit]) if metric else (pixel_qty, pixel_unit)
    area = unit["area_m2"] if metric else unit["area_px2"]
    basis = "m2" if metric else "px2"
    if pixel_unit == "count" and area is None:
        area = unit["area_m2"] if not metric else unit["area_px2"]
        basis = "m2" if not metric else "px2"
    density_unit = "percent" if pixel_unit == "px2" else f"{qty_unit}/100 {basis}"
    return {"quantity":qty, "quantity_unit":qty_unit,
            "density":100*qty/area if area is not None else None, "density_unit":density_unit}


class Judgment(BaseModel):
    mask_id: str
    kind: Kind | Literal["crack", "unknown"] | None = None
    severity: Grade | None = Field(default=None, description="Fixed grade shorthand. Equivalent to equal endpoints.")
    severity_lower: Grade | None = Field(default=None, description="Lowest admissible grade for this instance.")
    severity_upper: Grade | None = Field(default=None, description="Highest admissible grade for this instance.")
    evidence: Annotated[str, Field(min_length=1, max_length=1200)]
    box: Annotated[list[int], Field(min_length=4, max_length=4)] | None = None
    severity_basis: Literal["rgb_crop", "binary_mask", "measured_features", "provided", "ungradable"] | None = None
    severity_rationale: Annotated[str, Field(min_length=1,max_length=1800)] | None = None
    excluded_grade_evidence: dict[Grade,str] = Field(default_factory=dict,
        description="For every grade outside the submitted interval, explain the instance evidence excluding it.")
    forced_surface_estimate: bool = Field(default=False,
        description="Only when the user requests estimation despite insufficient severity evidence; inspect RGB per instance.")
    surface_estimate_assumptions: Annotated[str, Field(min_length=1,max_length=1800)] | None = None
    visual_confidence: Literal["low", "very_low"] | None = None
    preferred_severity: Grade | None = None

    @model_validator(mode="after")
    def ordered_severity(self):
        if (self.severity_lower is None) != (self.severity_upper is None):
            raise ValueError("Specify both severity endpoints, or omit both.")
        if self.severity_lower is not None:
            if GRADES.index(self.severity_lower) > GRADES.index(self.severity_upper):
                raise ValueError("Severity endpoints must be ordered L, M, H.")
            if (self.severity_lower,self.severity_upper)==("L","H") and not self.forced_surface_estimate:
                raise ValueError("L-H is not an evidence-narrowed severity assessment. Omit grades and mark ungradable.")
            if self.severity is not None and not self.severity == self.severity_lower == self.severity_upper:
                raise ValueError("Fixed severity conflicts with the supplied interval.")
        lo = self.severity_lower or self.severity
        hi = self.severity_upper or self.severity
        if self.preferred_severity is not None and (lo is None or not GRADES.index(lo) <= GRADES.index(self.preferred_severity) <= GRADES.index(hi)):
            raise ValueError("Preferred severity must lie within the submitted severity range.")
        if self.forced_surface_estimate:
            if (lo is None or self.severity_basis != "rgb_crop" or not self.severity_rationale
                    or not self.surface_estimate_assumptions or self.visual_confidence is None
                    or self.preferred_severity is None):
                raise ValueError("Forced surface estimation requires RGB basis, rationale, explicit assumptions, low confidence and a preferred grade.")
            if (lo,hi)==("L","H") and self.visual_confidence != "very_low":
                raise ValueError("An unresolved forced L-H range requires very_low confidence.")
        return self


class WindowReview(BaseModel):
    view_id: str
    evidence: Annotated[str, Field(min_length=1,max_length=1800)]


def window_plan(run):
    with (run / "windows.jsonl").open() as stream:
        return {row["view_id"]:row for row in map(json.loads,stream)}


def image_result(data,images):
    content=[TextContent(type="text",text=json.dumps(data,ensure_ascii=False))]
    for entry in images:
        path=Path(entry["path"])
        content.append(TextContent(type="text",text=entry.get("view_id","")+" "+entry["role"]))
        content.append(ImageContent(type="image",data=base64.b64encode(path.read_bytes()).decode("ascii"),
                                    mime_type="image/png" if path.suffix==".png" else "image/jpeg"))
    return ToolResult(content=content,structured_content=data)


def read_windows(run_id,cursor,limit):
    run,meta=load_run(run_id,None)
    if cursor>=meta["native_views"]["total"]:
        raise workflow.WorkflowError("Window cursor is past the end of the plan.")
    batch=worker("views",run,{"meta":meta,"cursor":cursor,"limit":limit})
    delivered=run/"delivered"
    delivered.mkdir(exist_ok=True)
    for row in batch["windows"]:
        files=[i for i in batch["images"] if i["view_id"]==row["view_id"]]
        target=delivered/f"{row['view_id']}.json"
        temporary=delivered/f"{row['view_id']}.{uuid4().hex}.tmp"
        workflow.write_json(temporary,dict(window=row,images={
            i["path"]:hashlib.sha256(Path(i["path"]).read_bytes()).hexdigest() for i in files}))
        temporary.replace(target)
    data={"run_id":run_id,**batch}
    return image_result(data,batch["images"])


def save_reviews(run_id,reviews):
    run,meta=load_run(run_id,None)
    plan=window_plan(run)
    saved=[]
    for review in reviews:
        if review.view_id not in plan or not (run/"delivered"/f"{review.view_id}.json").is_file():
            raise workflow.WorkflowError("Read the planned window before reviewing it.")
        row=plan[review.view_id]
        delivery=json.loads((run/"delivered"/f"{review.view_id}.json").read_text())
        for filename,expected in delivery["images"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
                raise workflow.WorkflowError("Native view changed. Read it again before saving the review.")
        target=run/"reviews"
        target.mkdir(exist_ok=True)
        payload={**review.model_dump(),"window":delivery["window"],"image_sha256":delivery["images"]}
        temporary=target/f"{review.view_id}.{uuid4().hex}.tmp"
        workflow.write_json(temporary,payload)
        temporary.replace(target/f"{review.view_id}.json")
        saved.append(review.view_id)
    return {"run_id":run_id,"saved_view_ids":saved,"coverage":check_window_coverage(run,meta,False)}


def check_window_coverage(run,meta,require_complete=True):
    if "native_views" not in meta:
        return {"required":False,"reviewed":0,"total":0,"complete":True,"legacy_preparation":True}
    total=meta["native_views"]["total"]
    reviewed=len(list((run/"reviews").glob("*.json")))
    complete=reviewed==total
    if require_complete and meta["native_review_required"] and not complete:
        raise workflow.WorkflowError(f"Native-window review incomplete ({reviewed}/{total}). "
                                     "Read and save the remaining planned windows before final scoring.")
    return {"required":meta["native_review_required"],"reviewed":reviewed,"total":total,"complete":complete,
            "reviews_dir":str(run/"reviews")}


def check_scale(scale, roi_area_px2=None):
    if scale is None:
        return {"status": "ask_user_for_scale", "question": SCALE_QUESTION}
    if scale.status == "unavailable":
        if any((scale.m_per_px, scale.pavement_area_m2, scale.station_map_path, scale.image_width_m, scale.image_height_m)):
            raise workflow.WorkflowError("Use status=known if you supply a physical scale.")
        if roi_area_px2 is None:
            return {"status": "unavailable", "reported_units": "px"}
        return {
            "status": "unavailable", "reported_units": "px",
            "reference_sample_area_m2_for_pci_proxy": REFERENCE_AREA_M2,
            "assumed_m_per_px_for_pci_proxy": math.sqrt(REFERENCE_AREA_M2 / roi_area_px2),
        }
    if not any((scale.m_per_px, scale.pavement_area_m2, scale.station_map_path, scale.image_width_m, scale.image_height_m)):
        raise workflow.WorkflowError("Known scale requires m_per_px or pavement_area_m2.")
    if roi_area_px2 is None:
        return scale.model_dump()
    m_per_px = scale.m_per_px or math.sqrt(scale.pavement_area_m2 / roi_area_px2)
    area = roi_area_px2 * m_per_px**2
    if scale.pavement_area_m2 is not None and not math.isclose(area, scale.pavement_area_m2, rel_tol=.02):
        raise workflow.WorkflowError("Physical scale, pavement area and ROI disagree by more than 2%.")
    return {
        "status": "known", "reported_units": "metric",
        "m_per_px": m_per_px, "pavement_area_m2": area,
        "provided_pavement_area_m2": scale.pavement_area_m2,
        "station_map_path": scale.station_map_path,
        "scale_source": "provided_gsd" if scale.m_per_px else "derived_from_pavement_area_and_pixel_roi",
    }


_view_process = None
_view_lock = threading.Lock()


def close_view_process():
    global _view_process
    if _view_process is not None:
        _view_process.terminate()
        try:
            _view_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _view_process.kill()
            _view_process.wait()
        _view_process = None


atexit.register(close_view_process)


def worker(action, run, payload):
    executable = os.environ.get("PCI_IMAGE_PYTHON") or sys.executable
    python = shutil.which(executable)
    if not python:
        raise workflow.WorkflowError("Set PCI_IMAGE_PYTHON to a Python with the packages in requirements-images.txt.")
    if action == "views" and os.environ.get("PCI_PERSISTENT_VIEWS") == "1":
        global _view_process
        with _view_lock:
            if _view_process is None or _view_process.poll() is not None:
                _view_process = subprocess.Popen(
                    [python, "-B", str(workflow.ROOT / "image_worker.py"), "--views-stream"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, encoding="utf-8", bufsize=1)
            process = _view_process
            process.stdin.write(json.dumps({"folder": str(run), "payload": payload}, allow_nan=False) + "\n")
            process.stdin.flush()
            if not select.select([process.stdout], [], [], 600)[0]:
                close_view_process()
                raise workflow.WorkflowError(f"Views timed out. Run folder: {run}")
            line = process.stdout.readline()
            if not line:
                close_view_process()
                raise workflow.WorkflowError(f"Persistent view worker exited. Run folder: {run}")
            response = json.loads(line)
            if not response["ok"]:
                raise workflow.WorkflowError(response["error"])
            return response["result"]
    command = [str(workflow.ROOT / "image_worker.py"), action, str(run)]
    if action == "intervals":
        command = [str(workflow.ENGINE.parent / "mask_intervals.py")]
    try:
        process = subprocess.run([python, "-B", *command],
                                 input=json.dumps(payload, allow_nan=False), text=True, encoding="utf-8",
                                 capture_output=True, check=False, timeout=600)
    except subprocess.TimeoutExpired:
        raise workflow.WorkflowError(f"{action.capitalize()} timed out. Run folder: {run}") from None
    if process.returncode:
        raise workflow.WorkflowError(f"{process.stderr.strip()[-1800:]} Run folder: {run}")
    return json.loads(process.stdout)


def prepare_sample(run, assembly, scale, sample_unit, road_mask=None):
    meta = worker("prepare", run, {"image_path":assembly["image_path"], "masks":[], "roi_path":road_mask,
                                  "station_map":scale.station_map_path})
    # Full-image dimensions refer to the first original, before mosaicking or cropping.
    original_width, original_height = assembly["first_source_size"]
    dimensions = [length / pixels for length, pixels in
                  ((scale.image_width_m, original_width), (scale.image_height_m, original_height)) if length is not None]
    if dimensions:
        if scale.station_map_path:
            raise ValueError("Use station mapping or a uniform full-image dimension for calibration.")
        gsd = scale.m_per_px or dimensions[0]
        if any(not math.isclose(gsd, value, rel_tol=.02) for value in dimensions):
            raise ValueError("Full-image dimensions disagree with the uniform pixel scale.")
        scale = scale.model_copy(update={"m_per_px":gsd})
    if scale.station_map_path and scale.m_per_px is None:
        scale = scale.model_copy(update={"m_per_px":abs(meta["source_transform"][0])})
    calibration=scale.model_copy(update={"pavement_area_m2":None}) if scale.station_map_path else scale
    info = check_scale(calibration, meta.get("pavement_area_px2"))
    if scale.station_map_path:
        info["pavement_area_m2"] = scale.pavement_area_m2 or meta["mapped_pavement_area_m2"]
        info["provided_pavement_area_m2"] = scale.pavement_area_m2
        info["mapped_pavement_area_m2"] = meta["mapped_pavement_area_m2"]
        info["scale_source"] = "station_to_UTM_and_sample_area"
    if dimensions:
        info.update(image_width_m=scale.image_width_m, image_height_m=scale.image_height_m,
                    scale_source="full_image_dimensions_checked")
    if sample_unit is not None:
        meta["sample_unit"] = resolve_sample_unit(sample_unit, info, meta["pavement_area_px2"])
    evidence = []
    for entry in meta["images"]:
        path = run / entry["path"]
        evidence.append({"role": entry["role"], "path": str(path), "sha256": entry["sha256"]})
    meta.update(mode="detect", assembly=assembly, scale=info, official_pci=False, run_id=run.name,
                evidence_images=evidence, workflow="sample_unit_detector_pci")
    meta["native_review_required"] = max(meta["width"],meta["height"]) > 1600
    meta["window_workflow"] = (
        "For large samples, use read_pavement_windows in batches, inspect every native view, and "
        "save_pavement_window_reviews for each batch. Finish all planned views before final scoring. "
        "768px views show fine cracks; 1536px views show area context at 768px display resolution. "
        "Use detector instance IDs and cropped SU coordinates for every window.")
    meta["status"] = "awaiting_detector"
    return meta


def load_run(run_id, mode):
    if len(run_id) != 32 or any(ch not in "abcdef0123456789" for ch in run_id):
        raise workflow.WorkflowError("Use the run_id returned by this agent's preparation tool.")
    run = workflow.RUNS / run_id
    file = run / "prepared.json"
    if not file.is_file():
        raise workflow.WorkflowError("Prepared evidence for this run could not be found.")
    meta = json.loads(file.read_text(encoding="utf-8"))
    if mode is not None and meta["mode"] != mode:
        raise workflow.WorkflowError("Use the scoring tool for the preparation workflow.")
    return run, meta
