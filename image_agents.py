"""Stage evidence for the conversational model, then score validated judgments."""

import base64
import atexit
from collections import defaultdict
from itertools import product
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
Semantics = Literal["unspecified", "crack_trace", "affected_region"]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
PCI = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Grade = Literal["L", "M", "H"]
GRADES = ("L", "M", "H")
MODES = {"image": "original_image", "mask": "unlabeled_mask", "labeled": "labeled_mask",
         "detect": "original_image_detection_rule_pci"}
REFERENCE_AREA_M2 = 5000 * .3048**2
SCALE_QUESTION = "Is a physical scale available? Supply meters per pixel, millimeters per pixel, or the valid pavement area in square meters. With no scale, I will use pixel units."
RULES = (
    "Judge only what the supplied evidence supports. Linear crack: isolated longitudinal/transverse; "
    "L is fine/intact, M has opening or minor secondary cracking, H has severely broken edges or "
    "strong secondary cracking. Alligator: small interconnected fatigue cells; distinguish it from "
    "larger rectangular block cracking. Grade network development, spalling and edge condition "
    "where visible. Patch requires surface evidence; pothole depth cannot be seen in a binary mask. "
    "Give severity_lower and severity_upper per instance. Use equal endpoints for a fixed grade, "
    "L–M or M–H only when instance-specific evidence supports both and excludes the other grade. "
    "Supply severity_basis, severity_rationale and excluded_grade_evidence for final grading. "
    "Omit grades when evidence is insufficient; unknown severity is ungradable, not an automatic L–H interval. "
    "When the user explicitly requests surface-based estimation despite insufficient evidence, inspect each "
    "RGB crop and submit forced_surface_estimate=true, surface_estimate_assumptions, visual_confidence=low "
    "or very_low, and a preferred_severity within the individually chosen range. Such ranges are "
    "uncalibrated plausibility assumptions, not evidence excluding other grades. Do not apply a blanket range. "
    "L-H is allowed only as an explicitly forced unresolved range and must not be called credible or narrowed. "
    "Separate the crack family from the PCI subtype. If cracking is recognizable but its subtype is uncertain, "
    "use kind=crack, give a preferred_crack_kind, instance-specific class_rationale and class_candidates. "
    "Do not discard that instance as unknown just because linear/alligator/block is uncertain. "
    "Infer these classes from mask morphology in the unlabeled route; RGB access for severity does not change that boundary. "
    "Select candidates individually, never assume every crack is linear. Use unknown only when even the distress "
    "family is unrecognizable or the target may be a non-distress artifact. "
    "A sparse crack-line mask uses crack_trace, a filled affected area uses affected_region. "
    "Use disjoint normalized 0–1000 [x0,y0,x1,y1] boxes for distinct instances in a combined mask, "
    "including instances of the same class that need independent grades. "
    "Never use crack extent alone for grade. Treat mask width as a proxy and use unknown grades when needed. "
    "Do not estimate PCI here."
)


class Scale(BaseModel):
    status: Literal["known", "unavailable"]
    m_per_px: Positive | None = None
    pavement_area_m2: Positive | None = None
    station_map_path: str | None = Field(default=None, description="Optional CAE station-to-UTM CSV for the aligned TIFF.")


class UnlabeledMask(BaseModel):
    path: str
    semantics: Semantics = "unspecified"
    bounds_xyxy: Annotated[list[int], Field(min_length=4,max_length=4)] | None = Field(
        default=None, description="Full-image pixel bounds for a cropped instance mask. Supply a full ROI or aligned original.")


class LabeledMask(UnlabeledMask):
    kind: Kind | Literal["alligator_crack"]


class Judgment(BaseModel):
    mask_id: str
    kind: Kind | Literal["crack", "unknown"] | None = None
    preferred_crack_kind: CrackKind | None = Field(default=None,
        description="Best per-instance PCI subtype when kind=crack. Linear uses length; alligator/block use affected area.")
    class_candidates: list[CrackKind] = Field(default_factory=list,
        description="Plausible crack subtypes based on this instance's mask, including the preferred subtype; not a blanket sensitivity set.")
    class_rationale: Annotated[str, Field(min_length=1,max_length=1800)] | None = None
    severity: Grade | None = Field(default=None, description="Fixed grade shorthand. Equivalent to equal endpoints.")
    severity_lower: Grade | None = Field(default=None, description="Lowest admissible grade for this instance.")
    severity_upper: Grade | None = Field(default=None, description="Highest admissible grade for this instance.")
    evidence: Annotated[str, Field(min_length=1, max_length=1200)]
    semantics: Semantics = "unspecified"
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
        if self.kind != "crack" and (self.class_candidates or self.preferred_crack_kind is not None):
            raise ValueError("Submit uncertain crack subtypes with kind=crack so candidate geometry and scoring are searched explicitly.")
        if self.kind == "crack":
            if (self.preferred_crack_kind is None or not self.class_candidates or not self.class_rationale
                    or self.preferred_crack_kind not in self.class_candidates):
                raise ValueError("A crack-family instance requires a preferred subtype, plausible class_candidates and an individual class_rationale.")
            if len(set(self.class_candidates)) != len(self.class_candidates):
                raise ValueError("Crack subtype candidates must be distinct.")
            if any(k in ("alligator","block_crack") for k in self.class_candidates) and self.semantics=="unspecified":
                raise ValueError("Crack-family area candidates require explicit crack_trace or affected_region semantics.")
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
    judgments: list[Judgment] = Field(default_factory=list, description="Optional per-mask class and grade observations for this view.")


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


def window_consensus(run,meta):
    observations=defaultdict(list)
    for path in sorted((run/"reviews").glob("*.json")):
        saved=json.loads(path.read_text())
        for decision in saved["judgments"]:
            observations[decision["mask_id"]].append((saved,decision))
    result=[]
    for item in meta["masks"]:
        mask_id=item["mask_id"]
        votes=defaultdict(float)
        n=defaultdict(int)
        ranges=defaultdict(list)
        for saved,decision in observations[mask_id]:
            kind=item["kind"] or decision["kind"]
            if kind not in ("linear_crack","alligator","block_crack","patch","pothole"):
                continue
            needed=768 if kind=="linear_crack" else 1536
            if saved["window"]["size"]!=needed:
                continue
            weight=saved["window"].get("mask_weights",{}).get(mask_id,0)
            votes[kind]+=weight
            n[kind]+=int(weight>0)
            low=decision["severity_lower"] or decision["severity"]
            high=decision["severity_upper"] or decision["severity"]
            if low is not None and high is not None:ranges[kind].append((low,high))
        candidate=max(votes,key=votes.get) if votes else None
        fraction=votes[candidate]/sum(votes.values()) if candidate and sum(votes.values()) else 0.
        minimum=1 if candidate=="linear_crack" else 3
        threshold=.44 if candidate=="linear_crack" else .50
        accepted=bool(candidate and n[candidate]>=minimum and fraction>threshold)
        result.append(dict(mask_id=mask_id,supplied_class=item["kind"],
                           consensus_class=item["kind"] or (candidate if accepted else None),
                           positive_windows=n.get(candidate,0),weighted_support=fraction,
                           status="protected_supplied_anchor" if item["kind"] else "consensus" if accepted else "pending_class_review",
                           severity_lower=None, severity_upper=None,
                           observed_severity_ranges=sorted(set(ranges[candidate])) if candidate else []))
    return result


def save_reviews(run_id,reviews,summarize=True):
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
        for judgment in review.judgments:
            if judgment.mask_id not in row["mask_ids"]:
                raise workflow.WorkflowError("A window judgment must refer to a mask visible in that window.")
            known=next(m["kind"] for m in meta["masks"] if m["mask_id"]==judgment.mask_id)
            if known and judgment.kind not in (None,known):
                raise workflow.WorkflowError(f"Keep the supplied class {known}.")
        target=run/"reviews"
        target.mkdir(exist_ok=True)
        payload={**review.model_dump(),"window":delivery["window"],"image_sha256":delivery["images"]}
        temporary=target/f"{review.view_id}.{uuid4().hex}.tmp"
        workflow.write_json(temporary,payload)
        temporary.replace(target/f"{review.view_id}.json")
        saved.append(review.view_id)
    return {"run_id":run_id,"saved_view_ids":saved,"coverage":check_window_coverage(run,meta,False),
            "instance_consensus":window_consensus(run,meta) if summarize else []}


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
        if scale.m_per_px is not None or scale.pavement_area_m2 is not None or scale.station_map_path:
            raise workflow.WorkflowError("Use status=known if you supply a physical scale.")
        if roi_area_px2 is None:
            return {"status": "unavailable", "reported_units": "px"}
        return {
            "status": "unavailable", "reported_units": "px",
            "reference_sample_area_m2_for_pci_proxy": REFERENCE_AREA_M2,
            "assumed_m_per_px_for_pci_proxy": math.sqrt(REFERENCE_AREA_M2 / roi_area_px2),
        }
    if scale.m_per_px is None and scale.pavement_area_m2 is None and not scale.station_map_path:
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


def prepare(mode, scale, image_path=None, masks=None, roi_path=None):
    instruction = check_scale(scale)
    if scale is None:
        return ToolResult(structured_content=instruction)
    if mode in ("image", "detect") and not image_path:
        raise workflow.WorkflowError("Provide an original image path.")
    if mode not in ("image", "detect") and not masks:
        raise workflow.WorkflowError("Provide at least one mask.")
    run = workflow.new_run()
    specs = [mask.model_dump() for mask in masks or []]
    for spec in specs:
        spec.setdefault("kind", None)
        if spec["kind"] == "alligator_crack":
            spec["kind"] = "alligator"
    meta = worker("prepare", run, {"image_path": image_path, "masks": specs, "roi_path": roi_path,
                                  "station_map": scale.station_map_path})
    if scale.station_map_path and scale.m_per_px is None:
        scale = scale.model_copy(update={"m_per_px":abs(meta["source_transform"][0])})
    calibration=scale.model_copy(update={"pavement_area_m2":None}) if scale.station_map_path else scale
    info = check_scale(calibration, meta.get("pavement_area_px2"))
    if scale.station_map_path:
        info["pavement_area_m2"] = scale.pavement_area_m2 or meta["mapped_pavement_area_m2"]
        info["provided_pavement_area_m2"] = scale.pavement_area_m2
        info["mapped_pavement_area_m2"] = meta["mapped_pavement_area_m2"]
        info["scale_source"] = "station_to_UTM_and_sample_area"
    evidence = []
    for entry in meta["images"]:
        path = run / entry["path"]
        evidence.append({"role": entry["role"], "path": str(path), "sha256": entry["sha256"]})
    meta.update(mode=mode, scale=info, official_pci=False, run_id=run.name,
                evidence_images=evidence, workflow=MODES[mode])
    meta["native_review_required"] = max(meta["width"],meta["height"]) > 1600
    meta["window_workflow"] = (
        "For large samples, use read_pavement_windows in batches, inspect every native view, and "
        "save_pavement_window_reviews for each batch. Finish all planned views before final scoring. "
        "768px views show fine cracks; 1536px views show area context at 768px display resolution. "
        "Use global instance mask IDs and original-image coordinates; never average tile PCI values.")
    workflow.write_json(run / "prepared.json", meta)
    instructions = {
        "image": "Use the original photograph only. Predict a visual plausibility PCI interval on 0–100, with separate "
                 "evidence for each endpoint and explicit uncertainty drivers. A point alone is not sufficient. "
                 "This is not a statistically calibrated confidence interval. "
                 "Use any user-provided scale as context. Ignore shadows, lane markings, and non-pavement. "
                 "With no scale work from visual proportions in pixels. Explain prominent evidence and uncertainty. "
                 "Call record_image_direct_pci with the run_id and your judgment; no DV/CDV lookup for this agent.",
        "detect": "Detect and classify individual distresses yourself from the original RGB only. "
                  "Do not import SAM3/RF-DETR masks, labels or archived grades. Do not directly estimate PCI. "
                  "Use global original-image pixel coordinates. Submit centerline polylines for linear cracks "
                  "and actual affected-region polygons for area distress, never a bounding box as affected area. "
                  "Record per-instance localization/class evidence and geometry limitations. "
                  "Call measure_image_detected_distresses, inspect every resulting RGB/geometry crop with "
                  "read_image_detection_crops, then submit independently justified severity ranges with "
                  "score_image_detected_pci. The local road_pci.py CLI performs deduct/CDV calculations. "
                  "Intervals are conditional on the detected inventory and drawn geometry; missed detections "
                  "and boundary uncertainty are not covered. " + RULES.replace(
                      "Infer these classes from mask morphology in the unlabeled route; RGB access for severity does not change that boundary. ",
                      "In this detection route infer classes and severity from the original RGB. "),
        "mask": RULES + " Infer class from the masks only. Use score_unlabeled_mask_pci with the run_id, "
                "type and semantics decisions, and measure_only=true. Then grade with those measurements "
                "and the original mask evidence and call the same tool with measure_only=false. "
                "Report the resulting PCI bounds and search status.",
        "labeled": RULES + " Keep the supplied class. Call score_labeled_mask_pci with measure_only=true to "
                   "get quantities and densities. Then grade with the measurements and the available images "
                   "and call the same tool with measure_only=false. If provided, use the aligned original for grading. "
                   "Report the resulting PCI bounds and search status.",
    }
    meta["status"] = "needs_visual_assessment"
    meta["model_instructions"] = instructions[mode]
    meta["model_instructions"] += " " + meta["window_workflow"]
    content = [TextContent(type="text", text=json.dumps(meta, ensure_ascii=False))]
    for entry in evidence:
        path = Path(entry["path"])
        content.append(TextContent(type="text", text=entry["role"]))
        content.append(ImageContent(type="image", data=base64.b64encode(path.read_bytes()).decode("ascii"),
                                    mime_type="image/png" if path.suffix == ".png" else "image/jpeg"))
    return ToolResult(content=content, structured_content=meta)


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


def record_direct(run_id, pci_lower, pci_upper, evidence, lower_evidence, upper_evidence, pci=None):
    if not (0<=pci_lower<pci_upper<=100):
        raise workflow.WorkflowError("Direct visual PCI requires ordered distinct lower and upper endpoints on 0–100.")
    if pci is not None and not pci_lower<=pci<=pci_upper:
        raise workflow.WorkflowError("Optional point estimate must lie inside the visual interval.")
    sample_run, meta = load_run(run_id, "image")
    coverage = check_window_coverage(sample_run, meta)
    run = workflow.new_run()
    result = {
        "run_id": run.name, "prepared_run_id": run_id, "workflow": "original_image",
        "pci": pci, "pci_interval":{"lower":pci_lower,"upper":pci_upper},
        "interval_type":"visual_plausibility_interval", "statistically_calibrated":False,
        "lower_evidence":lower_evidence,"upper_evidence":upper_evidence,
        "official_pci": False, "status": "provisional", "score_type": "vision_model_direct_interval",
        "evidence": evidence, "scale": meta["scale"], "image_sha256": meta["image_sha256"],
        "image_dimensions_px": [meta["width"], meta["height"]],
        "prepared_evidence": str(sample_run / "prepared.json"),
        "audit_json": str(run / "direct_pci.json"),
        "native_review": coverage,
    }
    workflow.write_json(run / "direct_pci.json", result)
    return result


def score_masks(mode, run_id, judgments, measure_only=False):
    """Resolve crack-family candidates jointly, remeasuring geometry per assignment.

    A generic crack is not an extra deduct curve and is never counted twice.
    Large candidate searches retain a clearly conditional preferred-class result.
    """
    if mode == "mask":
        from mask_rgb import classes
        prepared,_=load_run(run_id,mode)
        locked=prepared/"grading_rgb.json"
        if locked.is_file() and json.loads(locked.read_text())["classes"]!=classes([j.model_dump() for j in judgments]):
            raise workflow.WorkflowError("Keep the frozen mask-only classes, geometry and row order during RGB grading.")
    ambiguous = [j for j in judgments if j.kind == "crack"]
    if not ambiguous:
        return _score_masks_once(mode, run_id, judgments, measure_only)
    if mode != "mask":
        raise workflow.WorkflowError("Crack-family inference is for the unlabeled route; keep the supplied class in the labeled route.")
    # One complete geometry pass covers all instances, so area/line suppression
    # and duplicate ownership are recalculated when a candidate type changes.
    preferred=tuple(j.preferred_crack_kind for j in ambiguous)
    choice_count=math.prod(len(j.class_candidates) for j in ambiguous)
    scenarios=[preferred]
    search_limit=16
    if not measure_only and choice_count <= search_limit:
        scenarios += [s for s in product(*(j.class_candidates for j in ambiguous)) if s!=preferred]
    outcomes=[]
    for choices in scenarios:
        selected=iter(choices);expanded=[]
        for j in judgments:
            expanded.append(j.model_copy(update={"kind":next(selected)}) if j.kind=="crack" else j)
        outcomes.append(_score_masks_once(mode,run_id,expanded,measure_only))
    result=dict(outcomes[0])
    result["crack_family_instances"]=[dict(mask_id=j.mask_id,box=j.box,kind="crack",
        preferred_crack_kind=j.preferred_crack_kind,class_candidates=j.class_candidates,
        class_rationale=j.class_rationale) for j in ambiguous]
    result["class_search"]={"candidate_assignments":choice_count,"evaluated_assignments":len(outcomes),
        "limit":search_limit,"complete":not measure_only and choice_count==len(outcomes),
        "geometry_recomputed_per_assignment":True,
        "scenarios":[dict(assignment=[dict(mask_id=j.mask_id,box=j.box,kind=k) for j,k in zip(ambiguous,s,strict=True)],
            result_json=o["result_json"],pci_interval=o.get("pci_interval"),pci_proxy_interval=o.get("pci_proxy_interval"),
            conditional_pci_interval=o.get("conditional_pci_interval"),conditional_pci_proxy_interval=o.get("conditional_pci_proxy_interval"))
            for s,o in zip(scenarios,outcomes,strict=True)]}
    result["class_assessment_policy"]="crack_family_with_individual_subtype_candidates"
    result["measurement_scope"]="Displayed measurements and audit_files refer to the preferred subtype assignment; each candidate assignment has its own complete geometry/scoring audit in class_search.scenarios."
    if not measure_only:
        searches=[o["search"] for o in outcomes if "search" in o]
        result["search"]={"method":"geometry_scenarios_with_severity_search",
            "converged":result["class_search"]["complete"] and len(searches)==len(outcomes) and all(s["converged"] for s in searches),
            "boxes_evaluated":sum(s["boxes_evaluated"] for s in searches),
            "endpoint_search_gap_pci":max((s["endpoint_search_gap_pci"] for s in searches),default=0.) if result["class_search"]["complete"] else None,
            "scope":"Worst severity-search endpoint gap across all candidate class assignments; no class gap claimed if candidate search incomplete."}
        for name in ("pci_interval","pci_proxy_interval","conditional_pci_interval","conditional_pci_proxy_interval"):
            q=result.get(name)
            if q is None:continue
            result["preferred_class_"+name]=q
            if result["class_search"]["complete"] and all(o.get(name) is not None for o in outcomes):
                result[name]={"lower":min(o[name]["lower"] for o in outcomes),
                              "upper":max(o[name]["upper"] for o in outcomes)}
            else:
                result[name]=None
        result["interval_type"]="candidate_class_and_severity_pci_outer_interval"
        if not result["class_search"]["complete"]:
            result["status"]="preferred_class_estimate_only"
            result["class_search"]["limitation"]="Candidate combinations exceed the 16-assignment geometry budget. Only preferred-class conditional PCI is available; no complete class-uncertainty envelope is claimed."
        # Geometry may make a scenario empty/partial; never substitute a full
        # interval from the preferred scenario for the whole candidate set.
        elif not any(result.get(k) for k in ("pci_interval","pci_proxy_interval")):
            result["status"]="candidate_inventory_requires_review"
    aggregate=workflow.new_run()
    workflow.write_json(aggregate/"family_judgments.json",[j.model_dump() for j in judgments])
    result["preferred_scenario_result_json"]=outcomes[0]["result_json"]
    result["result_json"]=str(aggregate/"class_pci.json")
    result["family_judgments_json"]=str(aggregate/"family_judgments.json")
    # Keep run_id tied to the preferred scenario's audit files, while the
    # aggregate has its own explicit identifier.
    result["class_search_run_id"]=aggregate.name
    workflow.write_json(aggregate/"class_pci.json",result)
    return result


def _score_masks_once(mode, run_id, judgments, measure_only=False):
    from engine.pci_core import density

    prepared, meta = load_run(run_id, mode)
    coverage = check_window_coverage(prepared, meta, require_complete=not measure_only)
    scale = meta["scale"]
    pixel_area = meta["pavement_area_px2"]
    metric = scale["status"] == "known"
    scoring_scale = scale["m_per_px"] if metric else scale["assumed_m_per_px_for_pci_proxy"]
    scoring_area = scale["pavement_area_m2"] if metric else scale["reference_sample_area_m2_for_pci_proxy"]
    decisions = [row.model_dump() for row in judgments]
    supplied_kinds = {item["mask_id"]: item["kind"] for item in meta["masks"]}
    consensus={r["mask_id"]:r for r in window_consensus(prepared,meta)}
    applied = []
    for row in decisions:
        agreed=consensus.get(row["mask_id"],{})
        if row["kind"] is None and supplied_kinds.get(row["mask_id"]) is None:
            row={**row,"kind":agreed.get("consensus_class") or "unknown"}
        fixed = row["severity"]
        low = row["severity_lower"] or fixed
        high = row["severity_upper"] or fixed
        # Window observations inform review; they never silently broaden a final instance judgment.
        source = (row.get("severity_basis") or "vision_agent") if low is not None else "ungradable"
        if measure_only:
            low = high = None
            source = "pending_visual_grading"
        elif low is not None:
            if row["forced_surface_estimate"] and not meta.get("original_source") and not (prepared/"grading_rgb.json").exists():
                raise workflow.WorkflowError(f"{row['mask_id']}: forced surface estimation requires an aligned original RGB image.")
            if not row.get("severity_rationale") or row.get("severity_basis") in (None,"ungradable"):
                raise workflow.WorkflowError(f"{row['mask_id']}: final severity requires a basis and instance-specific rationale.")
            excluded=set(GRADES)-set(GRADES[GRADES.index(low):GRADES.index(high)+1])
            if not row["forced_surface_estimate"] and any(not row["excluded_grade_evidence"].get(grade,"").strip() for grade in excluded):
                raise workflow.WorkflowError(f"{row['mask_id']}: explain the evidence excluding every grade outside the interval.")
        applied.append({**row, "severity_lower": low, "severity_upper": high, "severity_source": source})
    measured = worker("measure", prepared, {"meta": meta, "scoring_scale": scoring_scale,
                                            "judgments": applied,"station_map":scale.get("station_map_path")})
    run = workflow.new_run()
    inventory = []
    measurements = []
    pixel_groups = {}
    ungradable=[]
    for i, row in enumerate(measured["rows"], 1):
        kind = row["kind"]
        record_id = f"{row['mask_id']}_{i}"
        low, high = row["severity_lower"], row["severity_upper"]
        grade = low if low == high else None
        quantity = row["length_px"] if kind == "linear_crack" else row["count"] if kind == "pothole" else row["measured_area_px2"]
        base_unit = "px" if kind == "linear_crack" else "count" if kind == "pothole" else "px2"
        units = {"px":"m", "count":"count", "px2":"m2"}
        bases = {"px":"100 px/px2", "px2":"percent", "count":"count/100 px2"}
        engine_qty = quantity * {"px":scoring_scale, "count":1, "px2":scoring_scale**2}[base_unit]
        if metric and scale.get("station_map_path"):
            engine_qty = row["mapped_length_m"] if kind == "linear_crack" else row["count"] if kind == "pothole" else row["mapped_area_m2"]
        raw_density = 100 * quantity / pixel_area
        engine_density = density(engine_qty, units[base_unit], scoring_area) if row["scored"] else None
        row = {**row, "record_id":record_id, "severity":grade,
               "severity_assessed":low is not None,
               "severity_rationale":applied[i-1].get("severity_rationale"),
               "excluded_grade_evidence":applied[i-1].get("excluded_grade_evidence"),
               "forced_surface_estimate":applied[i-1]["forced_surface_estimate"],
               "surface_estimate_assumptions":applied[i-1]["surface_estimate_assumptions"],
               "visual_confidence":applied[i-1]["visual_confidence"],
               "preferred_severity":applied[i-1]["preferred_severity"],
               "class_family":"crack" if applied[i-1]["class_candidates"] else None,
               "preferred_crack_kind":applied[i-1]["preferred_crack_kind"],
               "class_candidates":applied[i-1]["class_candidates"],
               "class_rationale":applied[i-1]["class_rationale"],
               "quantity_px_basis":quantity, "quantity_px_unit":base_unit,
               "density_px_basis":raw_density, "density_px_unit":bases[base_unit],
               "density_metric":engine_density if metric else None,
               "density_reference_area_proxy":engine_density if not metric else None,
               "measured_length_m":(row["mapped_length_m"] if scale.get("station_map_path") else row["length_px"] * scoring_scale) if metric else None,
               "measured_area_m2":(row["mapped_area_m2"] if scale.get("station_map_path") else row["measured_area_px2"] * scoring_scale**2) if metric else None,
               "mean_line_width_proxy_px": (row["measured_area_px2"] / row["length_px"]
                                            if row["length_px"] else None)}
        row["mean_line_width_proxy_mm"] = (
            1000 * row["mean_line_width_proxy_px"] * scoring_scale
            if metric and row["mean_line_width_proxy_px"] is not None else None
        )
        measurements.append(row)
        if row["scored"] and low is None:
            ungradable.append(record_id)
        if row["scored"] and low is not None:
            inventory.append({
                "sample_id":"mask_sample", "record_id":record_id, "distress":kind,
                "severity_lower":low, "severity_upper":high, "quantity":engine_qty, "unit":units[base_unit],
                "severity_source":row["severity_source"],
                "quantity_source":"pixel_geometry_with_calibrated_scale" if metric else "reference_area_pci_proxy",
            })
            key = (kind, low, high, bases[base_unit])
            pixel_groups[key] = pixel_groups.get(key, 0.) + raw_density
    workflow.write_csv(run / "measurements.csv", list(measurements[0]), measurements)
    workflow.write_json(run / "judgments.json", decisions)
    result = {
        "run_id": run.name, "prepared_run_id": run_id, "workflow": MODES[mode],
        "official_pci": False, "scale": scale, "pavement_area_px2": pixel_area,
        "assumed_full_canvas": meta["assumed_full_canvas"],
        "unassigned_pixels": measured["unassigned_pixels"],
        "cross_type_overlap_px2": measured["cross_type_overlap_px2"],
        "withheld_measurements": sum(row["withheld"] for row in measurements),
        "density_by_type_and_grade_px_basis": [
            {"kind":kind, "severity_lower":low, "severity_upper":high, "density":value, "unit":unit}
            for (kind, low, high, unit), value in pixel_groups.items()
        ],
        "measurements": measurements[:30], "measurements_csv": str(run / "measurements.csv"),
        "judgments_json": str(run / "judgments.json"), "prepared_json": str(prepared / "prepared.json"),
        "result_json": str(run / "mask_pci.json"),
        "native_review": coverage, "window_consensus":list(consensus.values()),
        "measurement":measured.get("measurement"),
        "pci_interval": None, "pci_proxy_interval": None,
        "ungradable_instances":ungradable,
        "assessment_policy":"instance_evidence_required; missing severity is never replaced by L-H",
        "forced_surface_estimate_count":sum(r["forced_surface_estimate"] for r in applied) if not measure_only else 0,
        "forced_unresolved_range_ids":[r["mask_id"] for r in applied if r["forced_surface_estimate"] and
                                       (r["severity_lower"],r["severity_upper"])==("L","H")] if not measure_only else [],
        "statistically_calibrated":False,
        "score_type":"physical_scale_pci_interval" if metric else "pixel_reference_area_pci_proxy_interval",
    }
    if result["forced_surface_estimate_count"]:
        result["assessment_policy"]="user_requested_per_instance_surface_estimates; explicit assumptions retained"
        result["interval_type"]="surface_assumption_pci_interval"
        result["severity_estimation_warning"]="Includes low-confidence surface estimates, not a calibrated confidence interval or proof excluding other severity grades. Forced L-H records remain unresolved."
    if measure_only:
        result["status"] = "needs_visual_severity_assessment"
    elif ungradable:
        result["status"]="needs_instance_severity_review"
        result["scoring_blocked_reason"]="Some measurable instances lack an evidence-supported severity interval. No full-inventory PCI was calculated."
    elif inventory:
        flag = result["score_type"]
        if result["withheld_measurements"] or result["unassigned_pixels"]:
            flag += " | incomplete_mask_inventory"
        if meta["assumed_full_canvas"]:
            flag += " | whole_canvas_assumed_pavement"
        if measured["cross_type_overlap_px2"]:
            flag += " | cross_type_mask_overlap_needs_review"
        workflow.write_csv(run / "samples.csv", ("sample_id","area_m2","status","flags"),
                           [{"sample_id":"mask_sample","area_m2":scoring_area,
                             "status":"provisional","flags":flag}])
        workflow.write_csv(run / "inventory.csv", list(inventory[0]), inventory)
        bounds = worker("intervals", run, {"pavement_area_m2": scoring_area, "inventory": inventory})
        bounds_file = run / "pci_bounds_details.json"
        workflow.write_json(bounds_file, bounds)
        field = "pci_interval" if metric else "pci_proxy_interval"
        interval={"lower": bounds["lower"], "upper": bounds["upper"]}
        if result["withheld_measurements"] or result["unassigned_pixels"]:
            result["conditional_"+field]=interval
        else:
            result[field]=interval
        result["search"] = {key: bounds[key] for key in (
            "method", "search_stop", "converged", "boxes_evaluated", "endpoint_search_gap_pci",
            "tolerance_pci", "box_limit", "min_witness_ceiling", "max_witness_floor",
        )}
        if "assignments_evaluated" in bounds:
            result["search"]["assignments_evaluated"] = bounds["assignments_evaluated"]
        result["audit_files"] = {
            "pci_bounds_details.json": str(bounds_file),
            "input_samples.csv": str(run / "samples.csv"),
            "input_inventory.csv": str(run / "inventory.csv"),
        }
        incomplete = result["withheld_measurements"] or result["unassigned_pixels"] or measured["cross_type_overlap_px2"]
        result["status"] = "partial_inventory" if incomplete else "provisional"
    else:
        result["status"] = "no_scorable_distress"
    workflow.write_json(run / "mask_pci.json", result)
    return result
