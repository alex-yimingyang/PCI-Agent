"""Local Paper2Agent-style MCP server for reproducible road asphalt PCI."""

import argparse
from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import BaseModel, Field

import image_agents as vision
import image_detection as detection
import pci_workflow as workflow
import mask_rgb
from condition import interval_name

Grade = Literal["L", "M", "H"]
Unknown = Literal["scenarios", "error"]
CurvePolicy = Literal["approximate", "strict"]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
INSTRUCTIONS = (
    "M1=image direct visual estimate, M2=detect from RGB, M3=mask unlabeled SAM3, M4=labeled masks. "
    "Report the PAVER condition at each PCI endpoint. "
    "Before each new image/mask sample, ask whether physical scale is available if the user has not answered. "
    "Use the provided scale or explicitly send status=unavailable. Inspect the prepared tool images yourself "
    "For large images read_pavement_windows and save_pavement_window_reviews until native coverage is complete. "
    "and complete the corresponding score tool. For masks provide severity bounds per instance "
    "and report the searched PCI interval or pixel proxy interval with search status. "
    "Measured CSVs use road_pci.py. Preserve the source and uncertainty; never fabricate evidence."
)
parser = argparse.ArgumentParser()
parser.add_argument("--agent", choices=tuple(vision.MODES))
AGENT = parser.parse_args().agent if __name__ == "__main__" else None
if AGENT:
    route = {
        "image": "Use prepare_image_direct_agent, inspect the original and record_image_direct_pci.",
        "mask": "Use prepare_unlabeled_mask_agent; infer classes from masks only and score_unlabeled_mask_pci. After measurement use read_mask_grading_crops for aligned RGB surface grading.",
        "labeled": "Use prepare_labeled_mask_agent, keep supplied classes and grade every instance. Review suspected false positives after grading. Score both inventories when requested.",
        "detect": "Use prepare_image_detection_agent. Detect instances and classes yourself from original RGB, "
                  "submit centerline/affected-region geometry, inspect measured crops, grade each instance, "
                  "then score_image_detected_pci through road_pci.py. No SAM3/RF-DETR or direct PCI guessing.",
    }[AGENT]
    INSTRUCTIONS = f"You are the {vision.MODES[AGENT]} PCI agent. {route} " + INSTRUCTIONS
mcp = FastMCP("road-pci", instructions=INSTRUCTIONS, strict_input_validation=True)


class Distress(BaseModel):
    kind: Literal["linear_crack", "alligator", "block_crack", "patch", "pothole"]
    quantity: Positive
    unit: Literal["m", "m2", "count"]
    severity: Grade | None = Field(default=None, description="Observed severity L/M/H. Leave empty when unknown.")
    severity_source: str | None = Field(default=None, description="Source of an explicitly supplied severity.")


def execute(function, *args):
    try:
        return function(*args)
    except (OSError, ValueError) as exc:
        raise ToolError(str(exc)) from None


def image_tool(mode, function, *args):
    if AGENT is not None and AGENT != mode:
        raise ToolError("Use the tools exposed for the selected agent.")
    result = execute(function, *args)
    if isinstance(result,dict):
        for key,value in list(result.items()):
            if isinstance(value,dict) and ('pci_interval' in key or 'pci_proxy_interval' in key or key in ('retain','exclude')) and {'lower','upper'} <= value.keys():
                result[key+'_condition'] = interval_name(value)
    return result


@mcp.tool
def read_pavement_windows(run_id: str, cursor: Annotated[int, Field(ge=0)] = 0,
                          limit: Annotated[int, Field(ge=1,le=4)] = 2) -> ToolResult:
    """Read a bounded batch of native 768/1536 overlapping windows. Inspect the returned RGB/mask
    evidence and save reviews, then follow next_cursor until the full sample is reviewed.
    """
    return execute(vision.read_windows,run_id,cursor,limit)


@mcp.tool
def save_pavement_window_reviews(run_id: str,
                                  reviews: Annotated[list[vision.WindowReview],Field(min_length=1,max_length=4)],
                                  summarize: bool = True) -> dict:
    """Checkpoint inspected windows with notes and optional per-mask class and grade observations.
    Repeating a view ID updates its review. Returns coverage and centrally weighted class consensus.
    """
    return execute(vision.save_reviews,run_id,reviews,summarize)


@mcp.tool
def prepare_image_direct_agent(image_path: str, physical_scale: vision.Scale | None = None,
                                 pavement_roi_mask: str | None = None) -> ToolResult:
    """Send the original pavement photo to the conversational vision model for direct PCI prediction.
    Ask about physical scale first unless already answered. Use unavailable to work from image pixels.
    """
    return image_tool("image", vision.prepare, "image", physical_scale, image_path,None,pavement_roi_mask)


@mcp.tool
def record_image_direct_pci(run_id: str, pci_lower: vision.PCI, pci_upper: vision.PCI,
                            evidence: Annotated[str, Field(min_length=1, max_length=1800)],
                            lower_evidence: Annotated[str, Field(min_length=1,max_length=1200)],
                            upper_evidence: Annotated[str, Field(min_length=1,max_length=1200)],
                            pci: vision.PCI | None = None) -> dict:
    """Record a visual PCI plausibility interval, endpoint evidence and an optional point inside it.
    This is a subjective image-based interval, not a calibrated statistical confidence interval.
    """
    return image_tool("image", vision.record_direct, run_id, pci_lower,pci_upper,evidence,lower_evidence,upper_evidence,pci)


@mcp.tool
def prepare_image_detection_agent(image_path: str, physical_scale: vision.Scale | None = None,
                                    pavement_roi_mask: str | None = None,
                                    pavement_polygon: list[detection.Point] | None = None) -> ToolResult:
    """M2: original RGB only, conversational model detects/classifies individual distresses.
    Ask for physical scale unless already answered; unavailable gives pixel measurements and a PCI proxy.
    Inspect native windows; do not import SAM3/RF-DETR distress masks or archived instance labels.
    Model-draw a pavement_polygon in global original pixels, or supply pavement_roi_mask, to exclude
    shoulder/background from the density denominator. Without one, valid image pixels are assumed pavement.
    """
    return image_tool("detect",detection.prepare,image_path,physical_scale,pavement_roi_mask,pavement_polygon)


@mcp.tool
def measure_image_detected_distresses(run_id: str, detections: list[detection.ImageDetection],
                                        confirmed_no_distress: bool = False,
                                        absence_evidence: str | None = None) -> dict:
    """Measure model-drawn original-pixel centerlines/actual affected-region polygons.
    Bounding boxes alone cannot supply area. Keep global unique instance IDs across overlapping windows.
    Returns quantities/densities; next inspect every RGB/annotation crop and grade per instance.
    An empty list requires explicit no-distress evidence and confirmation.
    """
    return image_tool("detect",detection.measure_detections,run_id,detections,confirmed_no_distress,absence_evidence)


@mcp.tool
def read_image_detection_crops(measurement_run_id: str, cursor: Annotated[int,Field(ge=0)] = 0,
                                limit: Annotated[int,Field(ge=1,le=4)] = 4) -> ToolResult:
    """Inspect paired original RGB/model geometry for every detected instance before final grading.
    Follow next_cursor; rerun measurement with corrected detections if boundaries or classes are wrong.
    """
    return image_tool("detect",detection.read_crops,measurement_run_id,cursor,limit)


@mcp.tool
def score_image_detected_pci(measurement_run_id: str, judgments: list[detection.DetectionGrade],
                              inventory_review_evidence: Annotated[str,Field(min_length=1,max_length=3000)]) -> dict:
    """Calculate rule-based PCI through the actual road_pci.py CLI, using individual severity ranges.
    Judgment mask_id is the model's instance_id. Document geometry review, RGB severity and exclusions
    or explicit surface assumptions. Exact grade enumeration up to 256 combinations; larger searches
    report conservative outer bounds checked against CLI-scored feasible mixed-grade assignments.
    Intervals condition on detected instances and drawn geometry; uncertain classes remain preferred-only.
    """
    return image_tool("detect",detection.score_detections,measurement_run_id,judgments,inventory_review_evidence)


@mcp.tool
def prepare_unlabeled_mask_agent(masks: Annotated[list[vision.UnlabeledMask], Field(min_length=1)],
                                  physical_scale: vision.Scale | None = None,
                                  pavement_roi_mask: str | None = None) -> ToolResult:
    """Prepare masks alone for the conversational model to determine types and grades.
    Ask about physical scale first unless already answered. Optional ROI is a binary pavement mask.
    """
    return image_tool("mask", vision.prepare, "mask", physical_scale, None, masks, pavement_roi_mask)


@mcp.tool
def read_mask_grading_crops(measurement_run_id: str, original_image_path: str,
                            cursor: Annotated[int,Field(ge=0)]=0,
                            limit: Annotated[int,Field(ge=1,le=4)]=4) -> ToolResult:
    """Freeze measured mask-only classes before viewing aligned RGB crops for severity.
    Pass the measurement result's class_search_run_id if present, otherwise its run_id.
    Follow next_cursor. Submit grades in frozen row order using the original preparation run_id."""
    return image_tool("mask",mask_rgb.read_crops,measurement_run_id,original_image_path,cursor,limit)


@mcp.tool
def score_unlabeled_mask_pci(run_id: str, judgments: Annotated[list[vision.Judgment], Field(min_length=1)],
                             measure_only: bool = False) -> dict:
    """First use measure_only=true with inferred types; then grade using the masks and measurements.
    Search the independent per-instance severity intervals. Return calibrated PCI bounds or pixel proxy bounds.
    """
    return image_tool("mask", vision.score_masks, "mask", run_id, judgments, measure_only)


@mcp.tool
def prepare_labeled_mask_agent(masks: Annotated[list[vision.LabeledMask], Field(min_length=1)],
                                physical_scale: vision.Scale | None = None,
                                pavement_roi_mask: str | None = None,
                                original_image_path: str | None = None) -> ToolResult:
    """Prepare class-labeled masks for visual severity assessment. The optional aligned RGB can inform grade.
    Ask about physical scale first unless already answered. Optional ROI is a binary pavement mask.
    """
    return image_tool("labeled", vision.prepare, "labeled", physical_scale, original_image_path, masks, pavement_roi_mask)


@mcp.tool
def score_labeled_mask_pci(run_id: str, judgments: Annotated[list[vision.Judgment], Field(min_length=1)],
                           measure_only: bool = False) -> dict:
    """First use measure_only=true with the supplied classes; then grade using the evidence and measurements.
    Search the independent per-instance severity intervals. Return calibrated PCI bounds or pixel proxy bounds.
    """
    return image_tool("labeled", vision.score_masks, "labeled", run_id, judgments, measure_only)


@mcp.tool
def calculate_road_pci(sample_id: str, area_m2: Positive, distresses: list[Distress],
                       confirmed_no_distress: bool = False, unknown: Unknown = "scenarios",
                       curve_policy: CurvePolicy = "approximate") -> dict:
    """Calculate one asphalt road sample from measured lengths, areas and counts.
    Use m for linear cracks, m2 for alligator/block/patch, and count for potholes.
    Return PCI and audit files.
    """
    rows = []
    for source in distresses:
        row = source.model_dump()
        row["distress"] = row.pop("kind")
        rows.append(row)
    return execute(workflow.calculate_values, sample_id, area_m2, rows,
                   confirmed_no_distress, unknown, curve_policy)


@mcp.tool
def calculate_road_pci_csv(samples_csv: str, inventory_csv: str, unknown: Unknown = "scenarios",
                           curve_policy: CurvePolicy = "approximate") -> dict:
    """Score local measured inventories in batch. Pass an absolute path to samples.csv
    (sample_id, area_m2) and inventory.csv (sample_id, distress, severity, quantity, unit).
    Inputs are copied into the run audit and the output includes a full results CSV.
    """
    return execute(workflow.calculate_csv, samples_csv, inventory_csv, unknown, curve_policy)


@mcp.tool
def explain_pci(run_id: str, sample_id: str, scenario: Grade = "M") -> dict:
    """Return the original engine's densities, deduct values, corrected deduct iterations,
    and severity evidence for a scored sample.
    """
    return execute(workflow.explain, run_id, sample_id, scenario)


@mcp.resource("pci://method", mime_type="text/markdown")
def method_notes() -> str:
    """Public guide for installation, inputs, methods and scoring."""
    return (workflow.ROOT / "README.md").read_text(encoding="utf-8")


if __name__ == "__main__":
    if AGENT:
        keep = {
            "image": {"prepare_image_direct_agent", "record_image_direct_pci"},
            "mask": {"read_mask_grading_crops","prepare_unlabeled_mask_agent", "score_unlabeled_mask_pci"},
            "labeled": {"prepare_labeled_mask_agent", "score_labeled_mask_pci"},
            "detect": {"prepare_image_detection_agent", "measure_image_detected_distresses",
                       "read_image_detection_crops", "score_image_detected_pci"},
        }[AGENT]
        keep |= {"read_pavement_windows","save_pavement_window_reviews"}
        mcp.enable(only=True, names=keep)
    mcp.run(transport="stdio", show_banner=False)
