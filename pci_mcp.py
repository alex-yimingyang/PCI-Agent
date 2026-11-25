"""Local Paper2Agent-style MCP server for reproducible road asphalt PCI."""

from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from pydantic import Field

import image_agents as vision
import image_detection as detection
import detector_stage
from condition import interval_name

mcp = FastMCP("road-pci", instructions=detector_stage.INSTRUCTIONS, strict_input_validation=True)


def execute(function, *args):
    try:
        return function(*args)
    except (OSError, ValueError) as exc:
        raise ToolError(str(exc)) from None


def image_tool(function, *args):
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
                                  reviews: Annotated[list[vision.WindowReview],Field(min_length=1,max_length=4)]) -> dict:
    """Checkpoint visual inspection of each native RGB window.
    Repeating a view ID updates its review. Returns native-window coverage.
    """
    return execute(vision.save_reviews,run_id,reviews)


@mcp.tool
def prepare_sample_unit(images: Annotated[list[vision.SourceImage],Field(min_length=1)],
                        model_address: str | None = None, physical_scale: vision.Scale | None = None,
                        sample_unit: vision.SampleUnit | None = None,
                        bounds_xyxy: Annotated[list[int],Field(min_length=4,max_length=4)] | None = None,
                        pavement_roi_mask: str | None = None,
                        pavement_polygon: Annotated[list[Annotated[list[Annotated[float,Field(allow_inf_nan=False)]],Field(min_length=2,max_length=2)]],Field(min_length=3)] | None = None,
                        confidence: Annotated[float,Field(gt=0,lt=1)] = 0.25) -> ToolResult:
    """Assemble and crop one SU and return original RGB for automatic travel-lane delineation.
    Pixel offsets and half-open crop bounds are on the common grid. A ROI mask covers the uncropped
    mosaic extent. Grade subsequent detections in cropped SU coordinates. Use mode=mosaic for a
    confirmed multi-image SU, or provided for a known total SU area. Ask for missing coverage.
    """
    return image_tool(detector_stage.detect,images,physical_scale,pavement_roi_mask,pavement_polygon,
                      model_address,confidence,sample_unit,bounds_xyxy)


@mcp.tool
def read_carriageway_detail(run_id: str, bounds_xyxy: Annotated[list[int],Field(min_length=4,max_length=4)]) -> ToolResult:
    """Inspect a detailed original RGB window for semantic road/shoulder segmentation."""
    return execute(detector_stage.road_detail,run_id,bounds_xyxy)


@mcp.tool
def segment_carriageway_and_detect(run_id: str, extent: Literal["mixed","all_road","no_road","uncertain"],
                                   evidence: Annotated[str,Field(min_length=1,max_length=1800)],
                                   regions: list[detector_stage.RoadRegion] | None = None) -> ToolResult:
    """Vision client: delineate travel lanes from RGB, exclude background, then run the selected detector.
    In mixed scenes supply one or more road polygons in native cropped SU coordinates, with optional holes.
    Use the original prepare_sample_unit run ID for a corrected delineation.
    """
    return execute(detector_stage.segment,run_id,extent,regions or [],evidence)


@mcp.tool
def read_detector_candidates(run_id: str, cursor: Annotated[int, Field(ge=0)] = 0,
                             limit: Annotated[int, Field(ge=1,le=4)] = 4) -> ToolResult:
    """Show original and detector-marked RGB per ID. Follow next_cursor and review every candidate."""
    return image_tool( detector_stage.read_candidates, run_id, cursor, limit)


@mcp.tool
def measure_image_detected_distresses(run_id: str, detections: list[detection.ImageDetection],
                                        confirmed_no_distress: bool = False,
                                        absence_evidence: str | None = None,
                                        dismissed_candidates: dict[str, str] | None = None) -> dict:
    """Measure model-reviewed SU-pixel centerlines/actual affected-region polygons.
    Bounding boxes alone cannot supply area. Keep global unique instance IDs across overlapping windows.
    Returns quantities/densities; next inspect every RGB/annotation crop and grade per instance.
    Account for every detector ID with traced geometry or a dismissal reason. An empty inventory requires no-distress evidence.
    """
    return image_tool(detection.measure_detections,run_id,detections,confirmed_no_distress,absence_evidence,dismissed_candidates)


@mcp.tool
def read_image_detection_crops(measurement_run_id: str, cursor: Annotated[int,Field(ge=0)] = 0,
                                limit: Annotated[int,Field(ge=1,le=4)] = 4) -> ToolResult:
    """Inspect paired original RGB/model geometry for every detected instance before final grading.
    Follow next_cursor; rerun measurement with corrected detections if boundaries or classes are wrong.
    """
    return image_tool(detection.read_crops,measurement_run_id,cursor,limit)


@mcp.tool
def score_image_detected_pci(measurement_run_id: str, judgments: list[detection.DetectionGrade],
                              inventory_review_evidence: Annotated[str,Field(min_length=1,max_length=3000)]) -> dict:
    """Calculate rule-based PCI through the actual road_pci.py CLI, using individual severity ranges.
    Judgment mask_id is the model's instance_id. Document geometry review, RGB severity and exclusions
    or explicit surface assumptions. Exact grade enumeration up to 256 combinations; larger searches
    report conservative outer bounds checked against CLI-scored feasible mixed-grade assignments.
    Intervals condition on detected instances and drawn geometry; uncertain classes remain preferred-only.
    """
    return image_tool(detection.score_detections,measurement_run_id,judgments,inventory_review_evidence)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
