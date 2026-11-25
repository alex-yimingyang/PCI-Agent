"""Local vision evidence preparation and geometry on the original pixel grid."""

import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def prepare(run, payload):
    from native_windows import prepare as prepare_native

    meta = prepare_native(run, payload)
    for entry in meta["images"]:
        entry["sha256"] = file_hash(run / entry["path"])
    if meta.get("original_source"):
        meta["image_sha256"] = file_hash(Path(meta["original_source"]["path"]))
    return meta


def bracket_hits(trace, reach):
    hits = np.zeros(trace.shape, np.uint8)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        span = int(reach / math.hypot(dy, dx))
        if span == 0:
            continue
        kernel = np.zeros((span * dy + 1, span * abs(dx) + 1), np.uint8)
        n = np.arange(span + 1)
        kernel[n * dy, span - n if dx < 0 else n * dx] = 1
        first = (span if dx < 0 else 0, 0)
        last = (kernel.shape[1] - 1 - first[0], kernel.shape[0] - 1)
        hits += cv2.dilate(trace, kernel, anchor=first, borderValue=0) & cv2.dilate(trace, kernel, anchor=last, borderValue=0)
    return hits


def footprint(trace, clip, scale, reach_m):
    """Close and bracket patterned cracks on a grid near 2 cm in the scoring scale."""
    h, w = trace.shape
    factor = max(1, round(.02 / scale))
    nh, nw = math.ceil(h / factor), math.ceil(w / factor)
    coarse = cv2.resize(trace.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0
    source = coarse.astype(np.uint8)
    cell = scale * max(h / nh, w / nw)
    radius = max(1, round(.12 / cell))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    close = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel) | source
    flood = np.pad(close, 1)
    cv2.floodFill(flood, None, (0, 0), 2, flags=4)
    distance = cv2.distanceTransform(1 - source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    reach = reach_m / cell
    fill = ((flood[1:-1, 1:-1] == 0) & (distance <= reach)) | (bracket_hits(source, reach) >= 2)
    return (trace | cv2.resize(fill.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)) & clip


if __name__ == "__main__":
    cv2.setNumThreads(1)
    if sys.argv[1:] == ["--views-stream"]:
        from native_windows import views
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = views(Path(request["folder"]), request["payload"])
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(response, allow_nan=False), flush=True)
        raise SystemExit(0)
    action, folder = sys.argv[1:]
    try:
        payload = json.load(sys.stdin)
        if action in ("detector_tiles", "detector_evidence"):
            import detector_views
            function = detector_views.tiles if action == "detector_tiles" else detector_views.evidence
            result = function(Path(folder), payload)
        elif action in ("road_view","road_mask","road_candidates"):
            import carriageway
            function={"road_view":carriageway.view,"road_mask":carriageway.mask,
                      "road_candidates":carriageway.filter_candidates}[action]
            result=function(Path(folder),payload)
        elif action == "assemble":
            from sample_imagery import assemble
            result = assemble(Path(folder), payload)
        elif action == "detection_geometry":
            from detection_geometry import measure as measure_detections

            result = measure_detections(Path(folder),payload)
        elif action == "views":
            from native_windows import views

            result = views(Path(folder), payload)
        else:
            result = {"prepare":prepare}[action](Path(folder), payload)
        json.dump(result, sys.stdout, allow_nan=False)
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
