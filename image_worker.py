"""Local vision evidence preparation and geometry on the original pixel grid."""

import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

def read_mask(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"{path.name}: expected one mask band/frame.")
        if image.mode in ("RGB", "RGBA", "P"):
            rgb = np.asarray(image.convert("RGBA"))
            if (rgb[:, :, 3] != 255).any() or not (
                np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
            ):
                raise ValueError(f"{path.name}: use a grayscale foreground mask, or a binary mask per class.")
            data = rgb[:, :, 0]
        else:
            data = np.asarray(image)
    if data.ndim != 2 or not np.isfinite(data).all() or (data < 0).any():
        raise ValueError(f"{path.name}: expected a nonnegative grayscale mask.")
    values = np.unique(data)
    if len(values) > 2 or (len(values) == 2 and values[0] != 0):
        raise ValueError(f"{path.name}: use zero background and one nonzero foreground value.")
    return data != 0


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


def skeleton_length(mask):
    # Zhang–Suen fallback matches the topology rules used by the CAE geometry workflow.
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(points)
    raster = np.pad(mask[y:y+h, x:x+w].astype(np.uint8), 1)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        raster = (cv2.ximgproc.thinning(raster * 255) != 0).astype(np.uint8)
    else:
        codes = np.arange(256)[:, None]
        bits = ((codes >> np.arange(8)) & 1).astype(bool)
        count = bits.sum(axis=1)
        valid = (count >= 2) & (count <= 6) & ((~bits & np.roll(bits, -1, axis=1)).sum(axis=1) == 1)
        rules = (
            valid & ~(bits[:, 0] & bits[:, 2] & bits[:, 4]) & ~(bits[:, 2] & bits[:, 4] & bits[:, 6]),
            valid & ~(bits[:, 0] & bits[:, 2] & bits[:, 6]) & ~(bits[:, 0] & bits[:, 4] & bits[:, 6]),
        )
        kernel = np.array([[128, 1, 2], [64, 0, 4], [32, 16, 8]], np.float32)
        changed = True
        while changed:
            changed = False
            for rule in rules:
                remove = (raster > 0) & rule[cv2.filter2D(raster, -1, kernel, borderType=cv2.BORDER_CONSTANT)]
                if remove.any():
                    raster[remove] = 0
                    changed = True
    line = raster[1:-1, 1:-1].astype(bool)
    horizontal = (line[:, :-1] & line[:, 1:]).sum()
    vertical = (line[:-1, :] & line[1:, :]).sum()
    crosses = ~line[:-1, 1:] & ~line[1:, :-1]
    mirror = ~line[:-1, :-1] & ~line[1:, 1:]
    diagonal = ((line[:-1, :-1] & line[1:, 1:]) & crosses).sum()
    diagonal += ((line[:-1, 1:] & line[1:, :-1]) & mirror).sum()
    return float(horizontal + vertical + math.sqrt(2) * diagonal)


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


def region_clip(box, w, h):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError("Boxes must have normalized [x0,y0,x1,y1] coordinates between 0 and 1000.")
    region = np.zeros((h, w), bool)
    region[math.floor(y0*h/1000):math.ceil(y1*h/1000),
           math.floor(x0*w/1000):math.ceil(x1*w/1000)] = True
    return region


def measure(run, payload):
    if payload["meta"].get("storage") == "windowed_tiff":
        from native_windows import measure as measure_native

        return measure_native(run, payload)
    meta = payload["meta"]
    path = run / "roi.png"
    if file_hash(path) != meta["roi_sha256"]:
        raise ValueError("ROI evidence changed after preparation.")
    roi = read_mask(path)
    known = {row["mask_id"]:row for row in meta["masks"]}
    masks = {}
    for mask_id, item in known.items():
        path = run / f"{mask_id}.png"
        if file_hash(path) != item["snapshot_sha256"]:
            raise ValueError(f"{mask_id}: evidence changed after preparation.")
        masks[mask_id] = read_mask(path)
    groups = {key:[] for key in masks}
    for row in payload["judgments"]:
        if row["mask_id"] not in groups:
            raise ValueError(f"Unknown mask ID {row['mask_id']!r}.")
        groups[row["mask_id"]].append(row)
    candidates = []
    unclassified = {}
    for mask_id, mask in masks.items():
        split = groups[mask_id]
        if not split or (len(split) > 1 and any(row["box"] is None for row in split)):
            raise ValueError(f"{mask_id}: give a judgment or separate mixed types into explicit boxes.")
        claimed = np.zeros_like(mask)
        item = known[mask_id]
        for row in split:
            clip = roi if row["box"] is None else roi & region_clip(row["box"], meta["width"], meta["height"])
            raw = mask & clip
            if not raw.any():
                raise ValueError(f"{mask_id}: a decision covers no foreground pixels.")
            if (raw & claimed).any():
                raise ValueError(f"{mask_id}: boxes overlap on foreground pixels.")
            claimed |= raw
            kind = item["kind"] or row["kind"]
            if item["kind"] and row["kind"] not in (None, item["kind"]):
                raise ValueError(f"{mask_id}: keep the supplied class {item['kind']}.")
            if not kind:
                raise ValueError(f"{mask_id}: provide a class decision or use unknown.")
            semantics = item["semantics"]
            if semantics == "unspecified":
                semantics = row["semantics"]
            if kind in ("alligator", "block_crack", "patch", "pothole") and semantics == "unspecified":
                raise ValueError(f"{mask_id}: specify whether the mask is an affected region or crack trace.")
            final = raw
            if kind in ("alligator", "block_crack") and semantics == "crack_trace":
                final = footprint(raw, clip, payload["scoring_scale"], .5 if kind == "alligator" else 1.5)
            qualified = kind != "unknown" and (kind not in ("patch", "pothole") or semantics != "crack_trace")
            candidates.append((row, kind, semantics, raw, final, qualified))
        if (mask & ~claimed).any():
            unclassified[mask_id] = int((mask & ~claimed).sum())

    # Assign duplicate same-type pixels by mask order, so interval choices
    # never change the footprint or quantity belonging to an instance.
    seen = {}
    area_distress = np.zeros_like(roi)
    cross_type_overlap = np.zeros_like(roi)
    for _, kind, _, _, final, valid in candidates:
        if valid and kind in ("alligator", "block_crack", "patch"):
            area_distress |= final
    rows = []
    for row, kind, semantics, raw, final, qualified in candidates:
        label = row["mask_id"]
        used = final.copy()
        overlap = 0
        if qualified:
            prior = seen.setdefault(kind, np.zeros_like(roi))
            blocked = prior | area_distress if kind == "linear_crack" else prior
            overlap = int((used & blocked).sum())
            used &= ~blocked
            prior |= used
            if kind != "linear_crack":
                for other_kind, other_mask in seen.items():
                    if kind != other_kind and other_kind != "linear_crack":
                        cross_type_overlap |= used & other_mask
        area = int(used.sum())
        length = skeleton_length(raw & used) if qualified and kind == "linear_crack" and area else 0.
        count = 1 if qualified and kind == "pothole" and area else 0
        quantity = length if kind == "linear_crack" else count if kind == "pothole" else area
        rows.append(dict(mask_id=label, kind=kind,
                         severity_lower=row["severity_lower"], severity_upper=row["severity_upper"],
                         severity_source=row["severity_source"],
                         class_source="provided" if known[label]["kind"] else "vision_agent",
                         evidence=row["evidence"], box=row["box"], semantics=semantics,
                         trace_or_mask_area_px2=int(raw.sum()), measured_area_px2=area,
                         added_affected_area_px2=int((used & ~raw).sum()),
                         length_px=length, count=count, overlap_removed_px2=overlap,
                         scored=bool(qualified and quantity > 0),
                         withheld=bool(not qualified or (kind == "linear_crack" and area and not length))))
    return {"rows":rows, "unassigned_pixels":unclassified,
            "cross_type_overlap_px2":int(cross_type_overlap.sum())}


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
        if action == "grading_crops":
            from native_windows import grading_crops
            result = grading_crops(Path(folder),payload)
        elif action == "detection_roi":
            from detection_geometry import pavement_roi

            result = pavement_roi(Path(folder),payload)
        elif action == "detection_geometry":
            from detection_geometry import measure as measure_detections

            result = measure_detections(Path(folder),payload)
        elif action == "views":
            from native_windows import views

            result = views(Path(folder), payload)
        else:
            result = {"prepare":prepare, "measure":measure}[action](Path(folder), payload)
        json.dump(result, sys.stdout, allow_nan=False)
    except (OSError, ValueError, KeyError, cv2.error) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
