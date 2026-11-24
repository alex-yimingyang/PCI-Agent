"""Search each measured instance's admissible grades with the CAE PCI scorer."""

import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys

import pci_bounds
import pci_core

GRADES = pci_bounds.GRADES
TOLERANCE = .001
BOX_LIMIT = 50_000
EXACT_LIMIT = 256


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exhaustive(rows, entries, area, curves, assignment_count):
    """Report attained extrema for a small set of independent assignments."""
    lowest = highest = None
    options = [range(entry.low, entry.high + 1) for entry in entries]
    for assignment in product(*options):
        inventory = [dict(distress=entry.kind, unit=entry.unit, quantity=entry.quantity,
                          severity=GRADES[grade]) for grade, entry in zip(assignment, entries, strict=True)]
        try:
            score = pci_core.score_inventory(inventory, area, curves)
        except ValueError as exc:
            if "FAA source table has no q=" in str(exc):
                return None  # Defer to CAE's conservative handling of unavailable CDV rows.
            raise
        candidate = (score["pci"], assignment, score)
        if lowest is None or candidate[0] < lowest[0]:
            lowest = candidate
        if highest is None or candidate[0] > highest[0]:
            highest = candidate
    endpoints = {}
    for label, candidate in (("lower", lowest), ("upper", highest)):
        pci, assignment, score = candidate
        endpoints[label] = {
            "pci": pci,
            "instance_grades": {row["record_id"]: GRADES[grade]
                                for row, grade in zip(rows, assignment, strict=True)},
            "max_cdv": score["max_cdv"],
            "deductions": score["deductions"],
            "iterations": score["iterations"],
            "flags": score["flags"],
        }
    return {
        "lower": lowest[0], "upper": highest[0],
        "min_witness_ceiling": lowest[0], "max_witness_floor": highest[0],
        "converged": True, "search_stop": "all_assignments_evaluated",
        "boxes_evaluated": 0, "assignments_evaluated": assignment_count,
        "endpoint_details": endpoints, "method": "exhaustive",
    }


def score_intervals(payload):
    rows, area = payload["inventory"], payload["pavement_area_m2"]
    if not rows or not math.isfinite(area) or area <= 0:
        raise ValueError("Interval scoring requires a positive sample area and a measured inventory.")
    entries = [pci_bounds.Entry(row["distress"], row["unit"], row["quantity"],
                                GRADES.index(row["severity_lower"]), GRADES.index(row["severity_upper"]))
               for row in rows]
    if any(entry.low > entry.high for entry in entries):
        raise ValueError("Severity intervals must be ordered L, M, H.")
    root = Path(__file__).resolve().parent
    curves_path = root / "curves/faa_road_ac_tables.json"
    curves = pci_core.load_curves(curves_path)
    assignment_count = math.prod(entry.high - entry.low + 1 for entry in entries)
    result = exhaustive(rows, entries, area, curves, assignment_count) if assignment_count <= EXACT_LIMIT else None
    if result is None:
        result = pci_bounds.bound(entries, area, curves, tolerance=TOLERANCE, box_limit=BOX_LIMIT)
        result["method"] = "cae_outer_bound"
    result["endpoint_search_gap_pci"] = max(result["min_witness_ceiling"] - result["lower"],
                                            result["upper"] - result["max_witness_floor"])
    result["tolerance_pci"] = TOLERANCE
    result["box_limit"] = BOX_LIMIT
    result["curve_status"] = curves["status"]
    result["curve_sha256"] = file_sha(curves_path)
    result["interval_engine_sha256"] = file_sha(root / "pci_bounds.py")
    return result


if __name__ == "__main__":
    try:
        json.dump(score_intervals(json.load(sys.stdin)), sys.stdout, allow_nan=False)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
