"""Drive the original road PCI executable and organize independent run artifacts."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine" / "road_pci.py"
RUNS = Path(os.environ.get("PCI_AGENT_RUNS", str(ROOT / "runs"))).expanduser().resolve()
FILES = ("pci_results.json", "pci_results.csv", "deduct_values.csv", "cdv_iterations.csv",
         "severity_decisions.csv", "excluded_samples.csv", "curves_used.json")


class WorkflowError(ValueError):
    """Input or scoring errors that can be shown to the user."""


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise WorkflowError(f"{path.name} has missing or duplicate column headers.")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise WorkflowError(f"{path.name} has malformed CSV rows.")
    return fields, rows


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def input_path(text):
    path = Path(text).expanduser()
    if not path.is_absolute() or path.suffix.casefold() != ".csv" or not path.is_file():
        raise WorkflowError("Provide an existing absolute local .csv path for each input.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise WorkflowError(f"Cannot read {path.name}: {exc.strerror}") from None


def new_run():
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / uuid4().hex
    run.mkdir()
    return run


def result_summary(result):
    bounds = {key: round(value["pci"], 6) for key, value in result["scenarios"].items()}
    return {
        "sample_id": result["sample_id"],
        "sample_area_m2": result.get("sample_area_m2"),
        "status": result["status"],
        "pci": bounds.get("M"),
        "severity_scenarios": bounds,
    }


def score(run, unknown, curve_policy):
    """Score run/samples.csv and run/inventory.csv via the unmodified CLI."""
    destination = run / "result"
    try:
        process = subprocess.run(
            [sys.executable, "-B", str(ENGINE), "--samples", str(run / "samples.csv"),
             "--inventory", str(run / "inventory.csv"), "--out", str(destination),
             "--unknown", unknown, "--curve-policy", curve_policy],
            cwd=ENGINE.parent, check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=90,
        )
    except subprocess.TimeoutExpired:
        raise WorkflowError(f"PCI calculation timed out. Input snapshots are in {run}.") from None
    except OSError as exc:
        raise WorkflowError(f"Could not launch the PCI engine: {exc.strerror}") from None
    (run / "engine.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    json_file = destination / "pci_results.json"
    if process.returncode:
        failures = []
        if json_file.is_file():
            parsed = json.loads(json_file.read_text(encoding="utf-8"))
            for result in parsed["samples"]:
                if result["status"] != "provisional":
                    detail = result.get("error") or result.get("scenario_errors") or result["status"]
                    failures.append(f"{result['sample_id']}: {detail}")
        detail = " | ".join(failures[:6]) or process.stderr.strip() or process.stdout.strip()
        raise WorkflowError(f"road_pci.py failed. {detail[:2000]} Audit directory: {run}")
    missing = [name for name in FILES if not (destination / name).is_file()]
    if missing:
        raise WorkflowError(f"road_pci.py returned without required audit files: {', '.join(missing)}")
    report = json.loads(json_file.read_text(encoding="utf-8"))
    if report.get("version") != "road-pci-1.0" or not report.get("samples"):
        raise WorkflowError("Unexpected output from the bundled road PCI engine.")
    audit = {name: str(destination / name) for name in FILES}
    audit["engine.log"] = str(run / "engine.log")
    audit["input_samples.csv"] = str(run / "samples.csv")
    audit["input_inventory.csv"] = str(run / "inventory.csv")
    return report, audit


def present_run(run, report, audit):
    return {
        "run_id": run.name,
        "results": [result_summary(result) for result in report["samples"][:30]],
        "sample_count": len(report["samples"]),
        "results_shown": min(len(report["samples"]), 30),
        "official_pci": report["official_pci"],
        "curve_status": report["curve_status"],
        "hashes": {
            "inventory_sha256": report["inventory_sha256"],
            "samples_sha256": report["samples_sha256"],
            "curve_sha256": report["curve_sha256"],
            "engine_sha256": hashlib.sha256(ENGINE.read_bytes()).hexdigest(),
        },
        "audit_files": audit,
    }


def calculate_csv(samples_csv, inventory_csv, unknown, curve_policy):
    samples = input_path(samples_csv)
    inventory = input_path(inventory_csv)
    run = new_run()
    (run / "samples.csv").write_bytes(samples)
    (run / "inventory.csv").write_bytes(inventory)
    report, audit = score(run, unknown, curve_policy)
    return present_run(run, report, audit)


def calculate_values(sample_id, area_m2, distresses, confirmed_no_distress, unknown, curve_policy):
    sample_id = sample_id.strip()
    if not sample_id:
        raise WorkflowError("Provide a sample ID.")
    if confirmed_no_distress and distresses:
        raise WorkflowError("A confirmed distress-free sample cannot include distress rows.")
    if not distresses and not confirmed_no_distress:
        raise WorkflowError("Provide measured distress rows or independent confirmation of no distress.")
    run = new_run()
    write_csv(run / "samples.csv", ("sample_id", "area_m2", "confirmed_no_distress"),
              [{"sample_id": sample_id, "area_m2": area_m2,
                "confirmed_no_distress": str(confirmed_no_distress).lower()}])
    headers = ("sample_id", "record_id", "distress", "severity", "quantity", "unit", "severity_source")
    write_csv(run / "inventory.csv", headers,
              [{"sample_id": sample_id, "record_id": f"distress_{i}", **row}
               for i, row in enumerate(distresses, 1)])
    report, audit = score(run, unknown, curve_policy)
    return present_run(run, report, audit)


def explain(run_id, sample_id, scenario):
    if len(run_id) != 32 or any(c not in "abcdef0123456789" for c in run_id):
        raise WorkflowError("Use the run_id returned by a PCI calculation.")
    run = RUNS / run_id
    source = run / "result" / "pci_results.json"
    if not source.is_file():
        raise WorkflowError("No completed scoring result for this run and method.")
    report = json.loads(source.read_text(encoding="utf-8"))
    result = next((r for r in report["samples"] if r["sample_id"] == sample_id), None)
    if result is None:
        raise WorkflowError(f"Sample {sample_id!r} is not included in this run.")
    data = result["scenarios"].get(scenario)
    if data is None:
        raise WorkflowError(f"Scenario {scenario} is unavailable for {sample_id}: {result.get('error', result['status'])}")
    _, decisions = read_csv(run / "result" / "severity_decisions.csv")
    decisions = [row for row in decisions if row["sample_id"] == sample_id]
    _, inventory = read_csv(run / "inventory.csv")
    measured = [row for row in inventory if row["sample_id"].strip() == sample_id]
    for decision, input_row in zip(decisions, measured, strict=True):
        decision["quantity_source"] = input_row.get("quantity_source", "")
    return {
        "run_id": run_id, "sample_id": sample_id,
        "method": "measured_inventory",
        "scenario": scenario, "scenario_output": data,
        "sample_area_m2": result["sample_area_m2"],
        "flags": result["flags"], "curve_status": report["curve_status"],
        "severity_decisions": decisions[:40], "severity_row_count": len(decisions),
        "audit_files": {name: str(source.parent / name) for name in FILES},
    }
