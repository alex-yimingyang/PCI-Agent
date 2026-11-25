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


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def new_run():
    RUNS.mkdir(parents=True, exist_ok=True)
    run = RUNS / uuid4().hex
    run.mkdir()
    return run


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
