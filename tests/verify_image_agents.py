"""Exercise the three independent MCP agents with explicit vision judgments."""

import asyncio
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "image": ("prepare_image_direct_agent", "record_image_direct_pci"),
    "mask": ("prepare_unlabeled_mask_agent", "score_unlabeled_mask_pci"),
    "labeled": ("prepare_labeled_mask_agent", "score_labeled_mask_pci"),
}


@asynccontextmanager
async def connect(mode, runs):
    server = StdioServerParameters(command="/bin/sh", args=[str(ROOT / "launch-mcp.sh"), "--agent", mode],
                                   env={"PCI_AGENT_RUNS":str(runs), "PCI_AGENT_PYTHON":sys.executable, "PCI_IMAGE_PYTHON":os.environ.get("PCI_IMAGE_PYTHON",sys.executable),
                                        "PCI_PERSISTENT_VIEWS":os.environ.get("PCI_PERSISTENT_VIEWS","0")})
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=50) as client:
            await client.initialize()
            extras = {"mask":{"read_mask_grading_crops"}}.get(mode,set())
            assert {tool.name for tool in (await client.list_tools()).tools} == set(MODES[mode]) | extras | {
                "read_pavement_windows","save_pavement_window_reviews"}
            yield client


async def call(client, name, args, image_count=None, error=None):
    result = await client.call_tool(name, args)
    if error is not None:
        assert result.is_error and error in result.content[0].text, result
        return
    assert not result.is_error and result.structured_content, result
    if image_count is not None:
        assert sum(part.type == "image" for part in result.content) == image_count
    return result.structured_content


def write_pgm(path, arr):
    path.write_bytes(b"P5\n12 12\n255\n" + bytes(value for row in arr for value in row))


async def verify(scratch):
    line = [[255 if y == 1 and 1 <= x <= 10 else 0 for x in range(12)] for y in range(12)]
    area = [[255 if 4 <= y <= 6 and 4 <= x <= 6 else 0 for x in range(12)] for y in range(12)]
    ring = [[255 if 2 <= y <= 9 and 2 <= x <= 9 and (y in (2, 9) or x in (2, 9)) else 0
             for x in range(12)] for y in range(12)]
    photo = [[80 + (x * y) % 60 for x in range(12)] for y in range(12)]
    for file, data in (("line.pgm", line), ("patch.pgm", area), ("ring.pgm", ring), ("raw.pgm", photo)):
        write_pgm(scratch / file, data)
    no_scale = {"status":"unavailable"}
    known = {"status":"known","m_per_px":.1}
    judgments = [{"mask_id":"mask_1","kind":"linear_crack","severity":"L","evidence":"Test line"},
                 {"mask_id":"mask_2","kind":"patch","severity_lower":"L","severity_upper":"M","evidence":"Test region"}]
    for j in judgments:
        j.update(severity_basis="rgb_crop",severity_rationale="Synthetic instance condition is intact; high severity excluded.",
                 excluded_grade_evidence={"M":"Fine narrow intact test line.","H":"No broken edges in synthetic evidence."})
    runs = scratch / "runs"

    async with connect("image", runs) as client:
        names = MODES["image"]
        args = {"image_path":str(scratch/"raw.pgm")}
        ask = await call(client, names[0], args)
        assert ask["status"] == "ask_user_for_scale" and not runs.exists()
        for scale in (no_scale, known):
            prepared = await call(client, names[0], {**args, "physical_scale":scale}, image_count=1)
            result = await call(client, names[1], {"run_id":prepared["run_id"],"pci":71,"pci_lower":65,"pci_upper":78,
                "evidence":"Test judgment","lower_evidence":"Potential missed fine cracks","upper_evidence":"Broad intact surface"})
            assert result["pci"] == 71 and result["image_dimensions_px"] == [12,12]
            assert result["pci_interval"]=={"lower":65,"upper":78} and not result["statistically_calibrated"]
            await call(client,names[1],{"run_id":prepared["run_id"],"pci_lower":75,"pci_upper":70,
                "evidence":"Test","lower_evidence":"Test","upper_evidence":"Test"},error="ordered distinct")
            assert result["scale"]["status"] == scale["status"]

    async with connect("mask", runs) as client:
        names = MODES["mask"]
        masks = [{"path":str(scratch/"line.pgm")},{"path":str(scratch/"patch.pgm"), "semantics":"affected_region"}]
        ask = await call(client, names[0], {"masks":masks})
        assert ask["status"] == "ask_user_for_scale"
        prepared = await call(client, names[0], {"masks":masks,"physical_scale":no_scale}, image_count=2)
        measuring = await call(client, names[1], {"run_id":prepared["run_id"],"judgments":judgments,"measure_only":True})
        assert measuring["status"] == "needs_visual_severity_assessment"
        assert measuring["pci_proxy_interval"] is None and measuring["measurements"][0]["severity_lower"] is None
        untyped = await call(client, names[1], {"run_id":prepared["run_id"],"judgments":judgments})
        proxy = untyped["pci_proxy_interval"]
        assert untyped["pci_interval"] is None and 0 <= proxy["lower"] < proxy["upper"] <= 100
        bounds = json.loads(Path(untyped["audit_files"]["pci_bounds_details.json"]).read_text())
        assert bounds["assignments_evaluated"] == 2
        for endpoint in bounds["endpoint_details"].values():
            assert endpoint["instance_grades"]["mask_1_1"] == "L"
            assert endpoint["instance_grades"]["mask_2_2"] in {"L","M"}
        assert untyped["scale"]["reference_sample_area_m2_for_pci_proxy"] == 5000 * .3048**2
        rows = {row["kind"]:row for row in untyped["measurements"]}
        assert rows["linear_crack"]["length_px"] == 9
        assert rows["linear_crack"]["density_px_basis"] == 100 * 9/144
        assert rows["patch"]["measured_area_px2"] == 9
        assert rows["patch"]["density_metric"] is None
        assert (rows["patch"]["severity_lower"], rows["patch"]["severity_upper"]) == ("L","M")
        assert set(row["class_source"] for row in untyped["measurements"]) == {"vision_agent"}
        unknown = [{"mask_id":r["mask_id"],"kind":"unknown","evidence":"Unable to establish class"} for r in judgments]
        withheld = await call(client, names[1], {"run_id":prepared["run_id"],"judgments":unknown})
        assert withheld["status"] == "no_scorable_distress" and withheld["pci_proxy_interval"] is None
        reversed_interval = [judgments[0], {**judgments[1], "severity_lower":"H"}]
        await call(client, names[1], {"run_id":prepared["run_id"],"judgments":reversed_interval},
                   error="Severity endpoints must be ordered")
        # A recognized crack remains an instance even if its PCI subtype is
        # uncertain. Joint candidate assignments must remeasure area vs length.
        family={**judgments[0],"kind":"crack","preferred_crack_kind":"linear_crack",
                "class_candidates":["linear_crack","block_crack"],"semantics":"crack_trace",
                "class_rationale":"Synthetic elongated trace; block-network interpretation retained for this test."}
        candidate=await call(client,names[1],{"run_id":prepared["run_id"],"judgments":[family,judgments[1]]})
        assert candidate['class_search']['complete'] and candidate['class_search']['evaluated_assignments']==2
        assert candidate['crack_family_instances'][0]['kind']=='crack'
        assert candidate['pci_proxy_interval'] and candidate['preferred_class_pci_proxy_interval']==proxy
        individual=[]
        for kind in family['class_candidates']:
            resolved={k:v for k,v in family.items() if k not in ('class_candidates','preferred_crack_kind')}
            one=await call(client,names[1],{"run_id":prepared["run_id"],"judgments":[{**resolved,"kind":kind},judgments[1]]})
            individual.append(one['pci_proxy_interval'])
            row=one['measurements'][0]
            assert row['kind']==kind
            assert row['length_px']>0 if kind=='linear_crack' else row['length_px']==0 and row['measured_area_px2']>0
        assert candidate['pci_proxy_interval']=={'lower':min(q['lower'] for q in individual),'upper':max(q['upper'] for q in individual)}
        assert candidate['measurements'][0]['class_family']=='crack'
        await call(client,names[1],{"run_id":prepared["run_id"],"judgments":[{
            **family,"kind":"linear_crack"},judgments[1]]},error="kind=crack")
        await call(client,names[1],{"run_id":prepared["run_id"],"judgments":[{
            **family,"preferred_crack_kind":None},judgments[1]]},error="preferred subtype")
        many=await call(client,names[0],{"masks":[masks[0]]*5,"physical_scale":no_scale})
        large=await call(client,names[1],{"run_id":many["run_id"],"judgments":[{
            **family,"mask_id":f'mask_{i}'} for i in range(1,6)]})
        assert large['status']=='preferred_class_estimate_only' and not large['class_search']['complete']
        assert large['pci_proxy_interval'] is None and large['preferred_class_pci_proxy_interval']

        reviewed=await call(client,names[0],{"masks":masks,"physical_scale":known})
        measured=await call(client,names[1],{"run_id":reviewed["run_id"],"judgments":judgments,"measure_only":True})
        crops=await call(client,"read_mask_grading_crops",{"measurement_run_id":measured["run_id"],
            "original_image_path":str(scratch/"raw.pgm")},image_count=2)
        assert crops["next_cursor"] is None and crops["total"]==2
        await call(client,names[1],{"run_id":reviewed["run_id"],"judgments":[{**judgments[0],"kind":"patch"},judgments[1]]},error="frozen mask-only classes")
        locked=await call(client,names[1],{"run_id":reviewed["run_id"],"judgments":judgments})
        assert locked["pci_interval"] and locked["pci_interval_condition"]

    async with connect("labeled", runs) as client:
        names = MODES["labeled"]
        args = {"masks":[{**mask,"kind":judgment["kind"]} for mask,judgment in zip(masks,judgments)],
                "original_image_path":str(scratch/"raw.pgm")}
        ask = await call(client, names[0], args)
        assert ask["status"] == "ask_user_for_scale"
        result = await call(client, names[0], {**args,"physical_scale":no_scale},image_count=3)
        typed = await call(client, names[1], {"run_id":result["run_id"],"judgments":judgments})
        assert typed["pci_proxy_interval"] == proxy
        assert set(row["class_source"] for row in typed["measurements"]) == {"provided"}
        calibrated = await call(client, names[0], {**args,"physical_scale":known},image_count=3)
        metric = await call(client, names[1], {"run_id":calibrated["run_id"],"judgments":judgments})
        interval = metric["pci_interval"]
        assert metric["pci_proxy_interval"] is None and 0 <= interval["lower"] < interval["upper"] <= 100
        line_row = next(row for row in metric["measurements"] if row["kind"] == "linear_crack")
        assert line_row["measured_length_m"] == .9 and line_row["density_metric"] is not None
        mismatch = [{**judgments[0],"kind":"patch"},judgments[1]]
        await call(client, names[1], {"run_id":calibrated["run_id"],"judgments":mismatch},
                   error="keep the supplied class")
        ring = await call(client, names[0], {
            "masks":[{"path":str(scratch/"ring.pgm"),"kind":"block_crack","semantics":"crack_trace"}],
            "physical_scale":{"status":"known","pavement_area_m2":1.44},
        }, image_count=1)
        filled = await call(client, names[1], {"run_id":ring["run_id"], "judgments":[
            {**judgments[0],"kind":"block_crack","evidence":"Synthetic closed block"},
        ]})
        assert filled["scale"]["m_per_px"] == .1
        assert filled["measurements"][0]["trace_or_mask_area_px2"] == 28
        assert filled["measurements"][0]["measured_area_px2"] > 28
        fixed = filled["pci_interval"]
        assert fixed["lower"] == fixed["upper"]
        omitted = await call(client, names[1], {"run_id":ring["run_id"], "judgments":[
            {"mask_id":"mask_1","evidence":"Unknown severity"},
        ]})
        assert omitted["measurements"][0]["severity_lower"] is None
        assert omitted["status"]=="needs_instance_severity_review" and omitted["pci_interval"] is None
        await call(client,names[1],{"run_id":ring["run_id"],"judgments":[{
            "mask_id":"mask_1","severity_lower":"L","severity_upper":"H","evidence":"No narrowing"}]},
            error="not an evidence-narrowed")
        await call(client,names[1],{"run_id":ring["run_id"],"judgments":[{
            "mask_id":"mask_1","severity":"L","evidence":"No structured rationale"}]},
            error="basis and instance-specific rationale")
        forced = {"mask_id":"mask_1","severity_lower":"L","severity_upper":"M",
                  "evidence":"User requests a surface estimate despite ambiguous opening.",
                  "severity_basis":"rgb_crop","severity_rationale":"Surface appears intact around the fine trace.",
                  "forced_surface_estimate":True,"visual_confidence":"low","preferred_severity":"L",
                  "surface_estimate_assumptions":"Assume obscured edges resemble visible intact surrounding pavement; no depth measurement."}
        estimated = await call(client,names[1],{"run_id":calibrated["run_id"],"judgments":[forced,judgments[1]]})
        assert estimated['pci_interval'] and estimated['forced_surface_estimate_count']==1
        assert estimated['interval_type']=='surface_assumption_pci_interval' and not estimated['statistically_calibrated']
        assert estimated['measurements'][0]['surface_estimate_assumptions']==forced['surface_estimate_assumptions']
        assert estimated['measurements'][0]['preferred_severity']=='L'
        await call(client,names[1],{"run_id":calibrated["run_id"],"judgments":[{
            **forced,"surface_estimate_assumptions":None},judgments[1]]},error="explicit assumptions")
        await call(client,names[1],{"run_id":calibrated["run_id"],"judgments":[{
            **forced,"preferred_severity":"H"},judgments[1]]},error="Preferred severity")
        unresolved = await call(client,names[1],{"run_id":calibrated["run_id"],"judgments":[{
            **forced,"severity_upper":"H","visual_confidence":"very_low"},judgments[1]]})
        assert unresolved['forced_unresolved_range_ids']==['mask_1']
        assert unresolved['pci_interval'] and 'unresolved' in unresolved['severity_estimation_warning']
        await call(client,names[1],{"run_id":ring["run_id"],"judgments":[{
            **forced,"kind":"block_crack"}]},error="aligned original RGB image")
        hole=await call(client,names[0],{"masks":[{"path":str(scratch/"patch.pgm"),"kind":"pothole","semantics":"affected_region"}],
            "physical_scale":known},image_count=1)
        fixed_hole=await call(client,names[1],{"run_id":hole["run_id"],"judgments":[{
            **judgments[0],"kind":"pothole","severity":"M","severity_basis":"measured_features",
            "severity_rationale":"Synthetic supplied diameter and depth support M.",
            "excluded_grade_evidence":{"L":"Depth excludes L.","H":"Depth below H threshold."}}]})
        assert fixed_hole['measurements'][0]['severity_lower']==fixed_hole['measurements'][0]['severity_upper']=='M'

    python = os.environ.get("PCI_IMAGE_PYTHON",sys.executable)
    engine = ROOT / "engine/mask_intervals.py"
    quantities = (22.11, 8.53, .88, 59.89)
    rows = [dict(record_id=str(i), distress="linear_crack", unit="m", quantity=q,
                 severity_lower="L", severity_upper="H") for i,q in enumerate(quantities)]
    command = [python, "-B", str(engine)]
    process = subprocess.run(command, input=json.dumps({"pavement_area_m2":465,"inventory":rows}),
                             capture_output=True, check=True, text=True)
    mixed = json.loads(process.stdout)
    assert mixed["assignments_evaluated"] == 81
    assert list(mixed["endpoint_details"]["upper"]["instance_grades"].values()) == ["L","L","M","L"]
    assert math.isclose(mixed["upper"], 87.07571481064588)
    assert mixed["upper"] > 86.99188336980802  # Previous uniform-L upper candidate.

    added = [dict(rows[0], record_id=str(i), quantity=1.7) for i in (4,5)]
    broad = subprocess.run(command, input=json.dumps({"pavement_area_m2":465,"inventory":rows+added}),
                           capture_output=True, check=True, text=True)
    caebound = json.loads(broad.stdout)
    assert caebound["method"] == "cae_outer_bound" and caebound["boxes_evaluated"] > 0
    assert caebound["lower"] <= caebound["min_witness_ceiling"]
    assert caebound["max_witness_floor"] <= caebound["upper"]

    print("Three isolated MCP agents passed. Verified all scale prompts, image delivery, fixed-label enforcement, "
          "pixel measurements, missing evidence, metric and pixel proxy intervals, mixed-grade extrema, and CAE fallback.")


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        asyncio.run(verify(Path(folder).resolve()))
