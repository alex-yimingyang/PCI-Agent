"""Provisional ROAD asphalt PCI using replaceable, digitized DV/CDV curves.

No airport curves are implemented. Every out-of-domain approximation is returned.
"""
import math
import itertools
import json
from pathlib import Path

FT_PER_M = 1 / 0.3048


def load_curves(path=None):
    path = Path(path or Path(__file__).parent / 'curves/faa_road_ac_tables.json')
    raw = json.loads(path.read_text(encoding='utf-8'))
    curves = {**raw, 'application': 'road_asphalt_research', 'dv': {}, 'cdv': {}}
    for kind, spec in raw['dv'].items():
        x = spec.get('x', raw['x_area'])
        ys = {s: spec[s] for s in ('L', 'M', 'H')}
        if any(len(y) > len(x) or len(y) < 2 for y in ys.values()):
            raise ValueError(f'Invalid DV table: {kind}')
        curves['dv'][kind] = {**spec, 'x': x, 'y': ys, 'log_x': True,
            'density_basis': {'m': '100 * ft / ft2', 'm2': 'area percent', 'count': 'count per 100 ft2'}[spec['unit']]}
    for q, y in raw['cdv'].items():
        curves['cdv'][q] = {'x': raw['cdv_x'], 'y': y, 'estimated': False}
    return curves


def density(quantity, unit, area_m2):
    if not all(math.isfinite(x) for x in (quantity, area_m2)):
        raise ValueError("Non-finite quantity/area")
    if quantity < 0 or area_m2 <= 0:
        raise ValueError("Quantity must be nonnegative; sample area must be positive")
    # Supplied digitized curves use feet and square feet, NOT m / m2.
    if unit == "m2":
        return 100 * quantity / area_m2
    if unit == "m":
        return 100 * quantity * FT_PER_M / (area_m2 * FT_PER_M**2)
    if unit == "count":
        return 100 * quantity / (area_m2 * FT_PER_M**2)
    raise ValueError(f"Unsupported quantity unit: {unit}")


def interpolate(xs, ys, value, *, log_x=False, policy="approximate", context="curve"):
    flags = []
    if not math.isfinite(value) or value < 0 or policy not in ('strict', 'approximate'):
        raise ValueError(f'Invalid interpolation input/policy: {context}')
    if len(xs) != len(ys) or len(xs) < 2 or any(b <= a for a, b in zip(xs, xs[1:])):
        raise ValueError(f"Invalid curve: {context}")
    if value == 0:
        return 0.0, flags
    if value < xs[0]:
        if policy == "strict":
            raise ValueError(f"{context}: {value:g} below curve domain {xs[0]:g}")
        flags.append(f"{context}: below domain; assumed linear bridge from (0,0)")
        return ys[0] * value / xs[0], flags
    if value > xs[-1]:
        if policy == "strict":
            raise ValueError(f"{context}: {value:g} above curve domain {xs[-1]:g}")
        flags.append(f"{context}: above domain; used last tabulated ordinate")
        return ys[-1], flags
    transform = math.log10 if log_x else float
    for i in range(len(xs) - 1):
        if value <= xs[i + 1]:
            fraction = (transform(value) - transform(xs[i])) / (transform(xs[i + 1]) - transform(xs[i]))
            return ys[i] + fraction * (ys[i + 1] - ys[i]), flags
    return ys[-1], flags


def bound_deduct(value, context, policy):
    if 0 <= value <= 100:
        return value, []
    if policy == "strict":
        raise ValueError(f"{context}: tabulated/interpolated deduct {value} outside 0..100")
    return min(100., max(0., value)), [f"{context}: deduct {value:g} clipped to 0..100"]


def corrected_pci(values, curves, policy="approximate"):
    """D6433-style m allowance, fractional DV, and iterative q correction."""
    dvs = sorted([float(v) for v in values if v > 0], reverse=True)
    if any(not math.isfinite(v) or not 0 <= v <= 100 for v in values):
        raise ValueError("Invalid DV")
    flags, trace = [], []
    # Caller, not this numeric function, decides whether an empty survey is credible.
    if sum(v > 2 for v in dvs) <= 1:
        maximum = min(100., sum(dvs))
        return {"pci": 100 - maximum, "max_cdv": maximum, "m": None,
                "iterations": [{"q": sum(v > 2 for v in dvs), "tdv": sum(dvs),
                                "cdv": maximum, "deduct_values": dvs}], "flags": flags}
    m = min(10., 1 + 9 / 98 * (100 - dvs[0]))
    n = int(math.floor(m))
    working = dvs[:n]
    if n < len(dvs) and m - n > 1e-12:
        working.append((m - n) * dvs[n])
    while True:
        q = sum(v > 2 for v in working)
        tdv = sum(working)
        if q <= 1:
            cdv = min(100., tdv)
        else:
            curve = curves['cdv'].get(str(q))
            if curve is None:
                available = sorted(int(k) for k in curves['cdv'])
                lower = [k for k in available if k < q]
                upper = [k for k in available if k > q]
                if policy == 'strict' or not lower or not upper:
                    raise ValueError(f'FAA source table has no q={q} curve; no extrapolation of missing q curves is permitted')
                lo, hi = max(lower), min(upper)
                a, b = curves['cdv'][str(lo)], curves['cdv'][str(hi)]
                if a['x'] != b['x']:
                    raise ValueError('Cannot interpolate q curves with different abscissae')
                weight = (q-lo)/(hi-lo)
                curve = {'x': a['x'], 'y': [u+(v-u)*weight for u,v in zip(a['y'],b['y'])]}
                flags.append(f'CDV q={q}: locally interpolated between official displayed q={lo} and q={hi}; not a displayed FAA row')
            if curve.get("estimated"):
                if policy == "strict":
                    raise ValueError(f"q={q}: source curve is estimated, not observed")
                flags.append(f"CDV q={q}: source authors estimated this curve")
            cdv, notes = interpolate(curve["x"], curve["y"], tdv,
                                     policy=policy, context=f"CDV q={q}")
            flags += notes
            cdv, notes = bound_deduct(cdv, f"CDV q={q}", policy)
            flags += notes
        trace.append({"q": q, "tdv": tdv, "cdv": cdv, "deduct_values": list(working)})
        if q <= 1:
            break
        k = min((i for i, v in enumerate(working) if v > 2), key=lambda i: working[i])
        working[k] = 2.
    maximum = max(row["cdv"] for row in trace)
    return {"pci": 100 - maximum, "max_cdv": maximum, "m": m,
            "iterations": trace, "flags": sorted(set(flags))}


def score_combination(rows, area_m2, curves, scenario="M", policy="approximate"):
    if curves.get("application") != "road_asphalt_research":
        raise ValueError("This engine only accepts the road-asphalt research curve schema")
    groups = {}
    for row in rows:
        severity = row.get("severity") or scenario
        if severity not in ("L", "M", "H"):
            raise ValueError(f"Invalid severity {severity}")
        key = (row["distress"], severity, row["unit"])
        groups[key] = groups.get(key, 0.) + row["quantity"]
    deductions, flags = [], []
    for (kind, severity, unit), quantity in sorted(groups.items()):
        spec = curves["dv"][kind]
        if spec["unit"] != unit:
            raise ValueError(f"Incorrect unit for {kind}: {unit}")
        d = density(quantity, unit, area_m2)
        # Some FAA rows stop at DV=100 before the common density header ends.
        ys = spec['y'][severity]
        xs = spec['x'][:len(ys)]
        if d > xs[-1] and ys[-1] == 100:
            dv, notes = 100., [f'DV {kind}/{severity}: saturated at last displayed DV=100']
        else:
            dv, notes = interpolate(xs, ys, d,
                                log_x=spec["log_x"], policy=policy,
                                context=f"DV {kind}/{severity}")
        flags += notes
        dv, notes = bound_deduct(dv, f"DV {kind}/{severity}", policy)
        flags += notes
        deductions.append({"distress": kind, "severity": severity, "quantity": quantity,
                           "unit": unit, "density": d, "density_basis": spec["density_basis"], "dv": dv})
    result = corrected_pci([r["dv"] for r in deductions], curves, policy)
    result.update(deductions=deductions, scenario_for_unknown_severity=scenario)
    result["flags"] = sorted(set(flags + result["flags"]))
    return result


def score_inventory(rows, area_m2, curves, scenario='M', policy='approximate'):
    """Score and check upward severity-merge alternatives (D6433 multi-severity rule).

    All combinations are enumerated, avoiding dependence on distress ordering.
    Does not change the original inventory; traces distinguish scoring variants.
    """
    if not math.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError('Sample area must be positive and finite')
    grouped = {}
    for row in rows:
        kind, unit = row['distress'], row['unit']
        severity = row.get('severity') or scenario
        if severity not in ('L', 'M', 'H'):
            raise ValueError(f'Invalid severity: {severity}')
        quantity = float(row['quantity'])
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError('Quantity must be finite and nonnegative')
        if kind not in curves['dv']:
            raise ValueError(f'Unsupported road distress {kind!r}; no rows dropped')
        if unit != curves['dv'][kind]['unit']:
            raise ValueError(f'Incorrect unit for {kind}: {unit}')
        key = (kind, unit)
        grouped.setdefault(key, {})[severity] = grouped.setdefault(key, {}).get(severity, 0.) + quantity
    original = []
    options = []
    for (kind, unit), values in sorted(grouped.items()):
        values = {s: q for s,q in values.items() if q > 0}
        if unit == 'm2' and sum(values.values()) > area_m2 * (1+1e-8):
            raise ValueError(f'{kind} total area exceeds sample area; check duplicate masks/units')
        present = [s for s in ('L','M','H') if s in values]
        base = [{'distress':kind,'unit':unit,'severity':s,'quantity':values[s]} for s in present]
        original.extend(base)
        alternatives = [base]
        if len(present) == 2:
            alternatives.append([{'distress':kind,'unit':unit,'severity':present[-1],'quantity':sum(values.values())}])
        elif len(present) == 3:
            l,m,h = (values[s] for s in ('L','M','H'))
            for v in ({'M':l+m,'H':h},{'L':l,'H':m+h},{'M':m,'H':l+h},{'H':l+m+h}):
                alternatives.append([{'distress':kind,'unit':unit,'severity':s,'quantity':q} for s,q in v.items()])
        options.append(alternatives)
    combinations = math.prod(len(o) for o in options)
    if combinations > 4096:
        raise ValueError('Too many severity combinations; split/review sample inventory')
    baseline = score_combination(original, area_m2, curves, scenario, policy)
    best = baseline
    checks = []
    for n, combination in enumerate(itertools.product(*options)):
        inventory = [r for part in combination for r in part]
        result = baseline if n == 0 else score_combination(inventory, area_m2, curves, scenario, policy)
        checks.append({'candidate':n,'pci':result['pci'],'inventory':inventory,'flags':result['flags']})
        if result['pci'] > best['pci'] + 1e-10:
            best = result
    best = dict(best)
    best['base_pci_before_severity_correction'] = baseline['pci']
    best['base_deductions'] = baseline['deductions']
    best['severity_correction_candidates'] = checks
    best['flags'] = sorted(set(best['flags'] + [
        'FAA_displayed_tables_with_local_log_linear_DV_and_linear_CDV_interpolation',
        'linear_and_count_density_use_English_curve_basis_pending_server_crosscheck',
        'not_certified_ASTM_D6433_or_PAVEAIR_engine_equivalence']))
    if best['pci'] != baseline['pci']:
        best['flags'].append('multi_severity_upward_merge_correction_applied')
    return best
