"""Bound PCI over admissible severity assignments using the local Road AC engine."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import math
import random

import numpy as np

import pci_core

GRADES = ("L", "M", "H")
EPS = 1e-9


@dataclass(frozen=True)
class Entry:
    kind: str
    unit: str
    quantity: float
    low: int
    high: int


def deduct(curves, kind, grade, density):
    spec = curves["dv"][kind]
    y = spec["y"][GRADES[grade]]
    x = np.array(spec["x"][:len(y)])
    d = np.maximum(density, 0.)
    return np.clip(np.where(d < x[0], y[0]*d/x[0],
                            np.interp(np.log(np.maximum(d, x[0])), np.log(x), y)), 0, 100)


def cdv(curves, q, total, upper):
    if q <= 1:
        return np.minimum(total, 100.)
    table = curves["cdv"]
    if str(q) in table:
        c = table[str(q)]
        return np.interp(total, c["x"], c["y"])
    left = [int(k) for k in table if int(k) < q]
    right = [int(k) for k in table if int(k) > q]
    if not left or not right:
        # Missing outer q rows cannot produce a point estimate. Their contribution
        # is enclosed by the standard corrected-deduct limits instead.
        return np.minimum(total, 100.) if upper else np.zeros_like(total)
    l, r = max(left), min(right)
    a, b = table[str(l)], table[str(r)]
    if a["x"] != b["x"]:
        raise ValueError("Cannot interpolate CDV tables with different abscissae")
    y = np.array(a["y"]) + (np.array(b["y"])-a["y"])*(q-l)/(r-l)
    return np.interp(total, a["x"], y)


def corrected_enclosure(curves, lo, hi):
    lo = np.sort(lo, axis=-1)[..., ::-1]
    hi = np.sort(hi, axis=-1)[..., ::-1]
    ranks = np.arange(lo.shape[-1])
    mlo = np.minimum(10., 1+9/98*(100-hi[..., 0]))
    mhi = np.minimum(10., 1+9/98*(100-lo[..., 0]))
    can_truncate = (hi > 2).sum(axis=-1) > 1
    must_truncate = (lo > 2).sum(axis=-1) > 1
    lo = lo*np.where(can_truncate[..., None], np.clip(mlo[..., None]-ranks, 0, 1), 1)
    hi = hi*np.where(must_truncate[..., None], np.clip(mhi[..., None]-ranks, 0, 1), 1)
    front_lo, front_hi = np.cumsum(lo, axis=-1), np.cumsum(hi, axis=-1)
    back_lo = np.cumsum(np.minimum(lo, 2)[..., ::-1], axis=-1)[..., ::-1]
    back_hi = np.cumsum(np.minimum(hi, 2)[..., ::-1], axis=-1)[..., ::-1]
    lower = np.zeros(lo.shape[:-1])
    upper = np.zeros_like(lower)
    for q in range(1, min(lo.shape[-1], 10)+1):
        a = front_lo[..., q-1]+(back_lo[..., q] if q < lo.shape[-1] else 0.)
        b = front_hi[..., q-1]+(back_hi[..., q] if q < hi.shape[-1] else 0.)
        l, h = cdv(curves, q, a, False), cdv(curves, q, b, True)
        if q > 1:
            l = np.where(lo[..., q-1] > 2, l, 0.)
            h = np.where(hi[..., q-1] > 2, h, 0.)
        lower, upper = np.maximum(lower, l), np.maximum(upper, h)
    return lower, upper


def grading_options(presence):
    """Same upward merges as pci_core.score_inventory, as source-grade groups."""
    base = ((0,), (1,), (2,))
    if len(presence) == 2:
        merged = [(), (), ()]
        merged[presence[-1]] = presence
        return (base, tuple(merged))
    if len(presence) == 3:
        return (base, ((), (0, 1), (2,)), ((0,), (), (1, 2)),
                ((), (1,), (0, 2)), ((), (), (0, 1, 2)))
    return (base,)


def dv_options(curves, kind, lo, hi, total, presence):
    grades = {0, 1, 2}
    lower, upper = [], []
    for groups in grading_options(presence):
        options = []
        for index, members in enumerate(groups):
            if not members:
                zero = np.zeros(len(lo))
                options.append((zero, zero))
                continue
            remaining = sorted(grades-set(members))
            a = np.maximum(lo[:, members].sum(axis=1), total-hi[:, remaining].sum(axis=1))
            b = np.minimum(hi[:, members].sum(axis=1), total-lo[:, remaining].sum(axis=1))
            # The output includes a small numerical cushion, also across curved table knots.
            options.append((deduct(curves, kind, index, a-EPS), deduct(curves, kind, index, b+EPS)))
        lower.append(np.column_stack([a for a, _ in options]))
        upper.append(np.column_stack([b for _, b in options]))
    return np.stack(lower, axis=1), np.stack(upper, axis=1)


def eval_signature(curves, kinds, totals, lo, hi, signature):
    tables = [dv_options(curves, k, lo[:,i], hi[:,i], totals[i], signature[i])
              for i,k in enumerate(kinds)]
    combos = np.array(list(product(*(range(a.shape[1]) for a, _ in tables))))
    lower = np.concatenate([a[:,combos[:,i],:] for i,(a,b) in enumerate(tables)], axis=2)
    upper = np.concatenate([b[:,combos[:,i],:] for i,(a,b) in enumerate(tables)], axis=2)
    a, b = corrected_enclosure(curves, lower, upper)
    return np.maximum(0., 100-b.min(axis=1)-EPS), np.minimum(100., 100-a.min(axis=1)+EPS)


def eval_boxes(curves, kinds, totals, lo, hi, signatures):
    groups = defaultdict(list)
    for i,s in enumerate(signatures):
        groups[s].append(i)
    bounds = np.empty((len(lo),2))
    for s, positions in groups.items():
        ncombos = math.prod(len(grading_options(p)) for p in s)
        stride = max(1, 100000//ncombos)
        for j in range(0,len(positions),stride):
            idx = positions[j:j+stride]
            a,b = eval_signature(curves,kinds,totals,lo[idx],hi[idx],s)
            bounds[idx] = np.column_stack((a,b))
    return bounds


def presence_cases(values):
    result = []
    for bits in range(1,8):
        presence = tuple(g for g in range(3) if bits & (1<<g))
        allowed = [(q,tuple(g for g in presence if a<=g<=b)) for q,a,b in values]
        if any(not possible for _,possible in allowed):
            continue
        hi = np.array([math.fsum(q for q,p in allowed if g in p) for g in range(3)])
        if any(hi[g] <= 0 for g in presence):
            continue
        lo = np.array([max(math.fsum(q for q,p in allowed if p==(g,)),
                           min((q for q,p in allowed if g in p),default=0.)) for g in range(3)])
        result.append((lo,hi,presence))
    if not result:
        raise ValueError("No feasible severity assignment for distress class")
    return result


def exact_cases(values):
    out = {}
    for assignment in product(*(range(a,b+1) for _,a,b in values)):
        amount = tuple(math.fsum(v[0] for v,g in zip(values,assignment) if g==i) for i in range(3))
        out[amount] = tuple(g for g,q in enumerate(amount) if q>0)
    return [(np.array(a),np.array(a),s) for a,s in out.items()]


def start_boxes(grouped):
    cases = [presence_cases(v) for v in grouped]
    count = math.prod(len(c) for c in cases)
    for i,values in enumerate(grouped):
        combinations = math.prod(b-a+1 for _,a,b in values)
        if combinations<=81 and count//len(cases[i])*combinations<=4096:
            updated = exact_cases(values)
            count = count//len(cases[i])*len(updated)
            cases[i] = updated
    lo, hi, sig = [], [], []
    for combo in product(*cases):
        a,b,s = zip(*combo)
        lo.append(a)
        hi.append(b)
        sig.append(s)
    return np.array(lo),np.array(hi),sig


def project(lo, hi, totals):
    # Every distress has a fixed total. Tighten each coordinate by the other two.
    total = totals[None,:,None]
    lo = np.maximum(lo, total-(hi.sum(axis=-1,keepdims=True)-hi)-EPS)
    hi = np.minimum(hi, total-(lo.sum(axis=-1,keepdims=True)-lo)+EPS)
    valid = ((lo<=hi+EPS).all(axis=(1,2)) & (lo.sum(axis=-1)<=totals+EPS).all(axis=1)
             & (hi.sum(axis=-1)>=totals-EPS).all(axis=1))
    # No negative quantities and no reversed intervals from floating point rounding.
    return lo[valid].clip(0),np.maximum(lo[valid],hi[valid]).clip(0),np.flatnonzero(valid)


def feasible_witnesses(curves, kinds, totals, grouped, entries, area):
    points, signatures = [], []
    uniform = {}
    rng = random.Random(318)
    for trial in range(27):
        assignment = [[max(a,min(b,trial)) if trial<3 else rng.randint(a,b) for _,a,b in v] for v in grouped]
        point = np.array([[math.fsum(q for (q,_,_),g in zip(v,gs) if g==s) for s in range(3)]
                          for v,gs in zip(grouped,assignment)])
        points.append(point)
        signatures.append(tuple(tuple(np.flatnonzero(row>0)) for row in point))
        if trial>=3:
            continue
        inventory = [dict(distress=e.kind,unit=e.unit,quantity=e.quantity,
                          severity=GRADES[max(e.low,min(e.high,trial))]) for e in entries]
        try:
            info = pci_core.score_inventory(inventory,area,curves)
            info.pop("scenario_for_unknown_severity",None)
            uniform[GRADES[trial]] = info
        except ValueError as exc:
            if "no q=" not in str(exc):
                raise
            uniform[GRADES[trial]] = {"pci":None,"reason":str(exc)}
    points = np.array(points)
    b = eval_boxes(curves,kinds,totals,points,points,signatures)
    for i,g in enumerate(GRADES):
        exact = uniform[g]["pci"]
        if exact is not None and not b[i,0]-1e-6 <= exact <= b[i,1]+1e-6:
            raise RuntimeError("PCI engine and interval scorer disagree on a feasible assignment")
        uniform[g]["point_interval"] = b[i].tolist()
    return float(b[:,1].min()),float(b[:,0].max()),uniform


def bound(entries, area, curves, tolerance=.01, box_limit=50000):
    """Return an outer envelope, including when the computation hits its search limit."""
    if not math.isfinite(area) or area<=0 or not math.isfinite(tolerance) or tolerance<=0 or box_limit<1:
        raise ValueError("Positive area, tolerance and box limit required")
    entries = [e for e in entries if e.quantity>0]
    if not entries:
        return dict(lower=100.,upper=100.,min_witness_ceiling=100.,max_witness_floor=100.,
                    boxes_evaluated=0,converged=True,search_stop="empty_inventory",uniform={})
    grouped = defaultdict(list)
    for e in entries:
        if curves["dv"][e.kind]["unit"] != e.unit or not 0<=e.low<=e.high<=2:
            raise ValueError(f"Invalid PCI unit or severity interval for {e.kind}")
        grouped[e.kind].append((pci_core.density(e.quantity,e.unit,area),e.low,e.high))
    kinds = sorted(grouped)
    quantities = [grouped[k] for k in kinds]
    totals = np.array([math.fsum(q for q,_,_ in v) for v in quantities])
    min_ceiling,max_floor,uniform = feasible_witnesses(curves,kinds,totals,quantities,entries,area)
    lo,hi,sig = start_boxes(quantities)
    checked=0
    stop="unresolved_curve_or_numeric_interval"
    envelope=[min_ceiling,max_floor]
    while len(lo):
        lo,hi,valid = project(lo,hi,totals)
        sig = [sig[i] for i in valid]
        if not len(lo):
            break
        b=eval_boxes(curves,kinds,totals,lo,hi,sig)
        checked+=len(lo)
        widths=(hi-lo).reshape(len(lo),-1)
        active=((b[:,0] < min_ceiling-tolerance)|(b[:,1] > max_floor+tolerance))
        active &= widths.max(axis=1)>EPS*8
        if checked+2*int(active.sum())>box_limit:
            stop="box_limit"
            active[:]=False
        terminal=b[~active]
        if len(terminal):
            envelope[0]=min(envelope[0],float(terminal[:,0].min()))
            envelope[1]=max(envelope[1],float(terminal[:,1].max()))
        if not active.any():
            break
        axes=widths[active].argmax(axis=1)
        sig=[s for s,k in zip(sig,active) if k]
        a,b=lo[active].copy(),hi[active].copy()
        left_hi,right_lo=b.copy(),a.copy()
        idx=np.arange(len(a))
        cut=(a.reshape(len(a),-1)[idx,axes]+b.reshape(len(b),-1)[idx,axes])/2
        left_hi.reshape(len(a),-1)[idx,axes]=cut
        right_lo.reshape(len(a),-1)[idx,axes]=cut
        lo,hi=np.concatenate((a,right_lo)),np.concatenate((left_hi,b))
        sig*=2
    if checked==0:
        raise RuntimeError("No feasible PCI interval boxes")
    gap=max(min_ceiling-envelope[0],envelope[1]-max_floor)
    return dict(lower=envelope[0],upper=envelope[1],min_witness_ceiling=min_ceiling,
                max_witness_floor=max_floor,boxes_evaluated=checked,converged=gap<=tolerance+EPS,
                search_stop=stop if gap>tolerance+EPS else "tolerance",uniform=uniform)
