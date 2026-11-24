"""Road AC feature-to-severity rules. Missing visual/depth evidence stays unknown.

Sources: FAA Roadway Distresses (AC), severitylevels40.htm (linear),
levelsofseverityforpotholes.htm / severitylevels43.htm (potholes).
Rules consume observed/model-estimated features; masks alone do not supply them.
"""
import math


def finite(value, name):
    v = float(value)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return v


def classify(kind, features):
    f = features or {}
    if kind == 'linear_crack':
        random = f.get('secondary_cracking')  # none, L, M, H
        if random not in (None, 'none', 'L', 'M', 'H'):
            raise ValueError('secondary_cracking: use none/L/M/H')
        broken = f.get('severely_broken_100mm_band')
        filled = f.get('filled')
        for name in ('filled', 'filler_satisfactory', 'severely_broken_100mm_band'):
            if name in f and type(f[name]) is not bool:
                raise ValueError(f'{name} must be JSON true/false')
        width = finite(f['mean_width_mm'], 'mean_width_mm') if f.get('mean_width_mm') is not None else None
        if broken is True or random in ('M','H') or (filled is False and width is not None and width > 75):
            return 'H', 'road_linear_high_condition'
        if random is None or broken is None:
            return None, 'missing_secondary_cracking_or_edge_condition'
        if random == 'L':
            return 'M', 'road_linear_light_secondary_cracking'
        if filled is True and f.get('filler_satisfactory') is True:
            return 'L', 'road_linear_satisfactory_filler'
        if filled is False and width is not None:
            if width < 10:
                return 'L', 'road_linear_unfilled_width_below_10mm'
            if width < 75:
                return 'M', 'road_linear_unfilled_width_10_to_below_75mm'
            return None, 'exact_75mm_boundary_requires_review'
        return None, 'missing_filling_or_width_evidence'
    if kind == 'pothole':
        if f.get('average_diameter_mm') is None or f.get('max_depth_mm') is None:
            return None, 'missing_pothole_diameter_or_depth'
        d = finite(f['average_diameter_mm'], 'average_diameter_mm')
        h = finite(f['max_depth_mm'], 'max_depth_mm')
        if d > 750:
            return ('M' if h <= 25 else 'H'), 'road_large_pothole_depth_rule'
        if d < 100 or h < 13:
            return None, 'outside_published_pothole_table'
        if d in (200,450):
            return None, 'shared_diameter_bin_boundary_requires_review'
        col = 0 if d < 200 else 1 if d < 450 else 2
        row = 0 if h <= 25 else 1 if h <= 50 else 2
        return (('L','L','M'),('L','M','H'),('M','M','H'))[row][col], 'road_pothole_depth_diameter_table'
    # Morphological grades cannot be inferred from mask area alone.
    return None, 'requires_type_specific_observation_or_provided_severity'


def resolve(item, kind, width_mm=None, allow_width_proxy=False):
    severity = item.get('severity')
    if severity in ('', 'UNKNOWN', 'unknown'):
        severity = None
    if severity not in (None,'L','M','H'):
        raise ValueError(f'Invalid severity {severity!r}')
    if severity:
        return severity, item.get('severity_source') or 'provided_unverified', []
    features = dict(item.get('severity_features') or {})
    value, reason = classify(kind, features)
    if value:
        return value, 'road_rule_from_supplied_features', [reason, 'severity_features_require_independent_validation']
    if allow_width_proxy and kind == 'linear_crack' and width_mm is not None:
        if item.get('mask_semantics') != 'crack_pixels':
            return None, 'scenario_assumption', ['width_proxy_requires_crack_pixels_mask', reason]
        proxy = {**features, 'mean_width_mm':width_mm, 'filled':False,
                 'secondary_cracking':'none', 'severely_broken_100mm_band':False}
        value, reason = classify(kind, proxy)
        return value, 'width_only_assumption' if value else 'scenario_assumption', [
            'assumed_unfilled_no_secondary_cracks_no_edge_breakup', reason]
    return None, 'scenario_assumption', [reason]
