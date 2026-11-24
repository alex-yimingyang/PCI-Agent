#!/usr/bin/env python3
"""Local road AC PCI from measured inventory. No GPU or third-party packages."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from pci_core import load_curves, score_inventory
from severity_rules import resolve

ROOT = Path(__file__).resolve().parent
ALIASES = {
    'longitudinal crack':'linear_crack','transverse crack':'linear_crack',
    'longitudinal_crack':'linear_crack','transverse_crack':'linear_crack',
    'alligator crack':'alligator','alligator cracking':'alligator',
    'block crack':'block_crack','block cracking':'block_crack',
    'patching':'patch','potholes':'pothole'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f'{path}: missing or duplicate headers')
        rows = list(reader)
    for n, row in enumerate(rows, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'{path}:{n}: malformed CSV')
    return [{k:v.strip() for k,v in r.items()} for r in rows]


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output exists. Use a new directory.')
    curve_path = Path(args.curves or ROOT/'curves/faa_road_ac_tables.json')
    curves = load_curves(curve_path)
    inventory, samples = read_csv(args.inventory), read_csv(args.results or args.samples)
    if not samples:
        raise ValueError('No samples')
    ids = [r.get('sample_id','') for r in samples]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError('Sample IDs must be present and unique')
    grouped = {sid:[] for sid in ids}
    for n,row in enumerate(inventory,2):
        if row.get('sample_id') not in grouped:
            raise ValueError(f'Inventory row {n}: sample not found')
        grouped[row['sample_id']].append(row)
    out.mkdir(parents=True)
    results, decisions, excluded, dv_rows, cdv_rows = [], [], [], [], []
    for sample in samples:
        sid = sample['sample_id']
        status = sample.get('status', 'ok')
        record = {'sample_id':sid, 'image':sample.get('image',''),
                  'status':'provisional', 'scenarios':{},
                  'flags':[sample.get('flags',''), 'selected_distress_inventory_only',
                           'model_measurements_and_severity_may_be_assumed']}
        if status not in ('ok','provisional'):
            excluded.append({**sample,'reason':f'upstream_status_{status}'})
            record.update(status='excluded',error=f'Incomplete upstream sample: {status}')
            results.append(record)
            continue
        try:
            area = float(sample.get('sample_area_m2') or sample.get('area_m2') or 'nan')
            if not math.isfinite(area) or area <= 0:
                raise ValueError('Sample area must be positive and finite')
            record['sample_area_m2'] = area
            rows = []
            for n,row in enumerate(grouped[sid]):
                label = row.get('distress','').casefold()
                kind = ALIASES.get(label,label)
                if kind not in curves['dv']:
                    raise ValueError(f'Unmapped road distress {label!r}; no rows dropped')
                unit = {'m²':'m2','m^2':'m2'}.get(row.get('unit'),row.get('unit'))
                quantity = float(row['quantity'])
                if not math.isfinite(quantity) or quantity <= 0:
                    raise ValueError('Quantity must be positive and finite')
                features = json.loads(row.get('severity_features') or '{}')
                severity, source, flags = resolve({**row,'severity_features':features},kind)
                if severity is None and args.unknown == 'error':
                    raise ValueError(f'{kind}: severity unknown; provide evidence or use --unknown scenarios')
                decisions.append({'sample_id':sid,'record_id':row.get('record_id',str(n)),
                    'distress':kind,'severity':severity or '', 'severity_source':source,
                    'reason':' | '.join(flags),'quantity':quantity,'unit':unit})
                record['flags'].extend(flags)
                if source in ('assumed','scenario_assumption','provided_unverified'):
                    record['flags'].append(f'{kind}:severity_source={source}')
                rows.append({'distress':kind,'severity':severity,'unit':unit,'quantity':quantity})
            if not rows and sample.get('confirmed_no_distress','').casefold() != 'true':
                raise ValueError('Empty inventory does not establish PCI 100')
            for scenario in ('L','M','H'):
                try:
                    result = score_inventory(rows,area,curves,scenario,args.curve_policy)
                    record['scenarios'][scenario] = result
                except (ValueError,KeyError) as exc:
                    record.setdefault('scenario_errors',{})[scenario] = str(exc)
            if not record['scenarios']:
                record['status'] = 'error'
                record['error'] = json.dumps(record['scenario_errors'],ensure_ascii=False)
            elif record.get('scenario_errors'):
                record['status'] = 'partial_scenarios'
            for scenario,result in record['scenarios'].items():
                for row in result['deductions']:
                    dv_rows.append({'sample_id':sid,'scenario':scenario,**row})
                for n,row in enumerate(result['iterations']):
                    cdv_rows.append({'sample_id':sid,'scenario':scenario,'iteration':n+1,
                        **row,'deduct_values':json.dumps(row['deduct_values'])})
        except (ValueError,KeyError,TypeError) as exc:
            record.update(status='error',error=str(exc),scenarios={})
        record['flags'] = sorted(set(f for f in record['flags'] if f))
        results.append(record)
    flat = []
    for r in results:
        flat.append({**{k:r.get(k,'') for k in ('sample_id','image','status','sample_area_m2','error')},
            'PCI_estimate':r['scenarios'].get('M',{}).get('pci',''),
            **{f'PCI_{s}':r['scenarios'].get(s,{}).get('pci','') for s in ('L','M','H')},
            'flags':' | '.join(r['flags'])})
    report = {'version':'road-pci-1.0', 'application':'road_asphalt_research',
              'official_pci':False,'curve_source':curves['source'],'curve_status':curves['status'],
              'curve_sha256':sha(curve_path),'inventory_sha256':sha(args.inventory),
              'samples_sha256':sha(args.results or args.samples),'arguments':vars(args),
              'notes':['Local rule-based estimation; does not call PAVEAIR.',
                       'Unknown L/M/H are sensitivity scenarios, not confidence bounds.',
                       'PCI_estimate is the M scenario; known severities remain fixed.',
                       'No road-section PCI is inferred from unrelated photos.'],
              'samples':results}
    (out/'pci_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'curves_used.json').write_bytes(curve_path.read_bytes())
    write_csv(out/'pci_results.csv',flat,['sample_id','image','status','sample_area_m2','PCI_estimate','PCI_L','PCI_M','PCI_H','flags','error'])
    write_csv(out/'severity_decisions.csv',decisions,['sample_id','record_id','distress','severity','severity_source','reason','quantity','unit'])
    write_csv(out/'deduct_values.csv',dv_rows,['sample_id','scenario','distress','severity','quantity','unit','density','density_basis','dv'])
    write_csv(out/'cdv_iterations.csv',cdv_rows,['sample_id','scenario','iteration','q','tdv','cdv','deduct_values'])
    write_csv(out/'excluded_samples.csv',excluded,list(dict.fromkeys(['sample_id','status','reason','flags','error']+[k for r in excluded for k in r])))
    n = sum(bool(r['scenarios']) for r in results)
    failures = sum(r['status'] in ('error','partial_scenarios') for r in results)
    print(f'Local ROAD PCI: {n}/{len(results)} samples scored; excluded={len(excluded)}; failures={failures}. Output: {out}')
    return 0 if n and not failures and (not excluded or args.on_incomplete == 'skip') else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',required=True,help='Measured distress_inventory.csv or prepared distresses.csv')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--results',help='Upstream pci_results.csv (area/status only; old PCI values ignored)')
    group.add_argument('--samples',help='samples.csv with sample_id and area_m2')
    p.add_argument('--out',required=True)
    p.add_argument('--curves')
    p.add_argument('--unknown',choices=['scenarios','error'],default='scenarios')
    p.add_argument('--curve-policy',choices=['approximate','strict'],default='approximate')
    p.add_argument('--on-incomplete',choices=['error','skip'],default='error',help='Both modes exclude incomplete samples; error also returns nonzero exit status')
    args = p.parse_args()
    try:
        return run(args)
    except (ValueError,KeyError,OSError,csv.Error) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
