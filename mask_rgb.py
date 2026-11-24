"""Reveal aligned surface evidence only after the mask-only classification step."""
import json
from pathlib import Path
import os
from tempfile import NamedTemporaryFile
import image_agents as vision
import pci_workflow as wf

def immutable(path, data):
    """Publish complete stage outputs once. Identical retries are harmless."""
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(encoded)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if path.read_bytes() != encoded:
                raise ValueError('This stage already has different decisions. Start a fresh run.')
    finally:
        temp.unlink()


def classes(rows):
    keys = ('mask_id','box','kind','preferred_crack_kind','class_candidates','semantics')
    return [{key: row.get(key) for key in keys} for row in rows]

def read_crops(measurement_run_id, image_path, cursor, limit):
    if len(measurement_run_id)!=32 or any(c not in '0123456789abcdef' for c in measurement_run_id):
        raise ValueError('Expected a measurement run ID.')
    measured=wf.RUNS/measurement_run_id
    summary_file=measured/'class_pci.json'
    if not summary_file.exists():
        summary_file=measured/'mask_pci.json'
    result=json.loads(summary_file.read_text())
    if result['status']!='needs_visual_severity_assessment' or result['workflow']!=vision.MODES['mask']:
        raise ValueError('Run mask-only classification and measure_only=true first.')
    prepared,meta=vision.load_run(result['prepared_run_id'],'mask')
    rows=json.loads(Path(result.get('family_judgments_json') or result['judgments_json']).read_text())
    if any(not row['kind'] for row in rows):
        raise ValueError('Record an explicit mask-only class or unknown for each candidate first.')
    if not 0<=cursor<len(rows):
        raise ValueError('Cursor outside the instance list.')
    path=Path(image_path).expanduser().resolve(strict=True)
    evidence={'path':str(path),'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
    crop_run=wf.new_run()
    batch=vision.worker('grading_crops',crop_run,dict(image=str(path),width=meta['width'],height=meta['height'],
        bounds={m['mask_id']:m['bounds_xyxy'] for m in meta['masks']},
        rows=rows[cursor:cursor+limit],cursor=cursor))
    immutable(prepared/'grading_rgb.json',dict(classes=classes(rows),rgb=evidence))
    return vision.image_result(dict(run_id=prepared.name,images=batch,total=len(rows),
        next_cursor=cursor+limit if cursor+limit<len(rows) else None),batch)
