"""Fixed validation records only. This module never opens the historical test JSON."""
from pathlib import Path
from collections import Counter
import numpy as np
from .io import read_json,sha,digest_object

SPLIT='data/splits/strict_case_level_v2_seed42'
TRAIN_NAME='train_pairs_3d_strict_train6_full.json'
VAL_NAME='val_pairs_3d_strict_val2_full.json'
SUBSET_NAME='eval_subsets/val2_stratified_300_per_case.json'


def parse_records(path):
    obj=read_json(path)
    if not isinstance(obj,list) or not obj:raise ValueError(f'Expected nonempty original list manifest: {path}')
    out=[]
    for r in obj:
        if not isinstance(r,dict) or not all(isinstance(r.get(k),str) and r[k] for k in ['case_id','input_path','output_path']):
            raise ValueError('Missing explicit string case_id/input_path/output_path. No identifier inference.')
        out.append({k:r[k] for k in ['case_id','input_path','output_path']})
    if len({(r['case_id'],r['input_path'],r['output_path']) for r in out})!=len(out):raise ValueError('Duplicate manifest records.')
    return out


def fixed_cohort(root,mode='smoke'):
    root=Path(root).resolve();sd=root/SPLIT
    train= parse_records(sd/TRAIN_NAME);val=parse_records(sd/VAL_NAME)
    tc={r['case_id'] for r in train};vc={r['case_id'] for r in val}
    if tc&vc:raise ValueError('Current train and validation case IDs overlap.')
    for key in ['input_path','output_path']:
        if {r[key] for r in train}&{r[key] for r in val}:raise ValueError('Current train/validation file paths overlap.')
    sources={TRAIN_NAME:sha(sd/TRAIN_NAME),VAL_NAME:sha(sd/VAL_NAME)}
    if mode=='historical-val600':
        sp=sd/SUBSET_NAME
        if not sp.is_file():raise FileNotFoundError(f'Exact archived validation subset is required; do not regenerate it: {sp}')
        selected=parse_records(sp);sources[SUBSET_NAME]=sha(sp)
        allowed={(r['case_id'],r['input_path'],r['output_path']) for r in val}
        if any((r['case_id'],r['input_path'],r['output_path']) not in allowed for r in selected):raise ValueError('Saved subset is not contained in current validation.')
        counts=Counter(r['case_id'] for r in selected)
        if len(selected)!=600 or set(counts)!=vc or len(vc)!=2 or set(counts.values())!={300}:
            raise ValueError('Historical mode requires exactly the saved 2-case, 300-records-per-case validation subset.')
    elif mode=='smoke':
        selected=[];counts=Counter()
        for r in val:
            if counts[r['case_id']]<2:selected.append(r);counts[r['case_id']]+=1
    else:raise ValueError('Only smoke and historical-val600 are supported. No test option exists.')
    rawroot=(root/'data/raw').resolve()
    for r in selected:
        for key in ['input_path','output_path']:
            p=Path(r[key]).resolve()
            if not p.is_relative_to(rawroot):raise ValueError('Selected array outside declared original data/raw. Do not remap silently.')
        r['record_id']=Path(r['input_path']).stem
        r['sample_id']=digest_object({k:r[k] for k in ['case_id','input_path','output_path']})[:20]
    if len({r['sample_id'] for r in selected})!=len(selected):raise ValueError('Sample-ID collision.')
    return selected,dict(mode=mode,partition='validation',source_kind='real',manifest_hashes=sources,
        n_records=len(selected),n_cases=len(vc),case_counts=dict(Counter(r['case_id'] for r in selected)),
        current_case_disjoint=True,patient_independence='unknown',legacy_checkpoint_training_exposure='not established by this code',
        old_test_json_opened=False,record_selection='preselected by manifest order; not by prediction or target values')


def read_array(path):
    p=Path(path);before=sha(p)
    a=np.load(p,allow_pickle=False)
    if not isinstance(a,np.ndarray) or a.shape!=(32,32,32) or not np.issubdtype(a.dtype,np.number) or not np.isfinite(a).all():
        raise ValueError(f'Invalid original array: {p}')
    if before!=sha(p):raise IOError(f'Array changed while reading: {p}')
    return a.astype(np.float32),before
