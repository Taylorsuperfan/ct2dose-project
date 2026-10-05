"""Add a NEW model's real dose predictions to an already evaluated, identical cohort.

No synthetic residual/reference substitutions and no numerical rescaling are guessed.
The external manifest is an explicit claim by its producer; declarations are not a
proof that the producer avoided label leakage. Record producer code and checkpoints.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .io import read_json,write_json,sha,digest_object
from .evaluate import aggregate,verify_record
from .metrics import score


def compare_external(baseline_run,prediction_manifest,out):
    base=Path(baseline_run);out=Path(out)
    if out.exists():raise FileExistsError('Use a new external-comparison directory.')
    status=read_json(base/'run_status.json');contract=read_json(base/'contract.local.json')
    if status.get('status')!='completed_validation_inference':raise ValueError('Baseline cohort has not completed.')
    h=digest_object(contract)
    ext=read_json(prediction_manifest)
    required=['method','data_scope','cohort_sha256','prediction_kind','units','training_provenance','producer_source_sha256','checkpoint_sha256','records']
    if any(k not in ext for k in required):raise ValueError('Incomplete external prediction contract.')
    method=ext['method']
    if not isinstance(method,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{1,70}',method):raise ValueError('Invalid method name.')
    if method in contract['methods']:raise ValueError('Do not relabel/overwrite an existing baseline.')
    if ext['data_scope']!='real_validation' or ext['prediction_kind']!='absolute_dose' or ext['units']!='stored_target_units':
        raise ValueError('Only real absolute-dose predictions in the same stored units can enter this table; no synthetic results/residual-only input.')
    if ext['cohort_sha256']!=digest_object(contract['records']):raise ValueError('Different cohort.')
    if not ext['training_provenance']:raise ValueError('State training data/exposure provenance explicitly.')
    for key in ['producer_source_sha256','checkpoint_sha256']:
        if not isinstance(ext[key],str) or not re.fullmatch('[0-9a-f]{64}',ext[key]):raise ValueError(f'Missing valid {key}.')
    erows=ext['records']
    if not isinstance(erows,list):raise ValueError('records must be a list.')
    lookup={r['sample_id']:r for r in erows}
    if len(lookup)!=len(erows) or set(lookup)!={r['sample_id'] for r in contract['records']}:raise ValueError('Missing/duplicate/extra records.')
    rows=[]
    # Derive all baseline metrics from hash-verified per-record artifacts, not arbitrary CSVs.
    for r in contract['records']:
        folder=base/'records.local'/r['sample_id'];m=verify_record(folder,r,h)
        rows.extend(m['metrics']);er=lookup[r['sample_id']]
        if er['ct_sha256']!=m['ct_sha256']:raise ValueError('CT identity mismatch.')
        p=Path(er['prediction_path'])
        if not p.is_absolute():p=Path(prediction_manifest).parent/p
        if p.suffix.lower()!='.npy' or sha(p)!=er['prediction_sha256']:raise ValueError('External prediction file/hash mismatch.')
        pred=np.load(p,allow_pickle=False)
        if pred.shape!=(32,32,32) or not np.isfinite(pred).all():raise ValueError('Invalid external prediction.')
        with np.load(folder/'arrays.npz',allow_pickle=False) as a:target=a['target_stored'].copy()
        factor=m['dose_scale_factor']
        rows.append(dict(method=method,case_id=r['case_id'],record_id=r['record_id'],sample_id=r['sample_id'],
                         dose_scale_factor=factor,**score(pred*factor,target*factor,factor)))
    out.mkdir(parents=True)
    rec,cases,summary=aggregate(rows)
    rec.to_csv(out/'metrics_records.local.csv',index=False);cases.to_csv(out/'metrics_cases.local.csv',index=False)
    summary.to_csv(out/'comparison.csv',index=False)
    write_json(out/'comparison_receipt.json',dict(baseline_contract_sha256=h,external_manifest_sha256=sha(prediction_manifest),
        cohort_sha256=ext['cohort_sha256'],external_training_provenance=ext['training_provenance'],
        status='COMMON_COHORT_METRICS_COMPUTED_NOT_CAUSAL_OR_CLINICAL_CLAIM',
        limitation='Source/checkpoint hashes are supplied by producer; independently audit labels, splits and inference inputs.',
        scientific_task_status='PENDING_HUMAN_REVIEW'))
    print(summary.to_string(index=False));return out


def produce_external(baseline_run,out,method,predict_native,training_provenance,producer_source_sha256,checkpoint_sha256):
    """Run an already-trained NEW model callback on the same CT-only cohort.

    predict_native receives a copy of raw CT only and must return absolute dose
    in stored numerical units. It does not receive targets, sample IDs or paths.
    This is an interface, not an implementation/training of a new HJD model.
    """
    from .data import read_array
    from .io import write_json
    base=Path(baseline_run);out=Path(out)
    if out.exists():raise FileExistsError('Use a new prediction export directory.')
    c=read_json(base/'contract.local.json');status=read_json(base/'run_status.json')
    if status.get('status')!='completed_validation_inference':raise ValueError('Baseline evaluation is not complete.')
    for label,v in [('source',producer_source_sha256),('checkpoint',checkpoint_sha256)]:
        if not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v):raise ValueError('Invalid producer '+label+' hash.')
    if not callable(predict_native) or not training_provenance:raise ValueError('Supply a predictor and truthful training provenance.')
    out.mkdir(parents=True);records=[]
    for r in c['records']:
        ct,h=read_array(r['input_path'])
        baseline_meta=read_json(base/'records.local'/r['sample_id']/'record.json')
        if h!=baseline_meta['ct_sha256']:raise IOError('Input CT changed.')
        pred=np.asarray(predict_native(ct.copy()))
        if pred.shape!=(32,32,32) or not np.isfinite(pred).all():raise ValueError('Invalid new-model prediction.')
        path=out/(r['sample_id']+'.npy');np.save(path,pred.astype(np.float32),allow_pickle=False)
        records.append(dict(sample_id=r['sample_id'],ct_sha256=h,prediction_path=path.name,prediction_sha256=sha(path)))
    manifest=dict(schema='real_dose_predictions_v1',method=method,data_scope='real_validation',
        cohort_sha256=digest_object(c['records']),prediction_kind='absolute_dose',units='stored_target_units',
        training_provenance=training_provenance,producer_source_sha256=producer_source_sha256,
        checkpoint_sha256=checkpoint_sha256,records=records)
    write_json(out/'predictions.local.json',manifest)
    return out/'predictions.local.json'
