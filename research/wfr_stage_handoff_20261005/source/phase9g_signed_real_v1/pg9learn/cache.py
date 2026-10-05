"""Versioned real-data cache: frozen Phase9G, real train6 labels, saved val600.

Never opens the historical/final test JSON. No random/artificial reference exists.
The cache is copied to local RAM once for training, avoiding Drive reads per batch.
"""
from pathlib import Path
from collections import Counter
import copy, json, io, math
import numpy as np
import torch
from p10recover.data import SPLIT, TRAIN_NAME, VAL_NAME, SUBSET_NAME, parse_records, read_array
from p10recover.io import sha, read_json, write_json, digest_object, save_arrays
from .common import SCOPE, code_hashes, single_writer


def key(r):return tuple(r[k] for k in ['case_id','input_path','output_path'])

def with_ids(r):
    r=dict(r);r['record_id']=Path(r['input_path']).stem
    r['sample_id']=digest_object({k:r[k] for k in ['case_id','input_path','output_path']})[:20]
    return r


def subset_by_case(rows,n,seed):
    """Hash-order sampling depends on identifiers only, never dose/model errors."""
    if n<1:raise ValueError('Positive records-per-case required.')
    selected=[]
    for case in sorted({r['case_id'] for r in rows}):
        group=[r for r in rows if r['case_id']==case]
        if len(group)<n:raise ValueError('Not enough records in a selected case.')
        selected += sorted(group,key=lambda r:digest_object({'seed':seed,'row':key(r)}))[:n]
    return selected


def select_real(root,baseline_run,n_per_case=32,seed=42):
    root=Path(root).resolve();old=Path(baseline_run).resolve();sd=root/SPLIT
    train=parse_records(sd/TRAIN_NAME);val=parse_records(sd/VAL_NAME);sub=parse_records(sd/SUBSET_NAME)
    tc=Counter(r['case_id'] for r in train);vc=Counter(r['case_id'] for r in val)
    if len(tc)!=6 or len(vc)!=2 or set(tc)&set(vc):raise ValueError('Expected disjoint real train6/val2 case manifests.')
    for field in ['input_path','output_path']:
        if {r[field] for r in train}&{r[field] for r in val}:raise ValueError('Train/validation path overlap.')
    counts=Counter(r['case_id'] for r in sub)
    if len(sub)!=600 or set(counts)!=set(vc) or set(counts.values())!={300}:raise ValueError('Saved validation subset must be 2x300.')
    allowed={key(r) for r in val}
    if any(key(r) not in allowed for r in sub):raise ValueError('Saved subset is not part of validation.')
    bc=read_json(old/'contract.local.json');bs=read_json(old/'run_status.json')
    if bs.get('status')!='completed_validation_inference' or bs.get('n_records')!=600:raise ValueError('The restored 600-record evaluation must already be complete.')
    if bs.get('contract_sha256')!=digest_object(bc):raise ValueError('Baseline status/contract mismatch.')
    if [key(r) for r in bc['records']] != [key(r) for r in sub]:raise ValueError('Current val600 differs from the previously evaluated cohort/order.')
    tr=[with_ids(r) for r in subset_by_case(train,n_per_case,seed)]
    va=[with_ids(r) for r in sub]
    for row in tr+va:
        for k in ['input_path','output_path']:
            if not Path(row[k]).resolve().is_relative_to(root/'data/raw'):raise ValueError('Array outside declared original raw tree.')
    # No assertion of independent patients or unseen upstream labels is made.
    info=dict(scope=SCOPE,seed=seed,n_train_records=len(tr),n_validation_records=len(va),
        train_case_counts=dict(Counter(r['case_id'] for r in tr)),validation_case_counts=dict(counts),
        full_train_manifest_records=len(train),selection='identifier-hash order per case; target-independent',
        manifest_hashes={name:sha(sd/name) for name in [TRAIN_NAME,VAL_NAME,SUBSET_NAME]},
        upstream_exposure='historically reused upstream checkpoints; not independently blind end-to-end',
        historical_test_json_opened=False,final_test_arrays_opened=0)
    return tr,va,info,bc


def validate_old_record(old,row,bc_hash):
    folder=Path(old)/'records.local'/row['sample_id']
    m=read_json(folder/'record.json')
    if m['sample_id']!=row['sample_id'] or m['contract_sha256']!=bc_hash:raise ValueError('Saved validation record identity mismatch.')
    a_path=folder/'arrays.npz'
    if sha(a_path)!=m['arrays_sha256']:raise IOError('Saved validation arrays changed.')
    with np.load(a_path,allow_pickle=False) as a:
        needed=['ct_normalized','target_stored','phase9g','phase10d_strict']
        if not set(needed).issubset(a.files):raise ValueError('Saved original predictions incomplete.')
        arrays={k:a[k].astype(np.float32,copy=True) for k in needed}
    if any(x.shape!=(32,32,32) or not np.isfinite(x).all() for x in arrays.values()):raise ValueError('Invalid saved validation array.')
    factor=float(m['dose_scale_factor'])
    if factor<=0 or not math.isfinite(factor):raise ValueError('Invalid legacy scaling factor.')
    # These are the exact saved legacy numerical conventions, not per-record inference calibration.
    clean=dict(ct=arrays['ct_normalized'],base=arrays['phase9g']*factor,
               target=arrays['target_stored']*factor)
    return clean,m,sha(folder/'record.json')


def cached_record(folder,row,contract_hash,verify_inputs=False):
    m=read_json(folder/'record.json')
    if m['sample_id']!=row['sample_id'] or m['contract_sha256']!=contract_hash:raise ValueError('Cache metadata identity mismatch.')
    if sha(folder/'arrays.npz')!=m['arrays_sha256']:raise IOError('Cache contents changed.')
    if verify_inputs:
        for field,h in [('input_path','ct_sha256'),('output_path','target_sha256')]:
            if sha(row[field])!=m[h]:raise IOError('Training source changed; refuse to silently update the frozen cache.')
    return m


def fit_norm(train_arrays):
    """One common residual RMS and input dose scale, fitted on train ONLY."""
    if not train_arrays:raise ValueError('No training arrays.')
    squares=[];bmax=0.;mass=[]
    for a in train_arrays:
        r=a['target'].astype(np.float64)-a['base'].astype(np.float64)
        squares.append(float(np.mean(r*r)));bmax=max(bmax,float(np.max(a['base'])))
        mass.append([float(np.maximum(r,0).mean()),float(np.maximum(-r,0).mean())])
    s=float(np.sqrt(np.mean(squares)))
    if not math.isfinite(s) or s<=0:raise ValueError('All selected training residuals are zero; no correction-learning task exists.')
    if not math.isfinite(bmax) or bmax<=0:raise ValueError('Nonpositive frozen Phase9G normalization scale.')
    return dict(fitted_on='selected_train6_only',residual_rms_model_scale=s,base_input_scale=bmax,
        initial_mass_mean_scaled=(np.mean(mass,axis=0)/s).tolist(),mass_convention='voxel sum, not energy',
        correction_target='real GT minus frozen Phase9G prediction',synthetic_reference=False,water_reference=False)


def audit_row(a,meta):
    r=a['target'].astype(float)-a['base'].astype(float)
    plus=np.maximum(r,0);minus=np.maximum(-r,0)
    return dict(sample_id=meta['sample_id'],case_id=meta['case_id'],split=meta['split'],
        residual_min=float(r.min()),residual_max=float(r.max()),positive_fraction=float((r>0).mean()),negative_fraction=float((r<0).mean()),
        positive_mass_model=float(plus.sum()),negative_mass_model=float(minus.sum()),
        reconstruction_max_abs=float(np.max(np.abs(r-plus+minus))),gt_overlap=float(np.mean(plus*minus)))


def prepare(root,baseline_run,out,pipeline,n_per_case=32,seed=42):
    """Computes new frozen predictions ONLY for the selected training records.
    Reuses every original validation prediction. Targets never enter upstream predict.
    """
    out=Path(out).resolve();root=Path(root).resolve();old=Path(baseline_run).resolve()
    if out.is_relative_to(root) or out==old or out.is_relative_to(old):raise ValueError('New cache must not be written into old data/evaluation directories.')
    tr,va,info,bc=select_real(root,old,n_per_case,seed)
    if pipeline.contract!=bc['pipeline']:raise ValueError('Frozen pipeline configuration/weights differ from the evaluated old system.')
    # Ensure the recovered base implementation is unchanged; wrapper models are new.
    from p10recover.io import code_hashes as legacy_hashes
    if legacy_hashes()!=bc['code_sha256']:raise ValueError('Recovered legacy source changed from the baseline run.')
    from .common import device_identity
    c=dict(schema='real_phase9g_cache_v1',selection=info,train=tr,validation=va,
           baseline_run=str(old),baseline_contract_sha256=digest_object(bc),frozen_pipeline=bc['pipeline'],
           producer_code=code_hashes(),train_prediction_environment=device_identity(pipeline.device))
    out.mkdir(parents=True,exist_ok=True)
    with single_writer(out):
        cp=out/'cache_contract.local.json'
        if cp.exists():
            if read_json(cp)!=c:raise ValueError('Cache contract differs. Use a NEW cache_id; never overwrite the cache.')
        else:
            if any(out.iterdir()):raise ValueError('Nonempty cache directory without contract.')
            write_json(cp,c)
        ch=digest_object(c);train_arrays=[];index=[];audits=[]
        for split,rows in [('train',tr),('validation',va)]:
            for j,row in enumerate(rows):
                if split=='validation':
                    # Reuse the existing files directly. Do NOT duplicate 600 dose archives on Drive.
                    src=old/'records.local'/row['sample_id']/'record.json'
                    m=read_json(src)
                    if m['contract_sha256']!=digest_object(bc) or m['sample_id']!=row['sample_id']:
                        raise ValueError('Saved validation metadata differs from the frozen cohort.')
                    index.append(dict(**row,split=split,storage='saved_legacy_validation',
                        record_meta_sha256=sha(src),arrays_sha256=m['arrays_sha256']))
                    if (j+1)%100==0 or j+1==len(rows):print(f'validation references {j+1}/{len(rows)} (no duplicated arrays)',flush=True)
                    continue
                folder=out/'records.local'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
                if (folder/'record.json').exists():
                    m=cached_record(folder,row,ch,verify_inputs=True)
                    with np.load(folder/'arrays.npz',allow_pickle=False) as f:
                        a={k:f[k].copy() for k in ['ct','base','target']}
                else:
                    ct,hct=read_array(row['input_path'])
                    cn=pipeline.normalize_ct(ct)
                    x=torch.from_numpy(cn)[None,None].to(pipeline.device)
                    with torch.inference_mode():
                        # Freeze and predict before target loading; never call the new model here.
                        pred=pipeline.legacy.predict_phase9g(x,steps=10)['phase9g_pred'][0,0].cpu().numpy().copy()
                    y,hy=read_array(row['output_path'])
                    ys,factor=pipeline.scale_target_for_legacy_evaluation(y)
                    a=dict(ct=cn,base=pred,target=ys)
                    m=dict(ct_sha256=hct,target_sha256=hy,dose_scale_factor=factor,producer='frozen_Phase9G_CT_only')
                    if any(v.shape!=(32,32,32) or not np.isfinite(v).all() for v in a.values()):
                        raise ValueError('Invalid cache array.')
                    save_arrays(folder/'arrays.npz',{k:np.asarray(v,np.float32) for k,v in a.items()})
                    m.update(sample_id=row['sample_id'],case_id=row['case_id'],record_id=row['record_id'],split=split,
                        contract_sha256=ch,arrays_sha256=sha(folder/'arrays.npz'))
                    write_json(folder/'record.json',m)
                train_arrays.append(a)
                index.append(dict(**row,split=split,record_meta_sha256=sha(folder/'record.json'),arrays_sha256=m['arrays_sha256']))
                audits.append(audit_row(a,m))
                if (j+1)%16==0 or j+1==len(rows):print(f'{split} cache {j+1}/{len(rows)} verified',flush=True)
        norm=fit_norm(train_arrays)
        write_json(out/'normalization.json',norm)
        write_json(out/'index.local.json',index)
        import pandas as pd
        pd.DataFrame(audits).to_csv(out/'residual_audit.local.csv',index=False)
        write_json(out/'READY.json',dict(status='READY_REAL_PHASE9G_RESIDUAL_CACHE',contract_sha256=ch,
            index_sha256=sha(out/'index.local.json'),norm_sha256=sha(out/'normalization.json'),
            n_train=len(tr),n_validation=len(va),scope=SCOPE,final_test_arrays_opened=0))
    print('CACHE READY:',out,flush=True)
    return out


class Cache:
    """Hash-verified host-memory copy. No further Drive reads inside training batches."""
    def __init__(self,path):
        self.path=Path(path);ready=read_json(self.path/'READY.json');self.contract=read_json(self.path/'cache_contract.local.json')
        if ready['contract_sha256']!=digest_object(self.contract):raise ValueError('Cache contract hash mismatch.')
        for f,h in [('index.local.json','index_sha256'),('normalization.json','norm_sha256')]:
            if sha(self.path/f)!=ready[h]:raise ValueError('Cache manifest/normalization mismatch.')
        self.norm=read_json(self.path/'normalization.json');self.rows=read_json(self.path/'index.local.json')
        self.identity=digest_object(ready);self.items=[]
        for j,row in enumerate(self.rows):
            is_old=row.get('storage')=='saved_legacy_validation'
            folder=(Path(self.contract['baseline_run']) if is_old else self.path)/'records.local'/row['sample_id']
            if sha(folder/'record.json')!=row['record_meta_sha256']:raise IOError('Cache metadata bytes changed.')
            m=read_json(folder/'record.json')
            raw=(folder/'arrays.npz').read_bytes()
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=row['arrays_sha256']:raise IOError('Cached array bytes changed.')
            with np.load(io.BytesIO(raw),allow_pickle=False) as a:
                if is_old:
                    factor=float(m['dose_scale_factor'])
                    ct=a['ct_normalized'].astype(np.float32,copy=True)
                    base=a['phase9g'].astype(np.float32,copy=True)*factor
                    target=a['target_stored'].astype(np.float32,copy=True)*factor
                    m['baseline_record_sha256']=row['record_meta_sha256']
                else:
                    ct=a['ct'].astype(np.float32,copy=True);base=a['base'].astype(np.float32,copy=True);target=a['target'].astype(np.float32,copy=True)
            if any(v.shape!=(32,32,32) or not np.isfinite(v).all() for v in [ct,base,target]):raise ValueError('Invalid cached real cube.')
            self.items.append(dict(row=row,meta=m,ct=ct,base=base,target=target))
            if (j+1)%128==0:print(f'RAM cache {j+1}/{len(self.rows)}',flush=True)
        self.train=[i for i,r in enumerate(self.rows) if r['split']=='train']
        self.val=[i for i,r in enumerate(self.rows) if r['split']=='validation']
        if not self.train or not self.val:raise ValueError('Missing train or validation data.')

    def select(self,indices,n_per_case,seed):
        rows=[self.rows[i] for i in indices]
        chosen={r['sample_id'] for r in subset_by_case(rows,n_per_case,seed)}
        return [i for i in indices if self.rows[i]['sample_id'] in chosen]

    def batch(self,indices,device):
        s=self.norm['residual_rms_model_scale'];bs=self.norm['base_input_scale']
        ct=np.stack([self.items[i]['ct'] for i in indices])[:,None]
        base=np.stack([self.items[i]['base'] for i in indices])[:,None]
        target=np.stack([self.items[i]['target'] for i in indices])[:,None]
        return {k:torch.from_numpy(np.asarray(v,np.float32)).to(device) for k,v in
                dict(condition=np.concatenate([ct,base/bs],axis=1),residual=(target-base)/s).items()}
