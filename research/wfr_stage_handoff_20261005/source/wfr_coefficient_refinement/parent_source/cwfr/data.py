"""Explicit train/validation access to the existing immutable real cache."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, OrderedDict
import io, re
import numpy as np
from .common import read, digest, sha, safe

class RealCache:
    def __init__(self, path):
        self.path=Path(path).resolve(strict=True)
        names=('READY.json','cache_contract.local.json','index.local.json','normalization.json')
        self.snapshot={str(self.path/n):sha(self.path/n) for n in names}
        self.ready=read(self.path/'READY.json'); self.contract=read(self.path/'cache_contract.local.json')
        if self.contract.get('schema')!='real_phase9g_cache_v1' or self.ready.get('status')!='READY_REAL_PHASE9G_RESIDUAL_CACHE':
            raise ValueError('Unsupported or incomplete cache. No raw-data fallback exists.')
        if self.ready.get('n_train')!=192 or self.ready.get('n_validation')!=600 or self.ready.get('final_test_arrays_opened')!=0:
            raise ValueError('Unexpected saved cache count/access contract.')
        self.contract_hash=digest(self.contract)
        if self.ready['contract_sha256']!=self.contract_hash: raise ValueError('Cache contract differs.')
        for fn,k in [('index.local.json','index_sha256'),('normalization.json','norm_sha256')]:
            if sha(self.path/fn)!=self.ready[k]: raise ValueError('Cache identity differs: '+fn)
        self.identity=digest(self.ready); self.rows=read(self.path/'index.local.json'); self.norm=read(self.path/'normalization.json')
        if len(self.rows)!=792 or len({r['sample_id'] for r in self.rows})!=792: raise ValueError('Expected unique train192 plus validation600 metadata.')
        if any(r['split'] not in ('train','validation') for r in self.rows): raise ValueError('Unknown split label. Test data are not supported.')
        self.train=[r for r in self.rows if r['split']=='train']; self.val=[r for r in self.rows if r['split']=='validation']
        tc=Counter(r['case_id'] for r in self.train); vc=Counter(r['case_id'] for r in self.val)
        if len(tc)!=6 or set(tc.values())!={32} or len(vc)!=2 or set(vc.values())!={300} or set(tc)&set(vc):
            raise ValueError('Expected disjoint six-by-32 training and two-by-300 validation records.')
        for key in ('input_path','output_path'):
            if {r[key] for r in self.train}&{r[key] for r in self.val}: raise ValueError('Train/validation path overlap.')
        for split,rows in [('train',self.train),('validation',self.val)]:
            declared=self.contract[split]
            fields=('sample_id','case_id','record_id','input_path','output_path')
            if [{k:r[k] for k in fields} for r in rows]!=[{k:r[k] for k in fields} for r in declared]:
                raise ValueError('Cache row order or identity differs from its contract.')
        if self.norm.get('fitted_on')!='selected_train6_only' or self.norm.get('water_reference') is not False or self.norm.get('synthetic_reference') is not False:
            raise ValueError('Unknown scale or reference semantics.')
        for k in ('residual_rms_model_scale','base_input_scale'):
            if not np.isfinite(self.norm[k]) or self.norm[k]<=0: raise ValueError('Invalid training scale: '+k)
        self.by_id={r['sample_id']:r for r in self.rows}
        self.old=Path(self.contract['baseline_run']).resolve()
        self.memo=OrderedDict(); self.opened={'train':set(),'validation':set()}

    def selection(self,stage,monitor_per_case=20,monitor_seed=732):
        if stage not in ('one','six','pilot'): raise ValueError('Unknown stage.')
        if stage=='pilot': tr=list(self.train)
        else:
            tr=[sorted([r for r in self.train if r['case_id']==c],key=lambda r:r['sample_id'])[0]
                for c in sorted({r['case_id'] for r in self.train})]
            if stage=='one': tr=tr[:1]
        if stage=='pilot':
            chosen=set()
            for case in sorted({r['case_id'] for r in self.val}):
                group=[r for r in self.val if r['case_id']==case]
                rank=lambda r:digest({'seed':monitor_seed,'row':tuple(r[k] for k in ('case_id','input_path','output_path'))})
                chosen.update(r['sample_id'] for r in sorted(group,key=rank)[:monitor_per_case])
            ev=[r for r in self.val if r['sample_id'] in chosen]
        else: ev=list(tr)
        return {'train_ids':[r['sample_id'] for r in tr],'monitor_ids':[r['sample_id'] for r in ev],
                'monitor_partition':'validation_selection_subset40' if stage=='pilot' else 'training_only_learnability',
                'validation_ids':[r['sample_id'] for r in self.val]}

    def load(self,sid,*,allow_validation=False):
        if sid not in self.by_id or re.fullmatch(r'[0-9a-f]{20}',sid) is None: raise ValueError('Unknown sample ID.')
        row=self.by_id[sid]; split=row['split']
        if split!='train' and not allow_validation: raise PermissionError('Validation arrays were not authorized for this operation.')
        if sid in self.memo:
            self.memo.move_to_end(sid); return self.memo[sid]
        root=self.old if split=='validation' else self.path
        folder=safe(root,'records.local/'+sid)
        if sha(folder/'record.json')!=row['record_meta_sha256']: raise ValueError('Cached metadata changed: '+sid)
        meta=read(folder/'record.json')
        expected=self.contract['baseline_contract_sha256'] if split=='validation' else self.contract_hash
        if meta['sample_id']!=sid or meta['case_id']!=row['case_id'] or meta['contract_sha256']!=expected:
            raise ValueError('Cached identity mismatch: '+sid)
        raw=(folder/'arrays.npz').read_bytes()
        import hashlib
        h=hashlib.sha256(raw).hexdigest()
        if h!=row['arrays_sha256'] or h!=meta['arrays_sha256']: raise ValueError('Cached arrays changed: '+sid)
        factor=float(meta['dose_scale_factor'])
        if not np.isfinite(factor) or factor<=0: raise ValueError('Invalid scale factor.')
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            if split=='train':
                if meta.get('producer')!='frozen_Phase9G_CT_only': raise ValueError('Unknown frozen producer.')
                ct=z['ct'].astype(np.float32); base=z['base'].astype(np.float32); target=z['target'].astype(np.float32)
                original={}
            else:
                ct=z['ct_normalized'].astype(np.float32)
                original={k:z[k].astype(np.float32) for k in ('target_stored','phase9g','phase10d_strict')}
                base=original['phase9g']*factor; target=original['target_stored']*factor
        if any(v.shape!=(32,32,32) or not np.isfinite(v).all() for v in (ct,base,target)): raise ValueError('Invalid real cube.')
        # Match the previous learning pipeline's float32 residual arithmetic.
        residual=(target-base)/self.norm['residual_rms_model_scale']
        item={'ct':ct,'base':base,'target':target,'residual':residual.astype(np.float32),
              'condition':np.stack([ct,base/self.norm['base_input_scale']]).astype(np.float32),
              'factor':factor,'row':row,'meta':meta,'original':original}
        self.opened[split].add(sid); self.memo[sid]=item
        if len(self.memo)>240: self.memo.popitem(last=False)
        return item

    def check_metadata_unchanged(self):
        for p,h in self.snapshot.items():
            if sha(p)!=h: raise ValueError('Source cache metadata changed: '+p)

    def access_summary(self):
        return {'training_arrays_opened':len(self.opened['train']), 'validation_arrays_opened':len(self.opened['validation']),
                'test_arrays_opened':0, 'raw_arrays_opened':0, 'new_baseline_predictions':0}
