"""Training-only bank preparation. Each bank is saved independently for recovery."""
from pathlib import Path
import hashlib, io, os, uuid
import numpy as np
from .common import read, write, digest, sha, safe, source_identity, separate, single_writer, finish, verify_finish
from .geometry import make_bank


def plan_identity(cache,cfg):
    sel=cache.selection(cfg.stage,cfg.monitor_records_per_case,cfg.monitor_seed)
    return {'scope':'APPROXIMATE_WEIGHTED_TRAIN_ONLY_OET_BANKS','cache_identity':cache.identity,'train_ids':sel['train_ids'],
            'config':{k:v for k,v in cfg.to_dict().items() if k in ('delta','source_points_per_bank','target_draws_per_bank','banks_per_record','coupling_iterations','coupling_relative_tolerance','coupling_seed')},
            'source_code':source_identity(), 'common_mass_divisor':32768,
            'inference_source_mass_per_voxel':1.,'normalization':cache.norm}


def prepare(cache,out,cfg):
    out=Path(out).resolve();separate(out,cache.path,cache.old);ident=plan_identity(cache,cfg)
    out.mkdir(parents=True,exist_ok=True)
    with single_writer(out):
        write(out/'contract.json',ident)
        if (out/'COMPLETE.json').exists():
            verify_finish(out,'training_couplings_prepared');print('Existing coupling banks verified.',flush=True);return
        banks=out/'banks.local';banks.mkdir(exist_ok=True);all_stats=[]
        for number,sid in enumerate(ident['train_ids'],1):
            item=cache.load(sid)
            for k in range(cfg.banks_per_record):
                name=f'{sid}_{k:02d}';dest=banks/(name+'.npz');meta=banks/(name+'.json')
                seed=int(hashlib.sha256(f'{cfg.coupling_seed}:{sid}:{k}'.encode()).hexdigest()[:16],16)
                if meta.exists():
                    m=read(meta)
                    if m['contract_sha256']!=digest(ident) or sha(dest)!=m['arrays_sha256']: raise ValueError('Existing coupling bank differs.')
                else:
                    data,stats=make_bank(item['residual'],cfg,np.random.default_rng(seed))
                    tmp=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp')
                    with tmp.open('xb') as f:np.savez_compressed(f,**data)
                    os.replace(tmp,dest)
                    m={'sample_id':sid,'bank':k,'seed':seed,'contract_sha256':digest(ident),'arrays_sha256':sha(dest),'stats':stats}
                    write(meta,m)
                all_stats.append(m)
            print(f'TRAIN COUPLINGS {number}/{len(ident["train_ids"])} saved/verified',flush=True)
        cache.check_metadata_unchanged()
        write(out/'bank_index.local.json',all_stats)
        summary={'n_training_records':len(ident['train_ids']),'n_banks':len(all_stats),
          'banks_reaching_iterate_tolerance':sum(m['stats']['converged_by_iterate_test'] for m in all_stats),
          'max_final_relative_change':max(m['stats']['relative_iterate_change'] for m in all_stats),
          'approximation':'finite support, finite iterations, fixed small bank set; not exact full-grid OT',**cache.access_summary()}
        write(out/'summary.json',summary);finish(out,'training_couplings_prepared',contract_sha256=digest(ident))
        print(summary,flush=True)

class PlanBank:
    def __init__(self,path,cache,cfg):
        self.path=Path(path);c=read(self.path/'contract.json')
        if c!=plan_identity(cache,cfg):raise ValueError('Training coupling data/config/source mismatch.')
        verify_finish(self.path,'training_couplings_prepared');self.memo={};self.cfg=cfg
        self.metas={(m['sample_id'],m['bank']):m for m in read(self.path/'bank_index.local.json')}
    def get(self,sid,k):
        key=(sid,k)
        if key not in self.memo:
            p=self.path/'banks.local'/f'{sid}_{k:02d}.npz';m=self.metas[key]
            if sha(p)!=m['arrays_sha256']:raise ValueError('Coupling file changed.')
            with np.load(p,allow_pickle=False) as z:self.memo[key]={n:z[n].copy() for n in z.files}
        return self.memo[key]
