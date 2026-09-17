"""Train-only optimization with verified resumable checkpoints and fixed evaluation."""
from pathlib import Path
from collections import defaultdict
import copy, random, shutil, math, time
import numpy as np
import pandas as pd
import torch
from p10recover.io import sha, read_json, write_json, digest_object
from .common import runtime_settings, device_identity, rng_state, restore_rng, code_hashes, single_writer, checkpoint_write, checkpoint_read, status, log, SCOPE
from .models import build, loss


def recipe(stage='pilot'):
    if stage not in ['small','pilot']:raise ValueError('Choose small or pilot.')
    return dict(stage=stage,seed=17,lr=3e-4,weight_decay=1e-5,batch_size=2,
        epochs=8 if stage=='small' else 12,updates_per_epoch=16 if stage=='small' else 32,
        dense_width=16,hjd_width=8,features=8,hidden=64,fm_particles=1024,
        ode_steps=8,particles=4096,source_seed=31415,
        lambda_mass=5.0,lambda_component=1.0,lambda_reconstruction=1.0,
        lambda_overlap=0.1,lambda_boundary=0.1,clip_grad=5.0,save_every_updates=16,
        evaluation_output='raw signed correction + frozen Phase9G; clamp absolute dose at zero for primary legacy comparison',
        model_selection='equal-case clipped absolute-dose RMSE on fixed train subset for small; fixed validation40 for pilot',
        deterministic_claim='RNG saved; CUDA grid_sample/scatter may remain nondeterministic')


def validate_recipe(r):
    needed=set(recipe())
    if set(r)!=needed:raise ValueError('Recipe fields changed; explicit versioned implementation required.')
    for k in ['seed','batch_size','epochs','updates_per_epoch','dense_width','hjd_width','features','hidden','fm_particles','ode_steps','particles','source_seed','save_every_updates']:
        if type(r[k]) is not int or (k not in ['seed','source_seed'] and r[k]<1):raise ValueError('Invalid integer '+k)
    for k in ['lr','clip_grad']:
        if not math.isfinite(r[k]) or r[k]<=0:raise ValueError('Invalid '+k)
    for k in ['weight_decay','lambda_mass','lambda_component','lambda_reconstruction','lambda_overlap','lambda_boundary']:
        if not math.isfinite(r[k]) or r[k]<0:raise ValueError('Invalid '+k)


@torch.no_grad()
def evaluate_subset(model,cache,indices,r,device):
    """Fixed target-free model input; targets are used only after predicting."""
    model.eval();rows=[];s=cache.norm['residual_rms_model_scale'];bs=cache.norm['base_input_scale']
    for i in indices:
        a=cache.items[i]
        cond=np.stack([a['ct'],a['base']/bs])[None]
        z=model.predict(torch.from_numpy(cond.astype(np.float32)).to(device),r['ode_steps'],r['particles'],r['source_seed'])
        pr=z['residual'][0,0].cpu().numpy()*s
        raw=a['base']+pr;pred=np.maximum(raw,0);target=a['target'];factor=float(a['meta']['dose_scale_factor'])
        err=(pred.astype(float)-target)/factor;be=(a['base'].astype(float)-target)/factor
        truth=target-a['base'];active=np.abs(truth)>0.01*s
        rows.append(dict(case_id=a['row']['case_id'],rmse=float(np.sqrt(np.mean(err*err))),
           base_rmse=float(np.sqrt(np.mean(be*be))),raw_rmse=float(np.sqrt(np.mean(((raw-target)/factor)**2))),
           sign_accuracy=float((np.sign(pr[active])==np.sign(truth[active])).mean()) if active.any() else None,
           outside=float(z['outside'].mean().cpu())))
    frame=pd.DataFrame(rows);case=frame.groupby('case_id').mean(numeric_only=True)
    out={k:float(v) if np.isfinite(v) else None for k,v in case.mean().to_dict().items()}
    out.update(n_records=len(rows),n_cases=len(case))
    return out


def _snapshot(run,model,optimizer,state,contract_hash,is_best=False):
    state=copy.deepcopy(state)
    state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),contract_sha256=contract_hash)
    pointer=checkpoint_write(run,state,f'step{state["step"]:07d}')
    write_json(Path(run)/'last.json',pointer,replace=True)
    if is_best:write_json(Path(run)/'best.json',pointer,replace=True)
    return pointer


def train(cache,method,out,r,device='cpu',max_updates_this_call=None):
    validate_recipe(r);out=Path(out).resolve()
    if out.is_relative_to(cache.path):raise ValueError('Training run must be separate from immutable cache.')
    # Selection is frozen before optimization, not conditioned on model errors.
    tr=cache.select(cache.train,2,731) if r['stage']=='small' else list(cache.train)
    ev=list(tr) if r['stage']=='small' else cache.select(cache.val,20,732)
    if any(cache.rows[i]['split']!='train' for i in tr):raise ValueError('Non-training record in optimizer selection.')
    selection=dict(train_ids=[cache.rows[i]['sample_id'] for i in tr],monitor_ids=[cache.rows[i]['sample_id'] for i in ev],
                   monitor_partition='train_only_learnability' if r['stage']=='small' else 'validation_selection_subset40')
    runtime_settings(r['seed']);d=torch.device(device)
    if d.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable.')
    identity=dict(recipe=r,method=method,cache_identity=cache.identity,selection=selection,code=code_hashes())
    contract=dict(**identity,environment=device_identity(d),scope=SCOPE,
                  fairness='same new-method record/update budget; old Phase10D training budget and upstream exposure differ')
    out.mkdir(parents=True,exist_ok=True)
    with single_writer(out):
        cp=out/'contract.local.json'
        if cp.exists():
            old=read_json(cp)
            if any(old[k]!=v for k,v in identity.items()):raise ValueError('Training recipe/data/source changed. New run_id required.')
            oldstatus=read_json(out/'run_status.json') if (out/'run_status.json').exists() else {}
            if oldstatus.get('status')=='completed_training':
                ck=checkpoint_read(out,read_json(out/'best.json'))
                if ck['contract_sha256']!=digest_object(old):raise ValueError('Completed best checkpoint identity mismatch.')
                print('Already completed; trained checkpoint verified:',out)
                return out
            if old!=contract:raise ValueError('Resume environment differs. Do not claim exact resume; use the original environment or a separately labelled new run.')
        else:
            if any(out.iterdir()):raise ValueError('Nonempty run directory without contract; no overwrite.')
            write_json(cp,contract);src=out/'source_snapshot';src.mkdir()
            root=Path(__file__).resolve().parent.parent
            for name in contract['code']:
                dst=src/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dst)
        ch=digest_object(contract)
        model=build(method,r,cache.norm).to(d);optimizer=torch.optim.AdamW(model.parameters(),lr=r['lr'],weight_decay=r['weight_decay'])
        if (out/'last.json').exists():
            st=checkpoint_read(out,read_json(out/'last.json'))
            if st['contract_sha256']!=ch:raise ValueError('Resume checkpoint contract mismatch.')
            model.load_state_dict(st.pop('model'),strict=True);optimizer.load_state_dict(st.pop('optimizer'))
            restore_rng(st.pop('rng'));st.pop('contract_sha256')
            log(out,f'RESUME: next epoch={st["epoch"]}, update={st["update"]}, global_step={st["step"]}')
        else:
            initial=evaluate_subset(model,cache,ev,r,d)
            st=dict(epoch=1,update=0,step=0,sums={},history=[],best_metric=float('inf'),initial_monitor=initial)
            write_json(out/'initial_monitor.json',initial)
            _snapshot(out,model,optimizer,st,ch)
            log(out,f'START {method}: train={len(tr)} monitor={len(ev)}; baseline RMSE={initial["base_rmse"]:.8g}')
        status(out,'training',contract_sha256=ch)
        call_updates=0;started=time.monotonic()
        try:
            while st['epoch']<=r['epochs']:
                while st['update']<r['updates_per_epoch']:
                    model.train()
                    ids=random.choices(tr,k=r['batch_size'])
                    b=cache.batch(ids,d);optimizer.zero_grad(set_to_none=True)
                    total,terms=loss(model,b,method,r)
                    if not torch.isfinite(total):raise FloatingPointError('Nonfinite training loss.')
                    total.backward()
                    gn=torch.nn.utils.clip_grad_norm_(model.parameters(),r['clip_grad'],error_if_nonfinite=True)
                    optimizer.step();st['step']+=1;st['update']+=1;call_updates+=1
                    for k,v in terms.items():st['sums'][k]=st['sums'].get(k,0.)+float(v.detach().cpu())
                    if st['step']%r['save_every_updates']==0:
                        _snapshot(out,model,optimizer,st,ch)
                        log(out,f'checkpoint step={st["step"]} epoch={st["epoch"]} update={st["update"]} loss={float(total.detach()):.6g}')
                    if max_updates_this_call is not None and call_updates>=max_updates_this_call:
                        _snapshot(out,model,optimizer,st,ch);status(out,'paused_at_saved_update',step=st['step'],contract_sha256=ch)
                        return out
                val=evaluate_subset(model,cache,ev,r,d)
                row=dict(epoch=st['epoch'],step=st['step'],**{'train_'+k:v/r['updates_per_epoch'] for k,v in st['sums'].items()},
                         monitor_rmse=val['rmse'],monitor_raw_rmse=val['raw_rmse'],monitor_base_rmse=val['base_rmse'],
                         monitor_sign_accuracy=val['sign_accuracy'],monitor_outside=val['outside'])
                st['history'].append(row);improved=val['rmse']<st['best_metric']
                if improved:st['best_metric']=val['rmse']
                log(out,f'epoch={st["epoch"]:02d} step={st["step"]} loss={row["train_total"]:.6g} monitor_RMSE={val["rmse"]:.8g} Phase9G={val["base_rmse"]:.8g} outside={val["outside"]:.4f}')
                st['epoch']+=1;st['update']=0;st['sums']={}
                _snapshot(out,model,optimizer,st,ch,is_best=improved)
                pd.DataFrame(st['history']).to_csv(out/'history.csv',index=False)
                write_json(out/'progress.json',dict(next_epoch=st['epoch'],step=st['step'],best_metric=st['best_metric']),replace=True)
            # No resume from small to pilot: pilot uses fresh weights and a separate recipe.
            best=checkpoint_read(out,read_json(out/'best.json'));model.load_state_dict(best['model'],strict=True)
            ending=evaluate_subset(model,cache,ev,r,d)
            write_json(out/'training_summary.json',dict(initial_monitor=st['initial_monitor'],best_monitor=ending,
                learned_beyond_zero_correction=ending['rmse']<ending['base_rmse'],
                claim='training subset learnability only' if r['stage']=='small' else 'small validation-selected pilot; not final test',
                parameter_count=sum(p.numel() for p in model.parameters()),global_updates=st['step'],
                current_call_seconds=time.monotonic()-started,monitor_partition=selection['monitor_partition']))
            status(out,'completed_training',contract_sha256=ch,step=st['step'],best_checkpoint=read_json(out/'best.json'))
            plot_history(out)
        except BaseException as exc:
            status(out,'interrupted_or_failed',contract_sha256=ch,error=type(exc).__name__,message=str(exc),
                resume_note='Resume from last.json only; unsaved updates may be replayed. Do not bypass contract checks.')
            raise
    return out


def load_trained(run,cache,device='cpu'):
    run=Path(run);c=read_json(run/'contract.local.json');s=read_json(run/'run_status.json')
    if s.get('status')!='completed_training':raise ValueError('Train run is not complete.')
    if c['cache_identity']!=cache.identity or c['code']!=code_hashes():raise ValueError('Trained model cache/source mismatch.')
    pointer=read_json(run/'best.json');ck=checkpoint_read(run,pointer)
    if ck['contract_sha256']!=digest_object(c):raise ValueError('Best checkpoint identity mismatch.')
    m=build(c['method'],c['recipe'],cache.norm).to(device);m.load_state_dict(ck['model'],strict=True);m.eval()
    return m,c,pointer


def plot_history(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);df=pd.read_csv(run/'history.csv')
    for col,ylabel in [('train_total','Training objective (not comparable across methods)'),('monitor_rmse','Clipped absolute-dose RMSE in stored units')]:
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(df.epoch,df[col],marker='o')
        if col=='monitor_rmse':ax.plot(df.epoch,df.monitor_base_rmse,label='Frozen Phase9G');ax.legend()
        ax.set(xlabel='Epoch',ylabel=ylabel,title=run.name);fig.tight_layout();fig.savefig(run/(col+'.png'),dpi=140);plt.close(fig)
