"""Real-data conditional training, target-free inference and explicit resume/fork."""
from pathlib import Path
import copy, time, random
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from . import common as io
from .model import ConditionalWFR
from .plans import PlanBank, plan_identity
from .geometry import sample_pairs, path_targets
from .metrics import score, residual_diagnostics
from .config import Config


def loss_step(model,items,banks,cfg):
    device=next(model.parameters()).device
    condition=torch.tensor(np.stack([a['condition'] for a in items]),device=device)
    features=model.encode_transport(condition)
    positions=[];velocities=[];growth=[];masses=[];times=[]
    # This legacy NumPy RNG is checkpointed. Preparation has a separate local generator.
    for item in items:
        k=int(np.random.randint(cfg.banks_per_record));bank=banks.get(item['row']['sample_id'],k)
        class Draws:
            @staticmethod
            def random(n):return np.random.random(n)
        x0,x1,m1=sample_pairs(bank,cfg.pairs_per_record,Draws())
        t=np.random.random(cfg.pairs_per_record)
        x,v,g,m=path_targets(x0,x1,m1,t,cfg.delta)
        positions.append(x);velocities.append(v);growth.append(g);masses.append(m);times.append(t)
    tensor=lambda x:torch.tensor(np.stack(x),dtype=torch.float32,device=device)
    x,v,g,m,t=map(tensor,(positions,velocities,growth,masses,times))
    pv,pg,_=model.vector_field(x,t,features)
    lv=(m*((pv-v)**2).sum(-1)).mean();lg=(m*(pg-g)**2).mean()
    logits=model.sign_logits(condition)[:,0]
    residual=tensor([a['residual'] for a in items]);rho=residual.abs();active=rho>cfg.sign_active_threshold
    labels=(residual>0).float();weight=rho*active
    sign_loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*weight).sum()/weight.sum().clamp_min(1e-12)
    total=lv+cfg.delta**2*lg+cfg.sign_loss_weight*sign_loss
    return total,{'flow_velocity':float(lv.detach()),'flow_growth':float(lg.detach()),'sign_bce':float(sign_loss.detach()),'total':float(total.detach())}


@torch.no_grad()
def evaluate_ids(model,cache,ids,cfg,device,*,allow_validation):
    rows=[]
    for sid in ids:
        item=cache.load(sid,allow_validation=allow_validation)
        result=model.predict(torch.tensor(item['condition'][None],device=device),cfg.ode_steps,cfg.query_chunk)
        scale=cache.norm['residual_rms_model_scale'];factor=item['factor']
        raw=(item['base'].astype(float)+scale*result['residual_normalized'])/factor
        # Existing model-space target is used in monitor selection, as in the prior pilot.
        y=item['target'].astype(float)/factor;base=item['base'].astype(float)/factor
        stats=score(np.maximum(raw,0),y,factor)
        stats.update(base_rmse=score(np.maximum(base,0),y,factor)['rmse_stored_units'],
                     raw_rmse=score(raw,y,factor)['rmse_stored_units'],
                     case_id=item['row']['case_id'],sample_id=sid)
        factors=residual_diagnostics(result['residual_normalized'],item['residual'],result['magnitude'],result['sign'])
        stats.update({k:factors[k] for k in ('residual_rmse_normalized','magnitude_mae_normalized','magnitude_total_relative_error','active_sign_accuracy')})
        stats['soft_sign_retained_magnitude_fraction']=result['diagnostics']['weighted_absolute_sign']
        rows.append(stats)
    frame=pd.DataFrame(rows);numeric=frame.drop(columns=['sample_id']).groupby('case_id').mean(numeric_only=True)
    out={k:(float(v) if np.isfinite(v) else None) for k,v in numeric.mean().items()}
    out.update(n_records=len(rows),n_cases=len(numeric))
    return out


def _save(run,model,optim,state,contract_hash,*,is_best=False):
    payload={'model':model.state_dict(),'optimizer':optim.state_dict(),'state':copy.deepcopy(state),
             'rng':io.rng_state(),'contract_sha256':contract_hash,'step':state['step']}
    pointer=io.save_checkpoint(run,payload)
    io.write(run/'last.json',pointer,replace=True)
    if is_best:io.write(run/'best.json',pointer,replace=True)
    return pointer


def _identity(cache,plans,cfg):
    return {'config':cfg.to_dict(),'cache_identity':cache.identity,'source_code':io.source_identity(),
            'plan_contract_sha256':io.digest(plan_identity(cache,cfg)),
            'selection':cache.selection(cfg.stage,cfg.monitor_records_per_case,cfg.monitor_seed),
            'scope':'REAL_CONDITIONAL_WFR_MAGNITUDE_GRID_SIGN_NOT_WATER_NOT_BLIND_FINAL_TEST'}


def train(cache,plans,run,cfg,device='cpu',max_updates=128):
    cfg.validate();run=Path(run).resolve();io.separate(run,cache.path,cache.old,plans)
    io.seed_all(cfg.seed);identity=_identity(cache,plans,cfg)
    if str(device).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA is not available.')
    run.mkdir(parents=True,exist_ok=True)
    with io.single_writer(run):
        cp=run/'contract.json';env=io.environment(device)
        if cp.exists():
            contract=io.read(cp)
            if contract['identity']!=identity:raise ValueError('Scientific identity changed. Use an explicit new experiment, not an overwritten contract.')
            if (run/'COMPLETE.json').exists():
                io.verify_finish(run,'conditional_training_completed');print('Completed training artifacts verified; no retraining.',flush=True);return
            if contract['environment']!=env:raise RuntimeError('Resume environment changed. Use the explicit fork command; do not edit or delete the old contract.')
        else:
            if any(run.iterdir()):raise ValueError('Nonempty run directory without contract.')
            contract={'identity':identity,'environment':env,'parent':None};io.write(cp,contract)
        ch=io.digest(contract);bank=PlanBank(plans,cache,cfg)
        model=ConditionalWFR(cfg.channels,cfg.point_hidden).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
        selection=identity['selection'];tr=selection['train_ids'];monitor=selection['monitor_ids'];allow_val=cfg.stage=='pilot'
        # Read only this stage's train arrays and explicitly allowed monitor arrays into the bounded cache.
        for j,sid in enumerate(tr,1):
            cache.load(sid)
            if j%32==0:print(f'RAM TRAIN {j}/{len(tr)}',flush=True)
        if (run/'last.json').exists():
            saved=io.load_checkpoint(run,io.read(run/'last.json'))
            if saved['contract_sha256']!=ch:raise ValueError('Checkpoint contract link mismatch.')
            model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key,value in state.items():
                    if torch.is_tensor(value):state[key]=value.to(device)
            state=saved['state'];io.restore_rng(saved['rng'])
            print(f'RESUME next update={state["step"]+1}, total budget={cfg.updates}',flush=True)
        else:
            initial=evaluate_ids(model,cache,monitor,cfg,device,allow_validation=allow_val)
            state={'step':0,'history':[],'best_metric':None,'initial_monitor':initial,'loss_records':[]}
            io.write(run/'initial_monitor.json',initial);_save(run,model,optimizer,state,ch)
            print(f'START {cfg.stage}: train={len(tr)}, monitor={len(monitor)}, zero-correction RMSE={initial["rmse_stored_units"]:.9g}',flush=True)
        count=0;begin=time.monotonic()
        try:
            while state['step']<cfg.updates:
                model.train();chosen=random.choices(tr,k=cfg.batch_size);items=[cache.load(sid) for sid in chosen]
                optimizer.zero_grad(set_to_none=True);loss,terms=loss_step(model,items,bank,cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss. No optimizer update was made.')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip,error_if_nonfinite=True)
                optimizer.step();state['step']+=1;count+=1
                terms.update(step=state['step'],gradient_norm=float(norm));state['loss_records'].append(terms)
                if state['step']%cfg.monitor_every==0 or state['step']==cfg.updates:
                    val=evaluate_ids(model,cache,monitor,cfg,device,allow_validation=allow_val)
                    hist={'step':state['step'],**val};state['history'].append(hist)
                    improved=state['best_metric'] is None or val['rmse_stored_units']<state['best_metric']
                    if improved:state['best_metric']=val['rmse_stored_units']
                    _save(run,model,optimizer,state,ch,is_best=improved)
                    pd.DataFrame(state['history']).to_csv(run/'monitor_history.csv',index=False)
                    pd.DataFrame(state['loss_records']).to_csv(run/'loss_history.csv',index=False)
                    print(f'UPDATE {state["step"]}/{cfg.updates} loss={terms["total"]:.6g} monitor_RMSE={val["rmse_stored_units"]:.9g} x_mean_pct={val["x_mean_pct"]}',flush=True)
                elif state['step']%cfg.save_every==0:_save(run,model,optimizer,state,ch)
                if max_updates is not None and count>=max_updates and state['step']<cfg.updates:
                    _save(run,model,optimizer,state,ch)
                    io.write(run/'status.json',{'status':'paused_at_saved_update','step':state['step'],'contract_sha256':ch},replace=True)
                    print('PAUSED safely. Re-run this exact command to continue.',flush=True);return
            selected=io.load_checkpoint(run,io.read(run/'best.json'));model.load_state_dict(selected['model'],strict=True)
            final=evaluate_ids(model,cache,monitor,cfg,device,allow_validation=allow_val)
            summary={'scope':identity['scope'],'stage':cfg.stage,'updates':state['step'],'selected_step':selected['step'],
                     'initial_monitor':state['initial_monitor'],'selected_monitor':final,
                     'parameters':sum(p.numel() for p in model.parameters()),'batch_size':cfg.batch_size,
                     'training_records':len(tr),'monitor_records':len(monitor),'training_seed':cfg.seed,
                     'optimizer_seconds_this_call_including_monitor':time.monotonic()-begin,
                     'learned_beyond_zero_on_monitor':final['rmse_stored_units']<final['base_rmse'],
                     'relative_monitor_rmse_improvement':1-final['rmse_stored_units']/final['base_rmse'] if final['base_rmse']>0 else None,
                     'learnability_note':'A positive improvement boolean alone is not an overfit/convergence certificate.',
                     'primary_selection':cfg.checkpoint_selection,'fairness':'same train192/monitor40 and nominal update budget only; architectures, operations and old training history differ',
                     'scientific_status':'PENDING_REVIEW_NO_IMPROVEMENT_GUARANTEE',**cache.access_summary()}
            io.write(run/'training_summary.json',summary);cache.check_metadata_unchanged()
            io.finish(run,'conditional_training_completed',contract_sha256=ch,updates=state['step'])
            print('TRAINING COMPLETE:',run,flush=True);print(summary,flush=True)
        except BaseException as exc:
            io.write(run/'failure.json',{'type':type(exc).__name__,'message':str(exc),'last_pointer_exists':(run/'last.json').exists(),
                                        'note':'Resume only a verified checkpoint; unsaved updates may be repeated.'},replace=True)
            raise


def load_model(run,cache,device='cpu'):
    run=Path(run);c=io.read(run/'contract.json');io.verify_finish(run,'conditional_training_completed')
    identity=c['identity'];cfg=Config(**identity['config']).validate()
    if identity['cache_identity']!=cache.identity or identity['source_code']!=io.source_identity():raise ValueError('Model cache/source differs.')
    pointer=io.read(run/'best.json');ck=io.load_checkpoint(run,pointer)
    if ck['contract_sha256']!=io.digest(c):raise ValueError('Model checkpoint identity mismatch.')
    model=ConditionalWFR(cfg.channels,cfg.point_hidden).to(device);model.load_state_dict(ck['model'],strict=True);model.eval()
    return model,cfg,c,pointer


def fork_run(cache,plans,parent,out,device,updates=None):
    """Explicit non-bitwise environment/budget continuation with retained ancestry."""
    parent=Path(parent).resolve();out=Path(out).resolve();io.separate(out,parent,cache.path,cache.old,plans)
    if out.exists() and any(out.iterdir()):raise ValueError('Fork destination must be new/empty.')
    old=io.read(parent/'contract.json');cfg=Config(**old['identity']['config']).validate()
    old_pointer=io.read(parent/'last.json');ck=io.load_checkpoint(parent,old_pointer)
    if ck['contract_sha256']!=io.digest(old):raise ValueError('Parent checkpoint/contract differs.')
    if updates is not None:
        from dataclasses import replace
        if updates<cfg.updates:raise ValueError('Do not shorten a budget in a continuation.')
        cfg=replace(cfg,updates=updates).validate()
    if ck['step']>=cfg.updates:raise ValueError('Already at budget; an extension needs a larger declared budget.')
    identity=_identity(cache,plans,cfg)
    for key in ('cache_identity','source_code','plan_contract_sha256','selection'):
        if old['identity'][key]!=identity[key]:raise ValueError('Fork only permits environment/budget changes, not different science/data.')
    io.seed_all(cfg.seed);env=io.environment(device)
    c={'identity':identity,'environment':env,'parent':{'run':str(parent),'contract_sha256':io.digest(old),'checkpoint':old_pointer,
       'kind':'explicit_non_bitwise_continuation','total_updates_include_parent':True}}
    out.mkdir(parents=True,exist_ok=True);io.write(out/'contract.json',c)
    # Preserve CPU/NumPy random states; explicitly reset CUDA RNG for changed hardware.
    if old['environment']!=env:
        current=io.rng_state();ck['rng']['cuda']=current['cuda']
    ck['contract_sha256']=io.digest(c);ck['state']['best_metric']=None
    pointer=io.save_checkpoint(out,ck);io.write(out/'last.json',pointer)
    print('Fork created, not trained:',out,flush=True)
