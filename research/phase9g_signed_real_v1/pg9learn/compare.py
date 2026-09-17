"""New heads vs the saved real legacy cohort; no re-evaluation of old networks."""
from pathlib import Path
import time, math, html
import numpy as np
import pandas as pd
import torch
from p10recover.io import read_json, write_json, sha, digest_object, save_arrays
from p10recover.metrics import score, LEGACY
from p10recover.evaluate import aggregate
from .common import code_hashes, device_identity, single_writer, SCOPE
from .train import load_trained


def residual_metrics(pred_r,truth_r,parts,s):
    """Scaled residuals and raw HJD components; do not conflate post-clamp parts."""
    active=np.abs(truth_r)>0.01*s
    p,n=parts[0],parts[1];total=float(np.sum(p)+np.sum(n))
    canonical_pos=np.maximum(pred_r,0);canonical_neg=np.maximum(-pred_r,0)
    return dict(active_fraction=float(active.mean()),
        active_sign_accuracy=float((np.sign(pred_r[active])==np.sign(truth_r[active])).mean()) if active.any() else None,
        positive_mass_model=float(p.sum()),negative_mass_model=float(n.sum()),
        true_positive_mass_model=float(np.maximum(truth_r,0).sum()),true_negative_mass_model=float(np.maximum(-truth_r,0).sum()),
        canonical_positive_mass_model=float(canonical_pos.sum()),canonical_negative_mass_model=float(canonical_neg.sum()),
        cancellation_ratio=2*float(np.minimum(p,n).sum())/total if total>0 else 0.,
        overlap_mean_residual_scaled=float(np.mean((p/s)*(n/s))),
        residual_rmse_model=float(np.sqrt(np.mean((pred_r.astype(float)-truth_r)**2))),
        sign_threshold_model=0.01*s)


def evaluate_new(cache,run,out,device='cpu'):
    out=Path(out).resolve();run=Path(run).resolve()
    if out.is_relative_to(run) or out.is_relative_to(cache.path):raise ValueError('Use a separate new evaluation directory.')
    model,tc,pointer=load_trained(run,cache,device);r=tc['recipe']
    if r['stage']!='pilot':raise ValueError('Small-train learnability runs are not entered in val600 comparisons.')
    method='new_'+tc['method']
    contract=dict(schema='real_phase9g_new_evaluation_v1',scope=SCOPE,train_contract_sha256=digest_object(tc),
        checkpoint=pointer,cache_identity=cache.identity,method=method,
        val_ids=[cache.rows[i]['sample_id'] for i in cache.val],
        environment=device_identity(device),code=code_hashes(),effective_solver='Euler left-endpoint time for correction ODE',
        ode_steps=r['ode_steps'],particles_per_branch=r['particles'] if tc['method']=='hjd_rf' else 0,
        velocity_head_calls=r['ode_steps']*(2 if tc['method']=='hjd_rf' else 1),
        shared_frozen_pipeline_cost='Phase9G inference is shared and excluded from correction-only timings',
        primary_clamp='absolute dose clamped at zero; raw metrics separately preserved',
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        pilot_training_records=len(tc['selection']['train_ids']),
        monitor_records=len(tc['selection']['monitor_ids']),
        optimizer_updates=r['epochs']*r['updates_per_epoch'],batch_size=r['batch_size'],
        shared_upstream='Frozen Phase9G; upstream cost and historical exposure not removed')
    ch=digest_object(contract);out.mkdir(parents=True,exist_ok=True)
    with single_writer(out):
        cp=out/'contract.local.json'
        if cp.exists():
            if read_json(cp)!=contract:raise ValueError('Evaluation identity/environment differs; use a new evaluation directory.')
        else:
            if any(out.iterdir()):raise ValueError('Nonempty evaluation directory without contract.')
            write_json(cp,contract)
        rows=[];rawrows=[];diag=[];s=cache.norm['residual_rms_model_scale'];bs=cache.norm['base_input_scale']
        write_json(out/'run_status.json',dict(status='evaluating',scope=SCOPE),replace=True)
        try:
            for j,i in enumerate(cache.val):
                item=cache.items[i];row=item['row'];a=item;folder=out/'records.local'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
                mp=folder/'record.json'
                if mp.exists():
                    m=read_json(mp)
                    if m['contract_sha256']!=ch or m['sample_id']!=row['sample_id'] or sha(folder/'arrays.npz')!=m['arrays_sha256']:raise ValueError('New prediction cache changed.')
                else:
                    condition=np.stack([a['ct'],a['base']/bs])[None].astype(np.float32)
                    x=torch.from_numpy(condition).to(device)
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                    start=time.perf_counter()
                    with torch.inference_mode():z=model.predict(x,r['ode_steps'],r['particles'],r['source_seed'])
                    if str(device).startswith('cuda'):torch.cuda.synchronize()
                    seconds=time.perf_counter()-start
                    # Target is used only after the target-free model call above.
                    pr=z['residual'][0,0].cpu().numpy()*s;parts=z['parts'][0].cpu().numpy()*s
                    target=a['target'];factor=float(a['meta']['dose_scale_factor'])
                    raw=a['base']+pr;prediction=np.maximum(raw,0)
                    if not np.isfinite(prediction).all():raise FloatingPointError('Nonfinite new-model prediction.')
                    meta=dict(method=method,case_id=row['case_id'],record_id=row['record_id'],sample_id=row['sample_id'])
                    m=dict(**meta,contract_sha256=ch,
                        primary_metrics=dict(**meta,**score(prediction,target,factor)),
                        raw_metrics=dict(**meta,**score(raw,target,factor)),
                        diagnostics=dict(**meta,**residual_metrics(pr,target-a['base'],parts,s),
                            outside_positive=float(z['outside'][0,0].cpu()),outside_negative=float(z['outside'][0,1].cpu()),
                            preclamp_negative_fraction=float((raw<0).mean()),correction_only_seconds=seconds),
                        checkpoint_sha256=pointer['sha256'],dose_scale_factor=factor)
                    save_arrays(folder/'arrays.npz',dict(prediction_stored=(prediction/factor).astype(np.float32),
                        prediction_raw_stored=(raw/factor).astype(np.float32),residual_stored=(pr/factor).astype(np.float32),
                        positive_stored=(parts[0]/factor).astype(np.float32),negative_stored=(parts[1]/factor).astype(np.float32)))
                    m['arrays_sha256']=sha(folder/'arrays.npz');write_json(mp,m)
                rows.append(m['primary_metrics']);rawrows.append(m['raw_metrics']);diag.append(m['diagnostics'])
                if (j+1)%25==0 or j+1==len(cache.val):print(f'{method} val600 {j+1}/{len(cache.val)} saved/verified',flush=True)
                write_json(out/'progress.json',dict(completed=j+1,total=len(cache.val)),replace=True)
            rec,cases,summary=aggregate(rows);rec.to_csv(out/'metrics_records.local.csv',index=False);cases.to_csv(out/'metrics_cases.local.csv',index=False);summary.to_csv(out/'summary.csv',index=False)
            rr,cc,ss=aggregate(rawrows);rr.to_csv(out/'raw_metrics_records.local.csv',index=False);ss.to_csv(out/'raw_summary.csv',index=False)
            pd.DataFrame(diag).to_csv(out/'residual_diagnostics.local.csv',index=False)
            write_json(out/'run_status.json',dict(status='completed_new_validation',method=method,n_records=len(rows),n_cases=len(cases),
                contract_sha256=ch,scope=SCOPE,test_arrays_opened=0,scientific_task_status='PENDING_HUMAN_REVIEW'),replace=True)
        except BaseException as exc:
            write_json(out/'run_status.json',dict(status='interrupted_or_failed',error=type(exc).__name__,message=str(exc),scope=SCOPE),replace=True)
            raise
    print(summary.to_string(index=False));return out


def report(cache,evaluations,out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a new report directory; do not overwrite earlier conclusions.')
    old=Path(cache.contract['baseline_run']);bc=read_json(old/'contract.local.json')
    if digest_object(bc)!=cache.contract['baseline_contract_sha256']:raise ValueError('Original baseline contract changed.')
    rows=[]
    for i in cache.val:
        a=cache.items[i];r=a['row'];m_path=old/'records.local'/r['sample_id']/'record.json'
        if sha(m_path)!=a['meta']['baseline_record_sha256']:raise ValueError('Old baseline metadata changed.')
        m=read_json(m_path)
        if m['arrays_sha256']!=sha(m_path.parent/'arrays.npz'):raise IOError('Old baseline prediction bytes changed.')
        rows.extend(m['metrics'])
    evals=[];names=set();resources=[]
    for ev in evaluations:
        ev=Path(ev);st=read_json(ev/'run_status.json');co=read_json(ev/'contract.local.json')
        if st.get('status')!='completed_new_validation' or co['cache_identity']!=cache.identity or st['contract_sha256']!=digest_object(co):raise ValueError('Incomplete or different new evaluation.')
        if co['method'] in names:raise ValueError('Duplicate new method in comparison.')
        names.add(co['method']);evals.append(ev)
        resources.append({k:co[k] for k in ['method','trainable_parameters','pilot_training_records','monitor_records','optimizer_updates','batch_size','effective_solver','ode_steps','particles_per_branch','velocity_head_calls','shared_upstream']})
        for i in cache.val:
            r=cache.rows[i];folder=ev/'records.local'/r['sample_id'];m=read_json(folder/'record.json')
            if m['contract_sha256']!=digest_object(co) or m['arrays_sha256']!=sha(folder/'arrays.npz'):raise IOError('New prediction evidence changed.')
            rows.append(m['primary_metrics'])
    if not evals:raise ValueError('No new model supplied; this cannot be labelled a new-method comparison.')
    out.mkdir(parents=True);df,cases,summary=aggregate(rows)
    df.to_csv(out/'metrics_records.local.csv',index=False);cases.to_csv(out/'metrics_cases.local.csv',index=False);summary.to_csv(out/'comparison.csv',index=False)
    # Paired differences are descriptive only; 600 patches are NOT 600 patients.
    deltas=[]
    fields=['rmse_stored_units','mae_stored_units','x_mean_pct','y_mean_pct','z_mean_pct']
    for method in sorted(names):
        for baseline in ['phase9g','phase10d_strict']:
            p=df[df.method==method].set_index('sample_id');b=df[df.method==baseline].set_index('sample_id')
            for sid in p.index:
                deltas.append(dict(method=method,baseline=baseline,case_id=p.loc[sid,'case_id'],sample_id=sid,
                    **{k+'_delta':float(p.loc[sid,k]-b.loc[sid,k]) for k in fields}))
    dd=pd.DataFrame(deltas);dd.to_csv(out/'paired_deltas_records.local.csv',index=False)
    dc=dd.groupby(['method','baseline','case_id']).mean(numeric_only=True).reset_index();dc.to_csv(out/'paired_deltas_cases.local.csv',index=False)
    pd.DataFrame(resources).to_csv(out/'new_method_resources.csv',index=False)
    plot_comparison(cache,evals,out)
    text='''# Real Phase9G signed-error correction pilot\n\nThis is NOT a water-reference, physical transport, or blind final-test experiment.\nFrozen Phase9G predictions are a learned baseline, not dose-to-water.\nTargets: stored real GT minus frozen Phase9G, in the original shared model scale.\nNew heads start from fresh weights; they do not stack on Phase10D.\nUpstream historical training exposure remains unresolved. Two validation cases\nwere already used for development. Primary dose outputs are clamped at zero;\nraw-dose metrics and raw HJD components are retained in each evaluation directory.\n\nBudget: pilot uses the identifier-selected train cache (default 32 real records/case x 6 cases), not all 12,000 training records. Exact selections and budgets are in new_method_resources.csv.\nNew RF/HJD share train/monitor selections and optimizer-update budget, but differ\nin architecture, losses and computation. Old Phase10D has its historical budget.\nAll legacy x/y/z axis definitions and GT-peak diagnostic profile selection are\npreserved; x denotes array W, not independently verified beam depth.\n\n'''+summary.to_string(index=False)+'\n\nPaired deltas: negative means lower error; percent-metric deltas are percentage points.\nNo claim of new-method superiority is generated automatically.\n'
    (out/'report.md').write_text(text,encoding='utf-8')
    images=''.join(f'<p><img style="max-width:100%" src="{p.relative_to(out)}"></p>' for p in sorted((out/'figures.local').glob('*.png')))
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><main style="max-width:1100px;margin:auto;font-family:sans-serif"><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre>'+images+'</main>',encoding='utf-8')
    write_json(out/'REPORT_READY.json',dict(scope=SCOPE,new_method_comparison_completed=True,n_records=len(cache.val),n_validation_cases=len({cache.rows[i]['case_id'] for i in cache.val}),
        methods=list(summary.method),cache_identity=cache.identity,evaluations=[str(p) for p in evals],scientific_task_status='PENDING_HUMAN_REVIEW'))
    write_json(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='FILES.json'})
    print(summary.to_string(index=False));print('REPORT:',out/'report.html');return out


def plot_comparison(cache,evaluations,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dst=Path(out)/'figures.local';dst.mkdir();seen=set()
    old=Path(cache.contract['baseline_run'])
    for i in cache.val:
        a=cache.items[i];r=a['row']
        if r['case_id'] in seen:continue
        seen.add(r['case_id']);sid=r['sample_id']
        with np.load(old/'records.local'/sid/'arrays.npz',allow_pickle=False) as f:
            target=f['target_stored'].copy();preds={k:f[k].copy() for k in ['phase9g','phase10d_strict']}
        for ev in evaluations:
            name=read_json(ev/'contract.local.json')['method']
            with np.load(ev/'records.local'/sid/'arrays.npz',allow_pickle=False) as f:preds[name]=f['prediction_stored'].copy()
        for axis in 'xyz':
            fig,ax=plt.subplots(figsize=(8,4.5))
            for j,(name,pred) in enumerate(preds.items()):
                prof=LEGACY.extract_axis_profiles(pred,target)[axis]
                if j==0:ax.plot(prof['gt'],label='Target')
                ax.plot(prof['pred'],label=name)
            ax.set(xlabel=f'Legacy {axis} array index; GT-peak diagnostic line (not verified beam/mm)',ylabel='Dose in stored numerical units',title='Real validation '+sid)
            ax.legend(fontsize=8);fig.tight_layout();fig.savefig(dst/f'{sid}_{axis}.png',dpi=140);plt.close(fig)
