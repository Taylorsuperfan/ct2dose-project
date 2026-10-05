"""Fixed validation inference and paired comparison to saved, unmodified baselines."""
from pathlib import Path
import time, uuid, os
import numpy as np
import pandas as pd
import torch
from . import common as io
from .metrics import score, residual_diagnostics, aggregate
from .train import load_model

NAMES={
 'shared_calibrated_predictor':'Shared calibrated dose predictor',
 'previous_final_system':'Previous final Practical dose system',
 'previous_direct_signed_flow':'Previous directly learned signed correction',
 'previous_positive_negative_flows':'Previous separate positive/negative correction flows',
 'conditional_wfr_grid_sign':'New: conditional WFR magnitude with grid sign',
 'conditional_wfr_hard_sign_sensitivity':'New: hard grid-sign sensitivity (secondary)',
}


def _save_npz(path,arrays):
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def evaluate(cache,run,out,device='cpu'):
    out=Path(out).resolve();io.separate(out,run,cache.path,cache.old);io.seed_all(12345)
    model,cfg,c,pointer=load_model(run,cache,device)
    if cfg.stage!='pilot':raise ValueError('Validation600 is available only for the declared pilot, not learnability runs.')
    ids=[r['sample_id'] for r in cache.val]
    core={'scope':'SAME_REAL_VALIDATION600_DEVELOPMENT_NOT_BLIND_TEST','train_contract':io.digest(c),
          'checkpoint':pointer,'cache_identity':cache.identity,'validation_ids':ids,
          'source_code':io.source_identity(),'ode_steps':cfg.ode_steps,'source_quadrature_points':32768,
          'trainable_parameters':sum(p.numel() for p in model.parameters()),'optimizer_updates':cfg.updates,
          'batch_size':cfg.batch_size,'pilot_training_records':len(c['identity']['selection']['train_ids']),
          'monitor_records':len(c['identity']['selection']['monitor_ids']),'selected_checkpoint_step':pointer['step'],
          'solver':'Heun_in_logit_position_and_log_mass','reconstruction':'trilinear_magnitude_then_grid_sign',
          'primary':'soft_sign_then_final_absolute_dose_clamp','raw_and_hard_saved_separately':True}
    out.mkdir(parents=True,exist_ok=True)
    with io.single_writer(out):
        cp=out/'contract.json'
        if cp.exists():
            old=io.read(cp)
            if old['identity']!=core:raise ValueError('Evaluation inputs/config differ; use a new evaluation directory.')
            if (out/'COMPLETE.json').exists():io.verify_finish(out,'validation_inference_completed');print('Saved evaluation verified; no inference.',flush=True);return
            if old['environment']!=io.environment(device):raise ValueError('Partial evaluation environment changed. Use a new evaluation directory.')
        else:io.write(cp,{'identity':core,'environment':io.environment(device)})
        ec=io.digest(io.read(cp));folder=out/'records.local';folder.mkdir(exist_ok=True);rows=[];rawrows=[];diags=[]
        for j,sid in enumerate(ids,1):
            record=folder/sid;record.mkdir(exist_ok=True);meta=record/'record.json';arr=record/'arrays.npz'
            item=cache.load(sid,allow_validation=True);target=item['original']['target_stored'];factor=item['factor']
            if meta.exists():
                m=io.read(meta)
                if m['contract_sha256']!=ec or m['source_arrays_sha256']!=item['row']['arrays_sha256'] or io.sha(arr)!=m['arrays_sha256']:
                    raise ValueError('Existing evaluation record changed.')
            else:
                if str(device).startswith('cuda'):torch.cuda.synchronize()
                start=time.monotonic()
                # No target, target total, target sign or case identifier enters predict().
                z=model.predict(torch.tensor(item['condition'][None],device=device),cfg.ode_steps,cfg.query_chunk)
                if str(device).startswith('cuda'):torch.cuda.synchronize()
                duration=time.monotonic()-start;S=cache.norm['residual_rms_model_scale']
                raw=(item['base'].astype(float)+S*z['residual_normalized'])/factor
                hardu=z['magnitude']*np.where(z['sign']>=0,1.,-1.)
                hardraw=(item['base'].astype(float)+S*hardu)/factor
                outputs={'prediction_stored':np.maximum(raw,0).astype(np.float32),'raw_prediction_stored':raw.astype(np.float32),
                         'hard_prediction_stored':np.maximum(hardraw,0).astype(np.float32),
                         'magnitude_normalized':z['magnitude'].astype(np.float32),'soft_sign':z['sign'].astype(np.float32)}
                _save_npz(arr,outputs)
                # Re-score serialized values; comparisons use the same stored-array arithmetic.
                stats=score(outputs['prediction_stored'],target,factor)
                rstats=score(outputs['raw_prediction_stored'],target,factor)
                hstats=score(outputs['hard_prediction_stored'],target,factor)
                diagnostic=residual_diagnostics(z['residual_normalized'],item['residual'],z['magnitude'],z['sign'])
                m={'sample_id':sid,'case_id':item['row']['case_id'],'contract_sha256':ec,'source_arrays_sha256':item['row']['arrays_sha256'],
                   'source_record_sha256':item['row']['record_meta_sha256'],'arrays_sha256':io.sha(arr),
                   'primary_metrics':stats,'raw_metrics':rstats,'hard_metrics':hstats,'diagnostics':diagnostic,
                   'inference_diagnostics':z['diagnostics'],'inference_seconds':duration}
                io.write(meta,m)
            rows.append({'method':'conditional_wfr_grid_sign','sample_id':sid,'case_id':m['case_id'],**m['primary_metrics']})
            rawrows.append({'method':'conditional_wfr_raw','sample_id':sid,'case_id':m['case_id'],**m['raw_metrics']})
            diags.append({'sample_id':sid,'case_id':m['case_id'],**m['diagnostics'],'inference_seconds':m['inference_seconds']})
            if j%25==0 or j==1:print(f'VALIDATION {j}/{len(ids)} saved/verified',flush=True)
        frame,cases,summary=aggregate(rows)
        frame.to_csv(out/'metrics_records.local.csv',index=False);cases.to_csv(out/'metrics_cases.local.csv',index=False)
        summary.to_csv(out/'summary.csv',index=False);pd.DataFrame(rawrows).to_csv(out/'raw_metrics.local.csv',index=False)
        pd.DataFrame(diags).to_csv(out/'factor_diagnostics.local.csv',index=False)
        cache.check_metadata_unchanged();io.finish(out,'validation_inference_completed',contract_sha256=ec,n_records=len(ids),n_cases=2,test_arrays_opened=0)
        print(summary.to_string(index=False),flush=True)


def _previous_eval(path,cache,expected_method):
    path=Path(path).resolve();c=io.read(path/'contract.local.json');s=io.read(path/'run_status.json')
    if c.get('schema')!='real_phase9g_new_evaluation_v1' or c['method']!=expected_method or c['cache_identity']!=cache.identity:
        raise ValueError('Previous correction evaluation has different method/cache identity.')
    if c['val_ids']!=[r['sample_id'] for r in cache.val]:raise ValueError('Previous correction validation cohort/order differs.')
    if s.get('status')!='completed_new_validation' or s['contract_sha256']!=io.digest(c):raise ValueError('Previous evaluation is incomplete.')
    return path,c


def compare(cache,new_eval,hjd_eval,direct_eval,out):
    out=Path(out).resolve();new_eval=Path(new_eval).resolve()
    io.separate(out,cache.path,cache.old,new_eval,hjd_eval,direct_eval)
    io.verify_finish(new_eval,'validation_inference_completed');nc=io.read(new_eval/'contract.json');ni=nc['identity']
    if ni['cache_identity']!=cache.identity or ni['validation_ids']!=[r['sample_id'] for r in cache.val]:raise ValueError('New evaluation cohort/cache mismatch.')
    old_contract=io.read(cache.old/'contract.local.json');old_status=io.read(cache.old/'run_status.json')
    if io.digest(old_contract)!=cache.contract['baseline_contract_sha256'] or old_status.get('status')!='completed_validation_inference' or old_status.get('n_records')!=600:
        raise ValueError('Previous final Practical evaluation is not the saved full600 baseline.')
    hjd,hc=_previous_eval(hjd_eval,cache,'new_hjd_rf');direct,dc=_previous_eval(direct_eval,cache,'new_residual_rf')
    inputs={'new':io.sha(new_eval/'contract.json'),'new_receipt':io.sha(new_eval/'FILES.json'),
            'previous_final':io.sha(cache.old/'contract.local.json'),'hjd':io.sha(hjd/'contract.local.json'),
            'direct':io.sha(direct/'contract.local.json'),'cache_identity':cache.identity}
    out.mkdir(parents=True,exist_ok=True);io.write(out/'contract.json',inputs)
    if (out/'COMPLETE.json').exists():io.verify_finish(out,'paired_comparison_completed');print('Existing paired comparison verified.');return
    rows=[];sample_manifest=[]
    for j,row in enumerate(cache.val,1):
        sid=row['sample_id'];item=cache.load(sid,allow_validation=True);target=item['original']['target_stored'];factor=item['factor']
        nf=new_eval/'records.local'/sid;nm=io.read(nf/'record.json')
        if nm['source_arrays_sha256']!=row['arrays_sha256'] or nm['case_id']!=row['case_id']:raise ValueError('New record pairing differs.')
        with np.load(nf/'arrays.npz',allow_pickle=False) as z:
            predictions={'conditional_wfr_grid_sign':z['prediction_stored'].copy(),
                         'conditional_wfr_hard_sign_sensitivity':z['hard_prediction_stored'].copy(),
                         'shared_calibrated_predictor':np.maximum(item['original']['phase9g'],0),
                         'previous_final_system':item['original']['phase10d_strict']}
        provenance={'sample_id':sid,'case_id':row['case_id'],'target_source_sha256':row['arrays_sha256']}
        for root,c,name in [(hjd,hc,'previous_positive_negative_flows'),(direct,dc,'previous_direct_signed_flow')]:
            folder=root/'records.local'/sid;m=io.read(folder/'record.json')
            if m['sample_id']!=sid or m['case_id']!=row['case_id'] or m['contract_sha256']!=io.digest(c):raise ValueError('Previous method record pairing differs.')
            h=io.sha(folder/'arrays.npz')
            if h!=m['arrays_sha256']:raise ValueError('Previous prediction bytes changed.')
            with np.load(folder/'arrays.npz',allow_pickle=False) as z:predictions[name]=z['prediction_stored'].copy()
            provenance[name+'_prediction_sha256']=h
        for name,pred in predictions.items():
            rows.append({'method':name,'sample_id':sid,'case_id':row['case_id'],**score(pred,target,factor)})
        sample_manifest.append(provenance)
        if j%100==0:print(f'PAIRED COMPARISON {j}/600, no model inference',flush=True)
    frame,cases,summary=aggregate(rows)
    summary.insert(1,'display_name',summary.method.map(NAMES))
    frame.to_csv(out/'paired_records.local.csv',index=False);cases.to_csv(out/'case_metrics.local.csv',index=False)
    summary.to_csv(out/'comparison.csv',index=False);io.write(out/'record_provenance.local.json',sample_manifest)
    delta=[]
    metrics=[c for c in cases.columns if c not in ('method','case_id')]
    for baseline in ('previous_final_system','previous_positive_negative_flows','previous_direct_signed_flow','shared_calibrated_predictor'):
        left=cases[cases.method=='conditional_wfr_grid_sign'].set_index('case_id');right=cases[cases.method==baseline].set_index('case_id')
        if set(left.index)!=set(right.index):raise ValueError('Case sets differ.')
        for case in left.index:delta.append({'baseline':baseline,'case_id':case,**{k+'_delta':float(left.loc[case,k]-right.loc[case,k]) for k in metrics}})
    pd.DataFrame(delta).to_csv(out/'paired_case_deltas.local.csv',index=False)
    budgets=[]
    for label,c in [('previous_positive_negative_flows',hc),('previous_direct_signed_flow',dc)]:
        budgets.append({'method':label,**{k:c.get(k) for k in ('trainable_parameters','pilot_training_records','monitor_records','optimizer_updates','batch_size','ode_steps','particles_per_branch','velocity_head_calls','effective_solver')}})
    # New budget comes from its linked training contract, not assumed from the method name.
    budgets.append({'method':'conditional_wfr_grid_sign',**{k:ni[k] for k in ('trainable_parameters','optimizer_updates','batch_size','pilot_training_records','monitor_records')},
                    'ode_steps':ni['ode_steps'],'inference_source_particles':32768,
                    'velocity_growth_field_evaluations_per_particle':2*ni['ode_steps'],'effective_solver':ni['solver'],
                    'training_contract_sha256':ni['train_contract']})
    pd.DataFrame(budgets).to_csv(out/'compute_scope.csv',index=False)
    io.write(out/'method_names.json',NAMES)
    write_report(out,summary)
    cache.check_metadata_unchanged();io.finish(out,'paired_comparison_completed',n_records=600,n_cases=2,n_methods=len(summary),
         scientific_status='DEVELOPMENT_COMPARISON_NO_AUTOMATIC_SUPERIORITY_CLAIM',test_arrays_opened=0)
    print(summary.to_string(index=False),flush=True);print('COMPARISON:',out,flush=True)


def write_report(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);figs=out/'figures';figs.mkdir(exist_ok=True)
    main=summary[summary.method!='conditional_wfr_hard_sign_sensitivity'].copy()
    for metric,name,title in [('rmse_stored_units','01_whole_cube_rmse','Whole-cube dose RMSE'),
                              ('x_mean_pct','02_array_x_percentage','Array-x dose-line mean percentage error'),
                              ('x_profile_rmse_stored_units','03_array_x_absolute','Array-x dose-line absolute RMSE')]:
        fig,ax=plt.subplots(figsize=(10,5));ax.barh(main.display_name,main[metric]);ax.invert_yaxis()
        ax.set_xlabel('Percent' if metric.endswith('_pct') else 'Stored dose units (not verified Gy)');ax.set_title(title)
        fig.tight_layout();fig.savefig(figs/(name+'.png'),dpi=160);plt.close(fig)
    lines=['# Conditional WFR real-data development comparison','',
           'All methods are scored from saved predictions on the same 600 cubes from two validation cases.',
           'The 40 model-selection records are included. Historical upstream exposure is not removed.',
           'No patient-level independent test, water-reference or clinical-readiness claim is made.','',
           'The previous final Practical system is Phase10D-strict, not a newly recreated Phase3 network.',
           'The previous positive/negative method and direct signed flow use their unmodified saved predictions.',
           'Metrics are recomputed uniformly from serialized float32 stored-unit predictions. Small differences from earlier pre-serialization tables are possible.',
           'Array x/y/z are indexing labels, not independently verified beam directions.','',
           '## Results','',summary.to_markdown(index=False), '',
           '## Interpretation rule','',
           'Lower error is better. Report the direction and magnitude of both whole-cube and dose-profile changes.',
           'A lower global RMSE alone is not evidence that the previous final profile objective improved.',
           'The hard-sign row is a predeclared secondary sensitivity, not a replacement chosen to win.',
           'Case-level deltas are provided without patient-independence or significance claims.','',
           'See compute_scope.csv and the training summary for unequal architectures, histories and computation.']
    for image in sorted(figs.glob('*.png')):lines+=['',f'![{image.stem}](figures/{image.name})']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
