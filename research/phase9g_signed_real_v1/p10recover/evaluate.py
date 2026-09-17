"""Resumable, read-only legacy inference and unified comparison on real validation."""
from pathlib import Path
import time, shutil, csv, json
from collections import defaultdict
import numpy as np
import pandas as pd
from .io import read_json,write_json,sha,canonical,digest_object,environment,code_hashes,save_arrays
from .pipeline import LegacyPipeline,METHODS
from .data import fixed_cohort,read_array
from .metrics import score


def aggregate(rows):
    df=pd.DataFrame(rows)
    keys=[k for k in df if k.endswith(('_pct','_stored_units','_fraction'))]
    cases=df.groupby(['method','case_id'],sort=True)[keys].mean().reset_index()
    summary=cases.groupby('method',sort=True)[keys].mean().reset_index()
    summary['n_cases']=summary['method'].map(cases.groupby('method').size())
    summary['n_records']=summary['method'].map(df.groupby('method').size())
    summary['aggregation']='equal-case mean of record metrics; no independent patient claim'
    return df,cases,summary


def verify_record(folder,row,contract_hash):
    meta=read_json(folder/'record.json')
    if meta['contract_sha256']!=contract_hash or meta['sample_id']!=row['sample_id']:raise ValueError('Cached record contract mismatch.')
    if sha(folder/'arrays.npz')!=meta['arrays_sha256']:raise IOError('Cached prediction bytes changed.')
    # No model re-evaluation for complete records, but preserve input identity.
    for key in ['input_path','output_path']:
        expected=meta['ct_sha256' if key=='input_path' else 'target_sha256']
        if sha(row[key])!=expected:raise IOError('Underlying medical file changed since cached prediction.')
    return meta


def run(root,out,mode='smoke',device='cpu',trusted=False,make_plots=True):
    root=Path(root).resolve();out=Path(out).resolve()
    if out==root or out.is_relative_to(root):raise ValueError('Outputs must be outside the original Practical tree.')
    out.mkdir(parents=True,exist_ok=True)
    pipeline=LegacyPipeline(root,device=device,trusted=trusted)
    cohort,cohort_meta=fixed_cohort(root,mode)
    contract={'schema':'phase10d_recovered_comparison_v1','pipeline':pipeline.contract,'cohort':cohort_meta,'records':cohort,
              'code_sha256':code_hashes(),'environment':environment(),'device':str(pipeline.device),
              'methods':list(METHODS),'comparison_kind':'legacy component comparison; no HJD/new-method prediction is supplied'}
    c_hash=digest_object(contract)
    if (out/'contract.local.json').exists():
        if read_json(out/'contract.local.json')!=contract:
            raise ValueError('Existing run has different code/checkpoints/cohort/environment. Use a new run directory; no silent mixing.')
    else:
        if any(out.iterdir()):raise FileExistsError('Nonempty output directory without run contract.')
        write_json(out/'contract.local.json',contract)
        src=out/'source_snapshot';src.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,src/p.name)
    write_json(out/'run_status.json',{'status':'running','contract_sha256':c_hash,'scientific_task_status':'PENDING_HUMAN_REVIEW'},replace=True)
    logpath=out/'inference.log'
    allrows=[]
    for i,row in enumerate(cohort):
        folder=out/'records.local'/row['sample_id'];folder.mkdir(parents=True,exist_ok=True)
        if (folder/'record.json').is_file():
            meta=verify_record(folder,row,c_hash);allrows.extend(meta['metrics']);message=f'{i+1}/{len(cohort)} cached {row["sample_id"]}'
        else:
            ct,ct_sha=read_array(row['input_path'])
            started=time.perf_counter()
            # IMPORTANT: inference receives CT only; target values have not yet been loaded.
            predictions=pipeline.predict_scaled(ct)
            seconds=time.perf_counter()-started
            target,target_sha=read_array(row['output_path'])
            target_scaled,factor=pipeline.scale_target_for_legacy_evaluation(target)
            metrics=[dict(method=m,case_id=row['case_id'],record_id=row['record_id'],sample_id=row['sample_id'],
                          dose_scale_factor=factor,**score(p,target_scaled,factor)) for m,p in predictions.items()]
            arrays={'target_stored':target,'ct_normalized':pipeline.normalize_ct(ct)}
            arrays.update({m:(p/factor).astype(np.float32) for m,p in predictions.items()})
            save_arrays(folder/'arrays.npz',arrays)
            meta=dict(sample_id=row['sample_id'],case_id=row['case_id'],record_id=row['record_id'],
                      contract_sha256=c_hash,ct_sha256=ct_sha,target_sha256=target_sha,
                      dose_scale_factor=factor,arrays_sha256=sha(folder/'arrays.npz'),
                      methods=list(METHODS),metrics=metrics,cumulative_chain_inference_seconds=seconds,
                      timing_note='shared intermediate chain; not a per-method latency benchmark')
            write_json(folder/'record.json',meta)
            allrows.extend(metrics);message=f'{i+1}/{len(cohort)} saved {row["sample_id"]} | chain seconds={seconds:.3f}'
        print(message,flush=True)
        with logpath.open('a',encoding='utf-8') as f:f.write(message+'\n')
        write_json(out/'progress.json',{'complete_records':i+1,'total_records':len(cohort)},replace=True)
    df,cases,summary=aggregate(allrows)
    df.to_csv(out/'metrics_records.local.csv',index=False);cases.to_csv(out/'metrics_cases.local.csv',index=False)
    summary.to_csv(out/'comparison.csv',index=False)
    # Save a common prediction contract for future new-method predictions. No placeholder model is ranked.
    write_json(out/'cohort_for_external_predictions.local.json',{
        'schema':'real_dose_predictions_v1','data_scope':'real_validation','cohort_sha256':digest_object(cohort),
        'unit':'same numerical units as target .npy (physical unit unverified)',
        'records':[dict(sample_id=r['sample_id'],case_id=r['case_id'],ct_path=r['input_path'],
                        ct_sha256=read_json(out/'records.local'/r['sample_id']/'record.json')['ct_sha256']) for r in cohort]},replace=True)
    if make_plots: plot_run(out,cohort)
    historical=None
    if mode=='historical-val600':
        historical=compare_reported(out,summary)
    write_report(out,summary,historical)
    write_json(out/'run_status.json',{'status':'completed_validation_inference','n_records':len(cohort),'n_methods':len(METHODS),
        'contract_sha256':c_hash,'scientific_task_status':'PENDING_HUMAN_REVIEW','new_method_comparison_completed':False,
        'model_state_binding':'strict','historical_numeric_identity':'not established merely by source or load success',
        'test_arrays_opened_by_this_runner':0},replace=True)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='FILES.json' and not p.name.startswith('._')}
    write_json(out/'FILES.json',files,replace=True)
    print('REPORT:',out/'report.html',flush=True)
    print(summary[['method','rmse_stored_units','x_mean_pct','y_mean_pct','z_mean_pct']].to_string(index=False))
    return out


def plot_run(out,cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .metrics import LEGACY
    d=Path(out)/'figures.local';d.mkdir(exist_ok=True)
    # A predetermined small display subset: first record in each validation case.
    seen=set()
    for r in cohort:
        if r['case_id'] in seen:continue
        seen.add(r['case_id'])
        with np.load(Path(out)/'records.local'/r['sample_id']/'arrays.npz',allow_pickle=False) as a:
            for axis in 'xyz':
                fig,ax=plt.subplots(figsize=(8,4.5));target=a['target_stored']
                for j,m in enumerate(METHODS):
                    p=LEGACY.extract_axis_profiles(a[m],target)[axis]
                    if j==0:ax.plot(p['gt'],label='Target')
                    ax.plot(p['pred'],label=m)
                ax.set(xlabel=f'Legacy {axis} array index, GT-peak line (not verified beam/mm)',
                       ylabel='Dose in stored numerical units',title=f'Validation record {r["sample_id"]}')
                ax.legend(fontsize=8);fig.tight_layout();fig.savefig(d/f'{r["sample_id"]}_{axis}.png',dpi=140);plt.close(fig)


def compare_reported(out,summary):
    report_path=Path(__file__).parent.parent/'records/historical_validation_reported.csv'
    old=pd.read_csv(report_path);rows=[]
    for _,a in old.iterrows():
        for method,col in [('phase9g','phase9g_mean_pct'),('phase10d_strict','phase10d_mean_pct')]:
            new=float(summary.loc[summary.method==method,f'{a.axis}_mean_pct'].iloc[0])
            rows.append(dict(axis=a.axis,method=method,archived_mean_pct=float(a[col]),fresh_mean_pct=new,
                             delta_percentage_points=new-float(a[col])))
    pd.DataFrame(rows).to_csv(Path(out)/'historical_summary_delta.csv',index=False)
    note={'comparison':'aggregate consistency check only','max_abs_delta_percentage_points':max(abs(r['delta_percentage_points']) for r in rows),
          'limitation':'The supplied archive has batch/local indices but no original manifest hash. Same filename/cohort size alone does not prove original record identity/order. No tuning to match these numbers.',
          'scientific_acceptance':'PENDING_HUMAN_REVIEW'}
    write_json(Path(out)/'historical_summary_check.json',note,replace=True);return note


def write_report(out,summary,historical):
    import html
    text='''# Recovered Phase10D-strict legacy-chain comparison\n\nREAL validation data, stored numerical units. Not a clinical/physical-unit validation.\n\nActual definitions were recovered from the supplied notebook; weights were strictly loaded at runtime.\nPhase5E RF here is the base used by Phase9G, not the early Phase3/Optuna system.\nNo artificial/water reference was generated. This table compares old-system components;\nno new HJD model has yet supplied predictions on this same real cohort.\n\nOriginal semantics intentionally preserved: CT normalization heuristic, dose-scale heuristic,\nCT-initialized Euler-like update with midpoint TIME and per-step nonnegative clipping,\nlegacy GT-peak line profiles. Feature/training x=D and reported x=W are inconsistent in the\noriginal source. They are NOT silently fixed; this affects physical/directional claims.\n\nCurrent train/validation case-list disjointness does not erase upstream training exposure.\nOnly validation arrays were evaluated; no historical or final test arrays were loaded.\n\n'''
    text+=summary.to_string(index=False)+'\n\n'
    if historical:text+='Archived summary check:\n'+json.dumps(historical,indent=2)+'\n'
    Path(out,'report.md').write_text(text,encoding='utf-8')
    images=''.join(f'<p><img style="max-width:100%" src="{html.escape(str(p.relative_to(out)))}"></p>' for p in sorted(Path(out,'figures.local').glob('*.png')))
    Path(out,'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Legacy comparison</title><main style="max-width:1100px;margin:auto;font-family:sans-serif"><pre style="white-space:pre-wrap">'+html.escape(text)+'</pre>'+images+'</main>',encoding='utf-8')
