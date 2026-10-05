"""Common stored-array metrics; explicit legacy axis/threshold conventions."""
import numpy as np
import pandas as pd


def score(pred_stored,target_stored,factor):
    p=np.asarray(pred_stored,dtype=np.float64);y=np.asarray(target_stored,dtype=np.float64)
    if p.shape!=y.shape or y.ndim!=3 or not np.isfinite(p).all() or not np.isfinite(y).all(): raise ValueError('Invalid prediction/target.')
    if not np.isfinite(factor) or factor<=0: raise ValueError('Invalid stored-to-model factor.')
    error=p-y;peak_idx=np.unravel_index(np.argmax(y),y.shape);a,b,c=peak_idx
    out={'rmse_stored_units':float(np.sqrt(np.mean(error**2))), 'mae_stored_units':float(np.abs(error).mean()),
         'negative_prediction_fraction':float((p<0).mean())}
    selectors={'x':(a,b,slice(None)),'y':(a,slice(None),c),'z':(slice(None),b,c)}
    for axis,sl in selectors.items():
        pred=p[sl];gt=y[sl];gt_model=gt*factor;pred_model=pred*factor
        valid=gt_model>=.01*max(float(gt_model.max()),1e-8)
        pct=100*np.abs(pred_model[valid]-gt_model[valid])/(np.abs(gt_model[valid])+1e-8)
        out[axis+'_mean_pct']=float(pct.mean()) if len(pct) else None
        out[axis+'_max_pct']=float(pct.max()) if len(pct) else None
        out[axis+'_profile_rmse_stored_units']=float(np.sqrt(np.mean((pred-gt)**2)))
        out[axis+'_valid_count']=int(valid.sum())
    return out


def residual_diagnostics(pred_u,true_u,pred_magnitude=None,sign=None):
    active=np.abs(true_u)>.01;truthmag=np.abs(true_u);N=true_u.size
    d={'residual_rmse_normalized':float(np.sqrt(np.mean((pred_u-true_u)**2))),
       'target_magnitude_total':float(truthmag.sum()),'target_signed_total':float(true_u.sum()),
       'active_sign_accuracy':float((np.sign(pred_u[active])==np.sign(true_u[active])).mean()) if active.any() else None,
       'active_voxels':int(active.sum())}
    if pred_magnitude is not None:
        d.update(magnitude_mae_normalized=float(np.abs(pred_magnitude-truthmag).mean()),
          magnitude_total_relative_error=float(abs(pred_magnitude.sum()-truthmag.sum())/truthmag.sum()) if truthmag.sum()>0 else None,
          oracle_sign_residual_rmse=float(np.sqrt(np.mean((pred_magnitude*np.sign(true_u)-true_u)**2))),
          oracle_magnitude_residual_rmse=float(np.sqrt(np.mean((truthmag*sign-true_u)**2))),
          oracle_scope='training/development diagnostics using target factors; never model scores')
    return d


def aggregate(rows):
    df=pd.DataFrame(rows)
    if df.duplicated(['method','sample_id']).any(): raise ValueError('Duplicate method/sample metric.')
    keys=[k for k in df.columns if k.endswith(('_stored_units','_pct','_fraction'))]
    cases=df.groupby(['method','case_id'],sort=True)[keys].mean().reset_index()
    summary=cases.groupby('method',sort=True)[keys].mean().reset_index()
    for i,row in summary.iterrows():
        d=df[df.method==row['method']];summary.loc[i,'n_records']=len(d);summary.loc[i,'n_cases']=d.case_id.nunique()
    summary['aggregation']='equal_case_mean_of_per_record_metrics; development_not_blind_test'
    return df,cases,summary
