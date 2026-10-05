"""Legacy profile semantics plus clearly separated stored-unit voxel errors."""
import numpy as np
from .pipeline import new_legacy_namespace
LEGACY=new_legacy_namespace()


def score(pred_scaled,target_scaled,factor):
    if pred_scaled.shape!=target_scaled.shape or not np.isfinite(pred_scaled).all():raise ValueError('Invalid prediction.')
    err=(pred_scaled.astype(np.float64)-target_scaled.astype(np.float64))/factor
    result={'rmse_stored_units':float(np.sqrt(np.mean(err**2))),'mae_stored_units':float(np.mean(np.abs(err))),
            'negative_prediction_fraction':float(np.mean(pred_scaled<0))}
    profiles=LEGACY.extract_axis_profiles(pred_scaled,target_scaled)
    for axis,d in profiles.items():
        # Original evaluator uses >= 1% of that GT line's peak and eps=1e-8 in MODEL scale.
        _,mean,maximum,n,thr=LEGACY.percentage_error_curve(d['pred'],d['gt'],threshold_frac=0.01)
        result[f'{axis}_mean_pct']=float(mean) if np.isfinite(mean) else None
        result[f'{axis}_max_pct']=float(maximum) if np.isfinite(maximum) else None
        result[f'{axis}_valid_count']=int(n)
        result[f'{axis}_threshold_model_scale']=float(thr)
        result[f'{axis}_profile_rmse_stored_units']=float(np.sqrt(np.mean(((d['pred'].astype(float)-d['gt'])/factor)**2)))
    return result
