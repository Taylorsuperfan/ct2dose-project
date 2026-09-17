"""Explicit binding of the recovered Phase5E -> 9D+ -> 9G -> 10D-strict chain."""
from pathlib import Path
import importlib.util, uuid, math
import numpy as np
import torch
from .io import sha

PHASE9D_NAME='ct2dose_phase9d_plus_stronger_axis_refinement_seed42_best.pt'
PHASE10D_NAME='ct2dose_phase10d_falloff_aware_direction_refinement_strict_train6_val2_seed42_best.pt'
# This is the recorded parent shown in notebook cells 24-27, NOT the initial strong variant in cell 3.
EXPECTED_PARENT='ct2dose_phase5e_along_falloff_from_phase5bplus_seed42_lr5e6_ep3_g0p02_af0p01_sl0p005_best.pt'
REGIONS={'main_frac':0.30,'core_frac':0.70,'shoulder_low':0.30,'shoulder_high':0.80}
METHODS=('phase5e_rf','phase9d_plus','phase9g','phase10d_strict')


def new_legacy_namespace():
    p=Path(__file__).with_name('legacy_defs.py')
    spec=importlib.util.spec_from_file_location('_legacy_'+uuid.uuid4().hex,p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def load_restricted(path,trust=False):
    if not trust:raise PermissionError('Explicitly confirm that these are your trusted original checkpoints.')
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(path)
    if path.stat().st_size>256*1024**2:raise ValueError('Checkpoint exceeds the 256 MiB inspection limit.')
    before=sha(path)
    # Fail closed. No automatic weights_only=False, key dropping, or guessed architecture.
    try: ck=torch.load(path,map_location='cpu',weights_only=True)
    except Exception as exc:
        raise RuntimeError(f'Restricted checkpoint read failed for {path.name}. '
                           'Do not disable restrictions to bypass it; retain the exact error: '+str(exc)) from exc
    if before!=sha(path):raise IOError('Checkpoint changed while being read.')
    if not isinstance(ck,dict) or not isinstance(ck.get('model_state_dict'),dict):
        raise ValueError(f'Expected original model_state_dict in {path.name}')
    return ck,before


def strict_load(model,state):
    expected=model.state_dict()
    if set(expected)!=set(state):
        raise ValueError(f'State key mismatch. missing={sorted(set(expected)-set(state))}, extra={sorted(set(state)-set(expected))}')
    for k,v in expected.items():
        w=state[k]
        if not torch.is_tensor(w) or v.shape!=w.shape or v.dtype!=w.dtype:raise ValueError(f'State shape/dtype mismatch: {k}')
        if w.is_floating_point() and not torch.isfinite(w).all():raise ValueError(f'Nonfinite checkpoint tensor: {k}')
    model.load_state_dict(state,strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return dict(tensor_count=len(expected),state_elements=sum(t.numel() for t in expected.values()),
                parameters=sum(t.numel() for t in model.parameters()))


class LegacyPipeline:
    """Inference only. Its call accepts CT, never the ground-truth dose."""
    def __init__(self,root,device='cpu',trusted=False):
        root=Path(root);ckdir=root/'outputs/checkpoints'
        q9,h9=load_restricted(ckdir/PHASE9D_NAME,trusted)
        q10,h10=load_restricted(ckdir/PHASE10D_NAME,trusted)
        recorded=q9.get('base_checkpoint')
        if not isinstance(recorded,str):raise ValueError('Phase9D+ has no recorded base_checkpoint.')
        if Path(recorded).name!=EXPECTED_PARENT:
            raise ValueError('Recorded parent differs from the supplied notebook. Stop rather than substitute another base.')
        # Exact basename relocation under the declared project, as in original cell 25.
        bp=ckdir/Path(recorded).name
        qb,hb=load_restricted(bp,trusted)
        c9=q9.get('phase9d_plus_config');c10=q10.get('config');cb=qb.get('config',{})
        if not isinstance(c9,dict) or not isinstance(c10,dict):raise ValueError('Original configs missing.')
        required9=['euler_steps','log_scale_bound','additive_scale']
        required10=['euler_steps','base_ch','delta_scale','core_protect_strength','tail_protect_strength','threshold_frac',*REGIONS]
        for c,keys in [(c9,required9),(c10,required10)]:
            for key in keys:
                if key not in c:raise ValueError(f'Missing recorded field: {key}')
                if not isinstance(c[key],(int,float)) or not math.isfinite(c[key]):raise ValueError(f'Invalid {key}')
        if c10['euler_steps']!=c9['euler_steps'] or c9['euler_steps']!=10:
            raise ValueError('Euler steps differ from the corrected notebook reproduction (10).')
        if any(float(c10[k])!=v for k,v in REGIONS.items()):
            raise ValueError('Phase10D region thresholds differ from original Phase10C globals; explicit review required.')
        if int(c10['base_ch'])!=16:raise ValueError('The recovered Phase10D definition expects its recorded width 16.')
        if 'strict_train_json' not in c10 or 'strict_val_json' not in c10:
            raise ValueError('Not the strict checkpoint config previously inspected.')
        self.legacy=new_legacy_namespace();m=self.legacy
        m.CONFIG={'euler_steps':10,'core_scale':0.990,'core_thr':0.70,'tau':0.04,'threshold_frac':float(c10['threshold_frac'])}
        # Original feature functions refer to PHASE10C_CONFIG, not PHASE10D_CONFIG.
        m.PHASE10C_CONFIG=dict(REGIONS);m.PHASE10D_CONFIG=dict(c10)
        m.base_model=m.ConditionalUNetFlow3D(in_ch=3,out_ch=1,base_ch=int(cb.get('base_ch',24)))
        m.model_9d_plus=m.MultiplicativeAdditiveRefineHead3D(in_ch=7,base_ch=16,
                      log_scale_bound=float(c9['log_scale_bound']),additive_scale=float(c9['additive_scale']))
        m.phase10d_model=m.FalloffAwareBoundedRefineHead3D(in_ch=11,base_ch=16,delta_scale=float(c10['delta_scale']))
        stats={n:strict_load(model,state) for n,model,state in [
            ('base',m.base_model,qb['model_state_dict']),('phase9d_plus',m.model_9d_plus,q9['model_state_dict']),
            ('phase10d_strict',m.phase10d_model,q10['model_state_dict'])]}
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable.')
        for model in [m.base_model,m.model_9d_plus,m.phase10d_model]:model.to(self.device)
        self.dose_scale=float(cb.get('dose_scale',1000.0))
        if self.dose_scale<=0 or not math.isfinite(self.dose_scale):raise ValueError('Invalid recorded dose_scale.')
        self.pre=m.CTDoseNpyDataset.__new__(m.CTDoseNpyDataset);self.pre.dose_scale=self.dose_scale
        self.contract=dict(status='STRICT_STATE_BINDING_PASSED_NOT_YET_NUMERICAL_REPRODUCTION',
            checkpoints=[dict(role='base',name=bp.name,sha256=hb),dict(role='phase9d_plus',name=PHASE9D_NAME,sha256=h9),
                         dict(role='phase10d_strict',name=PHASE10D_NAME,sha256=h10)],
            base_config=cb,phase9d_plus_config=c9,phase10d_config=c10,core_calibration=m.CONFIG,
            state_stats=stats,dose_scale=self.dose_scale,
            defaults_used=[k for k in ['base_ch','dose_scale'] if k not in cb],
            effective_solver='legacy Euler state update at (i+0.5)/steps; clamp after every step; NOT RK2',steps=10,
            profile_axes={'x':'array W, dimension 2 of [D,H,W]','y':'array H','z':'array D'},
            feature_depth_axis='array D (historically called x); preserved mismatch with reported x=W',
            training_profiles='geometric centre; historically x=D. Evaluation: through GT peak, x=W.',
            data_scope='development validation only; no claim of whole-pipeline case/patient independence',
            new_reference_created=False)

    def normalize_ct(self,ct_raw):
        a=np.asarray(ct_raw,dtype=np.float32)
        if a.shape!=(32,32,32) or not np.isfinite(a).all():raise ValueError('Expected a finite raw (32,32,32) CT array.')
        return self.pre.normalize_ct(a)

    def scale_target_for_legacy_evaluation(self,dose):
        a=np.asarray(dose,dtype=np.float32)
        if a.shape!=(32,32,32) or not np.isfinite(a).all():raise ValueError('Expected finite target (32,32,32).')
        factor=self.dose_scale if float(a.max())<0.1 else 1.0
        return self.pre.scale_dose_if_needed(a),factor

    @torch.inference_mode()
    def predict_scaled(self,ct_raw):
        ct=torch.from_numpy(self.normalize_ct(ct_raw))[None,None].to(self.device)
        out=self.legacy.predict_phase10d(ct,steps=10)
        mapping={'phase5e_rf':'base_pred','phase9d_plus':'phase9d_pred','phase9g':'phase9g_pred','phase10d_strict':'phase10d_pred'}
        result={name:out[key][0,0].detach().cpu().numpy().copy() for name,key in mapping.items()}
        if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite model prediction.')
        return result
