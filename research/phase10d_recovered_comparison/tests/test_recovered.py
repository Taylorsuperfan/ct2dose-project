from pathlib import Path
import json,hashlib,shutil
import numpy as np
import pytest
import torch
from p10recover.pipeline import new_legacy_namespace,LegacyPipeline,PHASE9D_NAME,PHASE10D_NAME,EXPECTED_PARENT,REGIONS,strict_load,METHODS
from p10recover.io import write_json,read_json,sha,digest_object
from p10recover.data import fixed_cohort,SPLIT,TRAIN_NAME,VAL_NAME,SUBSET_NAME
from p10recover.evaluate import run,aggregate
from p10recover.metrics import score
from p10recover.external import compare_external
from p10recover.verify import verify_source

torch.set_num_threads(1)

@pytest.fixture
def prepared(tmp_path):
    torch.manual_seed(7);root=tmp_path/'practical';out=tmp_path/'out';ck=root/'outputs/checkpoints';ck.mkdir(parents=True)
    m=new_legacy_namespace();d={**REGIONS,'seed':42,'epochs':2,'euler_steps':10,'base_ch':16,'delta_scale':.008,
      'core_protect_strength':.8,'tail_protect_strength':.25,'threshold_frac':.01,'strict_train_json':'recorded_train',
      'strict_val_json':'recorded_val','strict_test_json':'NEVER_OPEN_THIS','falloff_low_frac':.05,'falloff_high_frac':.45,
      'falloff_margin_log':.015,'perp_margin_abs':.0004,'core_margin_abs':.0004,
      **{f'lambda_{k}':1.0 for k in ['voxel','along','along_slope','along_log','falloff_log','falloff_no_worse','perp_no_worse','core_no_worse','residual_l1','smooth']}}
    m.PHASE10C_CONFIG=dict(REGIONS);m.PHASE10D_CONFIG=d
    torch.save({'model_state_dict':m.ConditionalUNetFlow3D(base_ch=24).state_dict(),'config':{'base_ch':24,'dose_scale':1000.}},ck/EXPECTED_PARENT)
    torch.save({'model_state_dict':m.MultiplicativeAdditiveRefineHead3D(log_scale_bound=.2,additive_scale=.012).state_dict(),
                'base_checkpoint':'/OLD_DRIVE/'+EXPECTED_PARENT,'phase9d_plus_config':{'euler_steps':10,'log_scale_bound':.2,'additive_scale':.012}},ck/PHASE9D_NAME)
    torch.save({'model_state_dict':m.FalloffAwareBoundedRefineHead3D().state_dict(),'config':d,'epoch':2},ck/PHASE10D_NAME)
    sd=root/SPLIT;sd.mkdir(parents=True);train=[];val=[]
    for split,cases in [('train',['train_a']),('val',['val_a','val_b'])]:
      for case in cases:
        for i in range(2):
          r=root/'data/raw'/case;r.mkdir(parents=True,exist_ok=True)
          ct=r/f'{i}_ct.npy';gt=r/f'{i}_gt.npy'
          grid=np.linspace(-.5,1,32**3,dtype=np.float32).reshape((32,)*3)
          np.save(ct,grid*1024-200);np.save(gt,(.001+.0002*grid).astype(np.float32))
          rec={'case_id':case,'input_path':str(ct),'output_path':str(gt)}
          (train if split=='train' else val).append(rec)
    write_json(sd/TRAIN_NAME,train);write_json(sd/VAL_NAME,val)
    return root,out,d


def test_source_exact():assert len(verify_source()['definitions'])==30


def test_state_counts():
 m=new_legacy_namespace()
 for width,expected in [(24,763993),(32,1357089)]:
  s=m.ConditionalUNetFlow3D(base_ch=width).state_dict();assert len(s)==26 and sum(v.numel() for v in s.values())==expected
 s=m.FalloffAwareBoundedRefineHead3D().state_dict();assert len(s)==18 and sum(v.numel() for v in s.values())==25697


def test_strict_rejects_missing():
 m=new_legacy_namespace().ConditionalUNetFlow3D();s=m.state_dict();s.pop('out_conv.bias')
 with pytest.raises(ValueError):strict_load(m,s)


def test_strict_rejects_dtype():
 m=new_legacy_namespace().ConditionalUNetFlow3D();s=m.state_dict();s['out_conv.bias']=s['out_conv.bias'].double()
 with pytest.raises(ValueError):strict_load(m,s)


def test_trust_required(prepared):
 with pytest.raises(PermissionError):LegacyPipeline(prepared[0])


def test_pipeline_binding_and_counts(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);assert p.contract['steps']==10
 assert p.contract['state_stats']['phase10d_strict']['parameters']==25697
 assert p.contract['checkpoints'][0]['name']==EXPECTED_PARENT


def test_actual_ct_normalization(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);a=np.zeros((32,)*3,np.float32);a[0,0,:4]=[-2048,-1024,0,2048]
 assert np.array_equal(p.normalize_ct(a)[0,0,:4],[0,0,.5,1])
 b=np.full((32,)*3,.25,np.float32);assert np.array_equal(p.normalize_ct(b),b)


def test_dose_scaling(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);a=np.full((32,)*3,.003,np.float32);scaled,f=p.scale_target_for_legacy_evaluation(a)
 assert f==1000 and np.allclose(scaled,3)
 a=np.full((32,)*3,.1,np.float32);scaled,f=p.scale_target_for_legacy_evaluation(a);assert f==1


def test_sampler_time_and_clipping():
 m=new_legacy_namespace()
 class Field(torch.nn.Module):
  def __init__(self):super().__init__();self.times=[]
  def forward(self,x,ct,t):self.times.append(float(t[0]));return -torch.ones_like(x)
 field=Field();x=torch.full((1,1,4,4,4),.2);y=m.euler_sample_ct2dose(field,x,steps=2)
 assert field.times==[.25,.75] and y.min()==0


def test_axis_mismatch_preserved():
 m=new_legacy_namespace();a=torch.arange(32.).reshape(1,1,32,1,1).expand(1,1,32,32,32)
 depth,_=m.make_coord_channels_like(a);assert depth[0,0,0,0,0]==0 and depth[0,0,-1,0,0]==1
 profiles=m.extract_axis_profiles(a[0],a[0]);assert np.ptp(profiles['x']['gt'])==0 and np.ptp(profiles['z']['gt'])==31


def test_zero_head_identity(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);h=p.legacy.phase10d_model
 with torch.no_grad():h.delta_head.weight.zero_();h.delta_head.bias.zero_()
 ct=torch.rand(1,1,32,32,32);pred=torch.rand_like(ct)
 got,aux=h(ct,pred);assert torch.equal(got,pred) and not aux['delta'].any()


def test_10d_correction_bound(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);ct=torch.rand(1,1,32,32,32);pred=torch.rand_like(ct)
 got,aux=p.legacy.phase10d_model(ct,pred)
 assert torch.max(torch.abs(aux['delta']))<=.008 and got.min()>=0


def test_finite_original_forward_and_loss(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);p.legacy.phase10d_model.requires_grad_(True)
 ct=torch.rand(1,1,32,32,32);pred=.1+torch.rand_like(ct);gt=.1+torch.rand_like(ct)
 y,aux=p.legacy.phase10d_model(ct,pred);loss,stats=p.legacy.phase10d_loss(y,pred,gt,aux)
 assert torch.isfinite(loss);loss.backward();assert p.legacy.phase10d_model.delta_head.weight.grad is not None


def test_ct_only_predict(prepared):
 p=LegacyPipeline(prepared[0],trusted=True);out=p.predict_scaled(np.ones((32,)*3,np.float32)*100)
 assert set(out)==set(METHODS) and all(v.shape==(32,)*3 and np.isfinite(v).all() for v in out.values())


def test_record_cohort(prepared):
 rows,meta=fixed_cohort(prepared[0]);assert len(rows)==4 and meta['current_case_disjoint'] and not meta['old_test_json_opened']


def test_no_test_mode(prepared):
 with pytest.raises(ValueError):fixed_cohort(prepared[0],'test')


def test_no_generated_historical_subset(prepared):
 with pytest.raises(FileNotFoundError):fixed_cohort(prepared[0],'historical-val600')


def test_overlap_fail(prepared):
 root=prepared[0];p=root/SPLIT/VAL_NAME;v=read_json(p);v[0]['case_id']='train_a';write_json(p,v,replace=True)
 with pytest.raises(ValueError):fixed_cohort(root)


def test_metric_exact_and_sign():
 a=np.ones((32,)*3,np.float32);m=score(a,a,1000);assert m['rmse_stored_units']==0 and m['x_mean_pct']==0
 m=score(a*2,a,1000);assert m['rmse_stored_units']==.001 and abs(m['x_mean_pct']-100)<1e-5


def test_case_equal_aggregation():
 rows=[dict(method='a',case_id='one',rmse_stored_units=1.),dict(method='a',case_id='one',rmse_stored_units=3.),dict(method='a',case_id='two',rmse_stored_units=8.)]
 _,c,s=aggregate(rows);assert s.rmse_stored_units.iloc[0]==5


def test_e2e_resume_and_no_source_change(prepared):
 root,out,_=prepared;before={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
 run(root,out,device='cpu',trusted=True,make_plots=False)
 h=sha(out/'comparison.csv');run(root,out,device='cpu',trusted=True,make_plots=False)
 assert h==sha(out/'comparison.csv');assert before=={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
 assert read_json(out/'run_status.json')['test_arrays_opened_by_this_runner']==0


def test_changed_input_cache_rejected(prepared):
 root,out,_=prepared;run(root,out,trusted=True,make_plots=False);r=read_json(root/SPLIT/VAL_NAME)[0];np.save(r['input_path'],np.zeros((32,)*3,np.float32))
 with pytest.raises(IOError):run(root,out,trusted=True,make_plots=False)


def test_external_no_synthetic_and_perfect(prepared,tmp_path):
 root,out,_=prepared;run(root,out,trusted=True,make_plots=False);c=read_json(out/'contract.local.json')
 records=[]
 for i,r in enumerate(c['records']):
  m=read_json(out/'records.local'/r['sample_id']/'record.json');pred=tmp_path/f'prediction_{i}.npy'
  with np.load(out/'records.local'/r['sample_id']/'arrays.npz') as a:np.save(pred,a['target_stored'])
  records.append(dict(sample_id=r['sample_id'],ct_sha256=m['ct_sha256'],prediction_path=str(pred),prediction_sha256=sha(pred)))
 manifest=dict(method='TEST_ONLY_NOT_A_REAL_MODEL',data_scope='synthetic',cohort_sha256=digest_object(c['records']),prediction_kind='absolute_dose',units='stored_target_units',training_provenance='software test arrays only',producer_source_sha256='a'*64,checkpoint_sha256='b'*64,records=records)
 p=tmp_path/'extra.json';write_json(p,manifest)
 with pytest.raises(ValueError):compare_external(out,p,tmp_path/'cmp_fail')
 manifest['data_scope']='real_validation';write_json(p,manifest,replace=True)
 compare_external(out,p,tmp_path/'cmp')
 df=__import__('pandas').read_csv(tmp_path/'cmp/comparison.csv');assert df.loc[df.method==manifest['method'],'rmse_stored_units'].iloc[0]==0


def test_plots_and_ct_only_external_export(prepared,tmp_path):
 from p10recover.external import produce_external
 root,out,_=prepared
 run(root,out,trusted=True,make_plots=True)
 assert len(list((out/'figures.local').glob('*.png')))==6
 calls=[]
 def predictor(ct):
  calls.append(ct.shape);return np.ones((32,)*3,np.float32)*.001
 man=produce_external(out,tmp_path/'ext_out','DemoModel',predictor,'software test only','c'*64,'d'*64)
 assert len(calls)==4
 compare_external(out,man,tmp_path/'ext_cmp')
 assert (tmp_path/'ext_cmp/comparison.csv').exists()
