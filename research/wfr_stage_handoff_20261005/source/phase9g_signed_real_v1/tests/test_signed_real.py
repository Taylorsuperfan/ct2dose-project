"""Isolated software tensors/files, NEVER user medical arrays or trained weights."""
from pathlib import Path
import copy, hashlib
import numpy as np
import pytest
import torch
from pg9learn.models import decompose,splat,sample_target,DenseResidualRF,HJDSpatialRF,loss,build
from pg9learn.cache import fit_norm,subset_by_case,select_real,Cache
from pg9learn.train import recipe,train,load_trained,evaluate_subset
from pg9learn.common import checkpoint_read,code_hashes
from p10recover.io import write_json,read_json,digest_object,sha,save_arrays
from p10recover.data import SPLIT,TRAIN_NAME,VAL_NAME,SUBSET_NAME

torch.set_num_threads(1)


def small_recipe():
    r=recipe('small');r.update(epochs=2,updates_per_epoch=2,batch_size=1,dense_width=2,hjd_width=4,features=2,hidden=8,
        fm_particles=12,particles=32,ode_steps=2,save_every_updates=1)
    return r


class MemoryCache:
    def __init__(self,path):
        self.path=Path(path);self.path.mkdir()
        self.rows=[];self.items=[]
        rng=np.random.default_rng(12)
        for j in range(8):
            ct=rng.uniform(0,1,(8,8,8)).astype('float32');base=np.ones_like(ct)*0.3
            target=base+0.02*(ct-0.5)
            row=dict(case_id='train'+str(j//2) if j<4 else 'val'+str(j//2),sample_id=f's{j}',record_id=f'r{j}',
                     split='train' if j<4 else 'validation',input_path=f'/ct{j}',output_path=f'/dose{j}')
            self.rows.append(row);self.items.append(dict(row=row,ct=ct,base=base,target=target,meta=dict(dose_scale_factor=1000.)))
        self.train=list(range(4));self.val=list(range(4,8));self.norm=fit_norm(self.items[:4]);self.identity='unit-test-only'
    select=Cache.select
    batch=Cache.batch


def test_hjd_decomposition_and_zero_branch():
    r=torch.tensor([-1.,0.,2.,0.]).view(1,1,1,2,2)
    parts,p,m,a=decompose(r)
    assert torch.equal(parts[:,:1]-parts[:,1:],r)
    assert (parts[:,0]*parts[:,1]).sum()==0
    assert torch.allclose(p.flatten(2).sum(-1),torch.ones(1,2))
    z=decompose(torch.zeros_like(r));assert not z[3].any() and torch.isfinite(z[1]).all()
    assert torch.isfinite(sample_target(z[1],z[3],8)).all()


def test_splat_mass_outside_and_gradient():
    x=torch.tensor([[[0.2,0.3,0.4],[1.4,0.3,0.2]]],requires_grad=True)
    p,o=splat(x,(4,4,4));assert torch.allclose(p.sum(),torch.tensor(1.));assert o.item()==.5
    (p.square().sum()).backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('method',['residual_rf','hjd_rf'])
def test_full_forward_backward_and_target_free_call(method):
    r=small_recipe();norm={'initial_mass_mean_scaled':[0.15,0.12]};m=build(method,r,norm)
    c=torch.rand(1,2,8,8,8);y=torch.randn(1,1,8,8,8)
    value,terms=loss(m,dict(condition=c,residual=y),method,r);value.backward()
    assert torch.isfinite(value)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    z=m.predict(c,r['ode_steps'],r['particles'],r['source_seed'])
    assert z['residual'].shape==y.shape and torch.isfinite(z['residual']).all()
    if method=='hjd_rf':assert (z['parts']>=0).all()


def test_dense_no_nonnegative_state_clamp():
    m=DenseResidualRF(2)
    with torch.no_grad():m.net.out_conv.bias.fill_(-1)
    pred=m.predict(torch.zeros(1,2,8,8,8),4)['residual']
    assert torch.allclose(pred,-torch.ones_like(pred))


def test_normalization_train_only_and_zero_task():
    a=dict(base=np.ones((2,2,2)),target=np.ones((2,2,2))*2)
    n=fit_norm([a]);assert n['residual_rms_model_scale']==1 and n['fitted_on']=='selected_train6_only'
    with pytest.raises(ValueError):fit_norm([dict(base=a['base'],target=a['base'])])


def test_identifier_selection_deterministic_no_target():
    rows=[dict(case_id=f'c{i//5}',input_path=f'ct{i}',output_path=f'dose{i}') for i in range(10)]
    assert subset_by_case(rows,2,17)==subset_by_case(list(reversed(rows)),2,17)
    assert len(subset_by_case(rows,2,17))==4


@pytest.mark.parametrize('method',['residual_rf','hjd_rf'])
def test_training_interrupted_resume_matches_continuous(tmp_path,method):
    cache=MemoryCache(tmp_path/'cache');r=small_recipe()
    run1=train(cache,method,tmp_path/'one',r,'cpu')
    run2=train(cache,method,tmp_path/'two',r,'cpu',max_updates_this_call=1)
    assert read_json(run2/'run_status.json')['status']=='paused_at_saved_update'
    train(cache,method,run2,r,'cpu')
    c1=checkpoint_read(run1,read_json(run1/'last.json'));c2=checkpoint_read(run2,read_json(run2/'last.json'))
    assert c1['history']==c2['history']
    for k in c1['model']:assert torch.equal(c1['model'][k],c2['model'][k])
    before=sha(run2/'last.json');train(cache,method,run2,r,'cpu');assert before==sha(run2/'last.json')


def test_reject_recipe_changes_and_tamper(tmp_path):
    c=MemoryCache(tmp_path/'c');r=small_recipe();out=train(c,'residual_rf',tmp_path/'run',r)
    rr=dict(r,lr=r['lr']*2)
    with pytest.raises(ValueError):train(c,'residual_rf',out,rr)
    p=read_json(out/'best.json');(out/p['path']).write_bytes(b'bad')
    with pytest.raises(ValueError):load_trained(out,c)


def test_match_cohort_and_no_test_option(tmp_path):
    root=tmp_path/'real';sd=root/SPLIT;sd.mkdir(parents=True);old=tmp_path/'old';old.mkdir()
    def record(case,i):return dict(case_id=case,input_path=str(root/'data/raw'/case/f'ct{i}.npy'),output_path=str(root/'data/raw'/case/f'dose{i}.npy'))
    tr=[record(f't{k}',i) for k in range(6) for i in range(2)]
    va=[record(f'v{k}',i) for k in range(2) for i in range(300)]
    for name,rows in [(TRAIN_NAME,tr),(VAL_NAME,va),(SUBSET_NAME,va)]:write_json(sd/name,rows)
    bc={'records':va};write_json(old/'contract.local.json',bc);write_json(old/'run_status.json',dict(status='completed_validation_inference',n_records=600,contract_sha256=digest_object(bc)))
    a,b,m,c=select_real(root,old,n_per_case=2);assert len(a)==12 and len(b)==600 and not m['historical_test_json_opened']
    va.reverse();write_json(sd/SUBSET_NAME,va,replace=True)
    with pytest.raises(ValueError):select_real(root,old,n_per_case=2)


def test_final_comparison_end_to_end(tmp_path):
    from pg9learn.compare import evaluate_new,report
    from p10recover.metrics import score
    cache=MemoryCache(tmp_path/'cache');old=tmp_path/'old';old.mkdir()
    records=[cache.rows[i] for i in cache.val];bc=dict(records=records)
    write_json(old/'contract.local.json',bc)
    cache.contract=dict(baseline_run=str(old),baseline_contract_sha256=digest_object(bc))
    for i in cache.val:
        a=cache.items[i];row=a['row'];folder=old/'records.local'/row['sample_id'];folder.mkdir(parents=True)
        factor=1000.;arrays=dict(target_stored=a['target']/factor,phase9g=a['base']/factor,phase10d_strict=(a['base']+.2*(a['target']-a['base']))/factor)
        save_arrays(folder/'arrays.npz',arrays)
        ms=[]
        for name in ['phase9g','phase10d_strict']:
            ms.append(dict(method=name,case_id=row['case_id'],record_id=row['record_id'],sample_id=row['sample_id'],**score(arrays[name]*factor,a['target'],factor)))
        write_json(folder/'record.json',dict(arrays_sha256=sha(folder/'arrays.npz'),metrics=ms))
        a['meta']['baseline_record_sha256']=sha(folder/'record.json')
    # Small software fixture selects at most two monitor records per case.
    orig_select=cache.select
    cache.select=lambda indices,n,seed:orig_select(indices,min(n,2),seed)
    r=small_recipe();r['stage']='pilot'
    run=train(cache,'residual_rf',tmp_path/'run',r)
    ev=evaluate_new(cache,run,tmp_path/'eval')
    rpt=report(cache,[ev],tmp_path/'report')
    assert read_json(rpt/'REPORT_READY.json')['new_method_comparison_completed']
    assert (rpt/'comparison.csv').exists() and len(list((rpt/'figures.local').glob('*.png')))==6
    assert read_json(ev/'run_status.json')['n_records']==4


def test_32cube_methods_shapes():
    c=torch.zeros(1,2,32,32,32);r=small_recipe()
    for name in ['residual_rf','hjd_rf']:
        m=build(name,r,{'initial_mass_mean_scaled':[.1,.1]})
        with torch.no_grad():p=m.predict(c,1,32)['residual']
        assert p.shape==(1,1,32,32,32) and torch.isfinite(p).all()


def test_prepare_reuses_saved_validation_and_resumes_cache(tmp_path,monkeypatch):
    import pg9learn.cache as mod
    from p10recover.io import code_hashes as old_hashes
    root=tmp_path/'real';rawroot=root/'data/raw';rawroot.mkdir(parents=True)
    old=tmp_path/'legacy';old.mkdir();rows=[]
    for k in range(4):
        ct=np.full((32,32,32),.2+k*.05,np.float32);gt=np.full_like(ct,.3+k*.01)
        ip=rawroot/f'ct{k}.npy';op=rawroot/f'dose{k}.npy';np.save(ip,ct);np.save(op,gt)
        rows.append(mod.with_ids(dict(case_id=('t' if k<2 else 'v')+str(k),input_path=str(ip),output_path=str(op))))
    tr,va=rows[:2],rows[2:];pipeline_contract={'unit_test_only':True}
    bc=dict(pipeline=pipeline_contract,code_sha256=old_hashes(),records=va)
    ch=digest_object(bc)
    for row in va:
        folder=old/'records.local'/row['sample_id'];folder.mkdir(parents=True)
        ct=np.load(row['input_path']);y=np.load(row['output_path']);base=np.ones_like(y)*.25
        save_arrays(folder/'arrays.npz',dict(ct_normalized=ct,target_stored=y,phase9g=base,phase10d_strict=base))
        write_json(folder/'record.json',dict(sample_id=row['sample_id'],contract_sha256=ch,arrays_sha256=sha(folder/'arrays.npz'),
            dose_scale_factor=1.,ct_sha256=sha(row['input_path']),target_sha256=sha(row['output_path'])))
    monkeypatch.setattr(mod,'select_real',lambda *args,**kw:(tr,va,{'unit_test_only':True},bc))
    class P:
        contract=pipeline_contract
        device=torch.device('cpu')
        calls=0
        def __init__(self):self.legacy=self
        def normalize_ct(self,a):return a
        def predict_phase9g(self,x,steps):
            self.calls+=1
            return {'phase9g_pred':x*.1+.25}
        def scale_target_for_legacy_evaluation(self,y):return y,1.
    p=P();out=mod.prepare(root,old,tmp_path/'cache',p,n_per_case=1)
    assert p.calls==2
    mod.prepare(root,old,out,p,n_per_case=1);assert p.calls==2
    c=mod.Cache(out);assert len(c.train)==2 and len(c.val)==2
    assert c.norm['residual_rms_model_scale']==pytest.approx(np.sqrt(((.3-.27)**2+(.31-.275)**2)/2),rel=2e-6)
    corrupt=out/'records.local'/tr[0]['sample_id']/'arrays.npz';corrupt.write_bytes(b'bad')
    with pytest.raises(IOError):mod.Cache(out)
