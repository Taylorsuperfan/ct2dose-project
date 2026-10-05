"""Synthetic schema fixtures, never measurements from user medical data."""
from pathlib import Path
import hashlib
import numpy as np
from cwfr import common as io


def make_cache(root):
    root=Path(root);cache=root/'cache';old=root/'old';cache.mkdir();old.mkdir()
    tr=[];va=[]
    for split,cases,count in [('train',6,32),('validation',2,300)]:
        for ci in range(cases):
            for i in range(count):
                case=f'{split}_{ci}';sid=hashlib.sha256(f'{case}:{i}'.encode()).hexdigest()[:20]
                r={'case_id':case,'sample_id':sid,'record_id':f'record_{i}',
                   'input_path':f'/synthetic_only/{case}/ct_{i}.npy','output_path':f'/synthetic_only/{case}/dose_{i}.npy'}
                (tr if split=='train' else va).append(r)
    bc={'records':va,'pipeline':{'scope':'SYNTHETIC_TEST_FIXTURE_NOT_REAL'},'code_sha256':{}}
    io.write(old/'contract.local.json',bc)
    io.write(old/'run_status.json',{'status':'completed_validation_inference','n_records':600,'contract_sha256':io.digest(bc)})
    contract={'schema':'real_phase9g_cache_v1','train':tr,'validation':va,'baseline_run':str(old),
              'baseline_contract_sha256':io.digest(bc),'selection':{'upstream_exposure':'SYNTHETIC_TEST_ONLY'}}
    io.write(cache/'cache_contract.local.json',contract);rows=[]
    # Arrays are materialized only when a test explicitly requests a record.
    for split,group in [('train',tr),('validation',va)]:
        for r in group:
            rows.append({**r,'split':split,'record_meta_sha256':'pending','arrays_sha256':'pending',
                         **({'storage':'saved_legacy_validation'} if split=='validation' else {})})
    norm={'fitted_on':'selected_train6_only','residual_rms_model_scale':.1,'base_input_scale':1.,
          'water_reference':False,'synthetic_reference':False,
          'test_fixture_warning':'These are manufactured schema-compatible arrays, not medical observations.'}
    io.write(cache/'normalization.json',norm)
    return cache,old,contract,rows


def materialize_record(cache,old,contract,row):
    shape=(32,32,32);grid=np.indices(shape)[2]/31
    ct=(.1+.1*grid).astype(np.float32);base=np.full(shape,1.,np.float32)
    target=(base+.05*(2*grid-1)).astype(np.float32);factor=1000.
    root=cache if row['split']=='train' else old;folder=root/'records.local'/row['sample_id'];folder.mkdir(parents=True)
    arrays={'ct':ct,'base':base,'target':target} if row['split']=='train' else {
        'ct_normalized':ct,'target_stored':target/factor,'phase9g':base/factor,'phase10d_strict':(base+.7*(target-base))/factor}
    np.savez_compressed(folder/'arrays.npz',**arrays)
    meta={'sample_id':row['sample_id'],'case_id':row['case_id'],'split':row['split'],
          'contract_sha256':io.digest(contract) if row['split']=='train' else contract['baseline_contract_sha256'],
          'arrays_sha256':io.sha(folder/'arrays.npz'),'producer':'frozen_Phase9G_CT_only','dose_scale_factor':factor}
    io.write(folder/'record.json',meta);row['arrays_sha256']=meta['arrays_sha256'];row['record_meta_sha256']=io.sha(folder/'record.json')
    return folder


def seal(cache,contract,rows):
    io.write(cache/'index.local.json',rows,replace=True)
    io.write(cache/'READY.json',{'status':'READY_REAL_PHASE9G_RESIDUAL_CACHE','contract_sha256':io.digest(contract),
        'index_sha256':io.sha(cache/'index.local.json'),'norm_sha256':io.sha(cache/'normalization.json'),
        'n_train':192,'n_validation':600,'scope':'SYNTHETIC_SCHEMA_FIXTURE','final_test_arrays_opened':0},replace=True)
