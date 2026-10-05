import unittest,tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from cwfr import common as io
from cwfr.config import Config
from cwfr.data import RealCache
from cwfr.plans import prepare
from cwfr.train import train,load_model,fork_run
from fixtures import make_cache,materialize_record,seal

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cache,self.old,self.contract,self.rows=make_cache(self.root)
        seal(self.cache,self.contract,self.rows)
    def tearDown(self):self.tmp.cleanup()
    def test_metadata_only_constructor(self):
        c=RealCache(self.cache);self.assertEqual(c.access_summary()['training_arrays_opened'],0)
    def test_val_requires_explicit_access(self):
        c=RealCache(self.cache)
        with self.assertRaises(PermissionError):c.load(c.val[0]['sample_id'])
    def test_one_selection_value_independent(self):
        c=RealCache(self.cache);s=c.selection('one');self.assertEqual(len(s['train_ids']),1)
        self.assertEqual(s['train_ids'],s['monitor_ids'])
    def test_six_selection(self):
        s=RealCache(self.cache).selection('six');self.assertEqual(len(s['train_ids']),6)
    def test_pilot_and_monitor_count(self):
        c=RealCache(self.cache);s=c.selection('pilot');self.assertEqual(len(s['train_ids']),192);self.assertEqual(len(s['monitor_ids']),40)
    def test_reads_selected_train_only(self):
        c=RealCache(self.cache);sid=c.selection('one')['train_ids'][0]
        row=next(r for r in self.rows if r['sample_id']==sid)
        materialize_record(self.cache,self.old,self.contract,row);seal(self.cache,self.contract,self.rows)
        c=RealCache(self.cache);a=c.load(sid);self.assertEqual(a['condition'].shape,(2,32,32,32))
        self.assertEqual(c.access_summary()['validation_arrays_opened'],0)
    def test_array_tampering(self):
        row=self.rows[0];folder=materialize_record(self.cache,self.old,self.contract,row);seal(self.cache,self.contract,self.rows)
        with (folder/'arrays.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):RealCache(self.cache).load(row['sample_id'])
    def test_normalization_tampering(self):
        n=io.read(self.cache/'normalization.json');n['base_input_scale']=2;io.write(self.cache/'normalization.json',n,replace=True)
        with self.assertRaises(ValueError):RealCache(self.cache)
    def test_validation_case_overlap(self):
        self.rows[-1]['case_id']=self.rows[0]['case_id'];seal(self.cache,self.contract,self.rows)
        with self.assertRaises(ValueError):RealCache(self.cache)
    def test_unknown_split(self):
        self.rows[-1]['split']='test';seal(self.cache,self.contract,self.rows)
        with self.assertRaises(ValueError):RealCache(self.cache)
    def test_path_escape(self):
        with self.assertRaises(ValueError):io.safe(self.root,'../outside')
    def test_no_overwrite_contract(self):
        p=self.root/'immutable.json';io.write(p,{'x':1})
        with self.assertRaises(RuntimeError):io.write(p,{'x':2})
    def test_output_separation(self):
        with self.assertRaises(ValueError):io.separate(self.cache/'outputs',self.cache)

class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name);cls.root=root
        cache,old,contract,rows=make_cache(root);seal(cache,contract,rows);c=RealCache(cache)
        ids=c.selection('six')['train_ids']
        for row in rows:
            if row['sample_id'] in ids:materialize_record(cache,old,contract,row)
        seal(cache,contract,rows);cls.cache=cache
        cls.cfg=Config(stage='one',channels=2,point_hidden=8,updates=4,batch_size=1,
             source_points_per_bank=8,target_draws_per_bank=8,banks_per_record=1,
             pairs_per_record=16,coupling_iterations=20,ode_steps=2,query_chunk=32768,monitor_every=2,save_every=1)
        cls.plans=root/'plans';prepare(RealCache(cache),cls.plans,cls.cfg)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_plan_preparation_does_not_read_validation(self):
        s=io.read(self.plans/'summary.json');self.assertEqual(s['validation_arrays_opened'],0);self.assertEqual(s['test_arrays_opened'],0)
    def test_resume_identical_cpu_state(self):
        a=self.root/'uninterrupted';b=self.root/'resumed'
        train(RealCache(self.cache),self.plans,a,self.cfg,'cpu',None)
        train(RealCache(self.cache),self.plans,b,self.cfg,'cpu',1)
        train(RealCache(self.cache),self.plans,b,self.cfg,'cpu',None)
        ca=io.load_checkpoint(a,io.read(a/'last.json'));cb=io.load_checkpoint(b,io.read(b/'last.json'))
        self.assertEqual(ca['state']['step'],4)
        for key in ca['model']:torch.testing.assert_close(ca['model'][key],cb['model'][key],rtol=0,atol=0)
        self.assertEqual(ca['state']['history'],cb['state']['history'])
    def test_new_inference_after_completed_load(self):
        out=self.root/'completed';train(RealCache(self.cache),self.plans,out,self.cfg,'cpu',None)
        cache=RealCache(self.cache);m,cfg,c,p=load_model(out,cache,'cpu')
        a=cache.load(cache.selection('one')['train_ids'][0]);z=m.predict(torch.tensor(a['condition'][None]),2,32768)
        self.assertTrue(np.isfinite(z['residual_normalized']).all());self.assertEqual(z['residual_normalized'].shape,(32,32,32))
    def test_pause_status_and_existing_progress(self):
        out=self.root/'paused';train(RealCache(self.cache),self.plans,out,self.cfg,'cpu',1)
        self.assertEqual(io.read(out/'last.json')['step'],1);self.assertFalse((out/'COMPLETE.json').exists())
    def test_explicit_fork_preserves_parent(self):
        parent=self.root/'parent';child=self.root/'child';cache=RealCache(self.cache)
        train(cache,self.plans,parent,self.cfg,'cpu',1);oldhash=io.sha(parent/'last.json')
        fork_run(cache,self.plans,parent,child,'cpu',updates=6)
        self.assertEqual(io.sha(parent/'last.json'),oldhash);self.assertEqual(io.read(child/'last.json')['step'],1)
        train(RealCache(self.cache),self.plans,child,replace(self.cfg,updates=6),'cpu',None)
        self.assertEqual(io.read(child/'training_summary.json')['updates'],6)
    def test_scientific_change_rejected(self):
        out=self.root/'change';train(RealCache(self.cache),self.plans,out,self.cfg,'cpu',1)
        with self.assertRaises(ValueError):train(RealCache(self.cache),self.plans,out,replace(self.cfg,learning_rate=.01),'cpu',1)
