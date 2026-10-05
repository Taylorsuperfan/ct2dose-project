"""End-to-end comparison tests with explicit synthetic cache-schema fixtures."""
import tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from cwfr import common as io
from cwfr.data import RealCache
from cwfr.evaluate import compare, _previous_eval
from fixtures import make_cache,materialize_record,seal

class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cache,old,contract,rows=make_cache(cls.root)
        for row in rows:
            if row['split']=='validation':materialize_record(cache,old,contract,row)
        seal(cache,contract,rows);cls.cache=cache;data=RealCache(cache)
        cls.new=cls.root/'new';cls.hjd=cls.root/'hjd';cls.direct=cls.root/'direct'
        ids=[r['sample_id'] for r in data.val]
        ni={'cache_identity':data.identity,'validation_ids':ids,'ode_steps':2,'solver':'SYNTHETIC_TEST_ONLY',
            'train_contract':'synthetic_only','trainable_parameters':0,'optimizer_updates':0,'batch_size':1,
            'pilot_training_records':0,'monitor_records':0}
        io.write(cls.new/'contract.json',{'identity':ni})
        ec={}
        for p,method in [(cls.hjd,'new_hjd_rf'),(cls.direct,'new_residual_rf')]:
            c={'schema':'real_phase9g_new_evaluation_v1','method':method,'cache_identity':data.identity,'val_ids':ids}
            io.write(p/'contract.local.json',c);io.write(p/'run_status.json',{'status':'completed_new_validation','contract_sha256':io.digest(c)})
            ec[p]=c
        for r in data.val:
            sid=r['sample_id'];a=data.load(sid,allow_validation=True);target=a['original']['target_stored'];base=a['original']['phase9g']
            for p,fraction in [(cls.new,.8),(cls.hjd,.4),(cls.direct,.5)]:
                folder=p/'records.local'/sid;folder.mkdir(parents=True)
                pred=(base+fraction*(target-base)).astype(np.float32)
                arrays={'prediction_stored':pred}
                if p==cls.new:arrays['hard_prediction_stored']=pred
                np.savez_compressed(folder/'arrays.npz',**arrays)
                m={'sample_id':sid,'case_id':r['case_id'],'arrays_sha256':io.sha(folder/'arrays.npz')}
                if p==cls.new:m['source_arrays_sha256']=r['arrays_sha256']
                else:m['contract_sha256']=io.digest(ec[p])
                io.write(folder/'record.json',m)
        io.finish(cls.new,'validation_inference_completed')
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_complete_comparison_six_methods_600(self):
        out=self.root/'comparison'
        compare(RealCache(self.cache),self.new,self.hjd,self.direct,out)
        s=pd.read_csv(out/'comparison.csv');self.assertEqual(len(s),6)
        self.assertTrue((s.n_records==600).all());self.assertTrue((s.n_cases==2).all())
        self.assertTrue((out/'figures/01_whole_cube_rmse.png').is_file())
        r=pd.read_csv(out/'paired_records.local.csv');self.assertEqual(len(r),3600)
    def test_previous_method_mismatch_stops(self):
        with self.assertRaises(ValueError):_previous_eval(self.hjd,RealCache(self.cache),'new_residual_rf')
    def test_scope_does_not_use_test_arrays(self):
        c=RealCache(self.cache);_previous_eval(self.hjd,c,'new_hjd_rf');self.assertEqual(c.access_summary()['test_arrays_opened'],0)
