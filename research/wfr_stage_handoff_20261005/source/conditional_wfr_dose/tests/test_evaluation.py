"""Two-record synthetic evaluation exercises real inference and saved-result reuse."""
import unittest,tempfile
from pathlib import Path
import torch
from cwfr import common as io
from cwfr.data import RealCache
from cwfr.config import Config
from cwfr.model import ConditionalWFR
from cwfr.evaluate import evaluate
from fixtures import make_cache,materialize_record,seal

class EvaluationTests(unittest.TestCase):
    def test_inference_and_reuse_on_two_synthetic_validation_records(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache,old,c,rows=make_cache(root)
            val=[r for r in rows if r['split']=='validation'];chosen=[val[0],val[300]]
            for row in chosen:materialize_record(cache,old,c,row)
            seal(cache,c,rows);data=RealCache(cache);data.val=chosen
            cfg=Config(channels=2,point_hidden=8,updates=4,ode_steps=2,query_chunk=32768)
            model=ConditionalWFR(2,8);run=root/'trained';run.mkdir()
            contract={'identity':{'config':cfg.to_dict(),'cache_identity':data.identity,'source_code':io.source_identity(),
                       'selection':{'train_ids':['synthetic_training_record'],'monitor_ids':[r['sample_id'] for r in chosen]}},
                      'scope':'UNIT_TEST_SYNTHETIC_CHECKPOINT_NOT_TRAINED_MEDICAL_MODEL'}
            io.write(run/'contract.json',contract)
            pointer=io.save_checkpoint(run,{'model':model.state_dict(),'contract_sha256':io.digest(contract),'step':4})
            io.write(run/'best.json',pointer);io.finish(run,'conditional_training_completed')
            out=root/'evaluated';evaluate(data,run,out,'cpu')
            complete=io.verify_finish(out,'validation_inference_completed');self.assertEqual(complete['n_records'],2)
            before=io.sha(out/'FILES.json');evaluate(data,run,out,'cpu');self.assertEqual(io.sha(out/'FILES.json'),before)
