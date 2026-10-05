"""Full file/optimizer workflow on explicitly synthetic parent and cache fixtures."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import torch
import pandas as pd
from cwfr import common as io
from refine.config import RefinementConfig
from refine.cache import register,prepare,FrozenData
from refine.train import train,fork
from refine.evaluation import finalize,evaluate
from refine.compare import compare
from fixture_study import make_parent,fast_identity_magnitude


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        io.seed_all(5)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.cache,cls.parent,cls.hjd,cls.direct=make_parent(cls.root)
        cls.work=cls.root/"refinement"
        cls.cfg=RefinementConfig(updates=4,monitor_every=2,save_every=2,learning_rate=.003)
        register(cls.cache,cls.parent,cls.work,cls.cfg)
        prepare(cls.cache,cls.parent,cls.work,"cpu",max_records=1)
        with patch("cwfr.model.ConditionalWFR.predict",fast_identity_magnitude):
            prepare(cls.cache,cls.parent,cls.work,"cpu",max_records=None)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_01_cache_counts_and_train_only_normalization(self):
        f=FrozenData(self.work)
        self.assertEqual(len(f.train_ids),192);self.assertEqual(len(f.monitor_ids),40)
        self.assertEqual(len(list((f.root/"records.local").iterdir())),232)
        self.assertEqual(io.read(f.root/"normalizers.json")["fitted_on"],"parent_train192_only_at_unchanged_initialization")
        self.assertFalse(io.read(f.root/"summary.json")["validation_transport_inference"])
        with self.assertRaises(PermissionError):f.load(f.monitor_ids[0])

    def test_02_cache_reuse_does_not_infer(self):
        with patch("refine.parent.ParentExperiment.load_model",side_effect=AssertionError("Unexpected model load")):
            prepare(self.cache,self.parent,self.work,"cpu")

    def test_03_resume_matches_continuous_cpu(self):
        full=self.work/"runs.local/continuous";resume=self.work/"runs.local/resumed"
        train(self.work,"composition","cpu",None,full)
        train(self.work,"composition","cpu",1,resume)
        train(self.work,"composition","cpu",None,resume)
        a=io.load_checkpoint(full,io.read(full/"last.json"));b=io.load_checkpoint(resume,io.read(resume/"last.json"))
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
        self.assertEqual(a["state"]["selected_step"],b["state"]["selected_step"])

    def test_04_explicit_fork_preserves_parent(self):
        source=self.work/"runs.local/fork_parent";child=self.work/"runs.local/fork_child"
        train(self.work,"profile","cpu",1,source);before=io.sha(source/"contract.json")
        fork(self.work,source,child,"cpu");train(self.work,"profile","cpu",None,child)
        self.assertEqual(before,io.sha(source/"contract.json"))
        self.assertEqual(io.read(source/"last.json")["step"],1);self.assertEqual(io.read(child/"last.json")["step"],4)

    def test_05_unapproved_environment_change_stops(self):
        run=self.work/"runs.local/env_stop";train(self.work,"composition","cpu",1,run)
        with patch("refine.train.io.environment",return_value={"test":"different_environment"}):
            with self.assertRaises(RuntimeError):train(self.work,"composition","cpu",None,run)

    def test_06_selection_pointer_recovery(self):
        run=self.work/"runs.local/selection_recovery";train(self.work,"composition","cpu",1,run)
        original=(run/"selected.json").read_bytes();(run/"selected.json").unlink()
        with self.assertRaises(ValueError):train(self.work,"composition","cpu",None,run)
        (run/"selected.json").write_bytes(original);train(self.work,"composition","cpu",None,run)

    def test_07_train_finalize_evaluate_and_compare_600(self):
        for arm in ("composition","profile"):train(self.work,arm,"cpu",None)
        a=self.work/"runs.local/composition_seed29";b=self.work/"runs.local/profile_seed29"
        self.assertEqual(io.read(a/"contract.json")["identity"]["schedule_sha256"],
                         io.read(b/"contract.json")["identity"]["schedule_sha256"])
        pair=finalize(self.work);self.assertTrue(pair["selection_locked_before_new_validation600"])
        out=evaluate(self.cache,self.parent,self.work,"cpu",max_records=11)
        self.assertFalse((out/"COMPLETE.json").exists())
        evaluate(self.cache,self.parent,self.work,"cpu",max_records=None)
        self.assertEqual(io.read(out/"COMPLETE.json")["n_records"],600)
        comp=compare(self.cache,self.parent,self.work,self.hjd,self.direct)
        df=pd.read_csv(comp/"comparison.csv")
        self.assertEqual(len(df),7);self.assertTrue((df.n_records==600).all())
        self.assertEqual(len(pd.read_csv(comp/"fixed_budget_last.csv")),2)
        self.assertTrue((comp/"figures/02_x_profile_percentage.png").exists())
        with patch("refine.evaluation.load_locked_heads",side_effect=AssertionError("Unexpected inference")):
            evaluate(self.cache,self.parent,self.work,"cpu")

    def test_08_frozen_record_tampering_is_rejected(self):
        f=FrozenData(self.work);sid=f.train_ids[0];p=f.root/"records.local"/sid/"arrays.npz"
        original=p.read_bytes();p.write_bytes(b"tampered")
        try:
            with self.assertRaises(ValueError):FrozenData(self.work).load(sid)
        finally:p.write_bytes(original)

    def test_09_changed_protocol_is_rejected(self):
        p=self.work/"protocol.json";raw=p.read_bytes();v=io.read(p);v["config"]["seed"]=77;io.write(p,v,replace=True)
        try:
            with self.assertRaises(ValueError):FrozenData(self.work)
        finally:p.write_bytes(raw)

    def test_10_register_does_not_overwrite(self):
        with self.assertRaises(RuntimeError):register(self.cache,self.parent,self.work,replace(self.cfg,learning_rate=.04))

if __name__=="__main__":unittest.main()
