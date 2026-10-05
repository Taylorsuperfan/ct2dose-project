"""Manufactured schema fixtures. Completed parent markers here are fixtures, not training claims."""
from pathlib import Path
import numpy as np
import torch
from cwfr import common as io
from cwfr.config import Config
from cwfr.data import RealCache
from cwfr.model import ConditionalWFR
from legacy_fixture import make_cache, materialize_record, seal


def make_parent(root):
    root = Path(root)
    cache, old, contract, rows = make_cache(root)
    for row in rows:
        materialize_record(cache, old, contract, row)
    seal(cache, contract, rows)
    data = RealCache(cache)
    cfg = Config(stage="pilot", channels=2, point_hidden=4, ode_steps=1, query_chunk=8192)
    model = ConditionalWFR(2, 4)
    parent = root / "parent_work"; run = parent / "runs.local/pilot_seed17"
    identity = {"config": cfg.to_dict(), "cache_identity": data.identity,
                "source_code": io.source_identity(), "selection": data.selection("pilot"),
                "scope": "SYNTHETIC_PARENT_FIXTURE_NOT_A_MEDICAL_EXPERIMENT"}
    c = {"identity": identity, "environment": io.environment("cpu"), "parent": None}
    io.write(run / "contract.json", c)
    pointer = io.save_checkpoint(run, {"model": model.state_dict(), "contract_sha256": io.digest(c), "step":384})
    io.write(run / "best.json", pointer)
    io.finish(run, "conditional_training_completed", contract_sha256=io.digest(c), updates=384)
    evaluation = parent / "evaluations.local/pilot_seed17_val600"
    ec = {"identity": {"source_code": io.source_identity(), "cache_identity": data.identity,
          "train_contract": io.digest(c), "checkpoint": pointer,
          "validation_ids": [r["sample_id"] for r in data.val]}, "environment": io.environment("cpu")}
    io.write(evaluation / "contract.json", ec)
    hjd = root / "old_hjd"; direct = root / "old_direct"; previous = {}
    for folder, method in ((hjd,"new_hjd_rf"),(direct,"new_residual_rf")):
        mc={"schema":"real_phase9g_new_evaluation_v1", "method":method,
            "cache_identity":data.identity,"val_ids":[r["sample_id"] for r in data.val]}
        io.write(folder/"contract.local.json", mc)
        io.write(folder/"run_status.json", {"status":"completed_new_validation", "contract_sha256":io.digest(mc)})
        previous[folder]=mc
    for row in data.val:
        sid=row["sample_id"]; item=data.load(sid,allow_validation=True)
        base=(item["base"].astype(float)/item["factor"]).astype(np.float32)
        folder=evaluation/"records.local"/sid;folder.mkdir(parents=True)
        np.savez_compressed(folder/"arrays.npz",prediction_stored=base,raw_prediction_stored=base,
                            magnitude_normalized=np.ones_like(base),soft_sign=np.zeros_like(base))
        io.write(folder/"record.json", {"sample_id":sid,"case_id":row["case_id"],
            "contract_sha256":io.digest(ec),"source_arrays_sha256":row["arrays_sha256"],
            "source_record_sha256":row["record_meta_sha256"],"arrays_sha256":io.sha(folder/"arrays.npz")})
        for oldfolder, amount in ((hjd,.35),(direct,.45)):
            of=oldfolder/"records.local"/sid;of.mkdir(parents=True)
            pred=base+amount*(item["original"]["target_stored"]-base)
            np.savez_compressed(of/"arrays.npz",prediction_stored=pred.astype(np.float32))
            io.write(of/"record.json", {"sample_id":sid,"case_id":row["case_id"],
                "contract_sha256":io.digest(previous[oldfolder]),"arrays_sha256":io.sha(of/"arrays.npz")})
    io.finish(evaluation,"validation_inference_completed",contract_sha256=io.digest(ec),n_records=600)
    return cache,parent,hjd,direct


def fast_identity_magnitude(self, condition, steps=16, query_chunk=8192):
    """Test-only substitute for repeated particle integration; the real predictor is tested separately."""
    shape=tuple(condition.shape[2:])
    with torch.no_grad(): sign=torch.tanh(self.sign_logits(condition)/2)[0,0].cpu().numpy().astype(float)
    return {"magnitude":np.ones(shape),"sign":sign}
