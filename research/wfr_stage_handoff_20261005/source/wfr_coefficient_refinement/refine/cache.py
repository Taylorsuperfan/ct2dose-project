"""Cache the frozen magnitude once. Do not recompute validation transport."""
from pathlib import Path
import copy
import time
import numpy as np
import torch
from cwfr import common as io
from cwfr.data import RealCache
from .config import RefinementConfig
from .parent import ParentExperiment
from .integrity import (require, source_identity, assert_outputs_separate, save_npz,
                        save_weights, state_digest, Receipt)
from .model import CoefficientHead
from .losses import components


def context(cache_path, parent_work, work, cfg):
    data = RealCache(cache_path)
    parent = ParentExperiment(parent_work, data)
    assert_outputs_separate(work, parent.work, data)
    identity = {
        "schema": "frozen_wfr_coefficient_pair_1",
        "parent": parent.identity,
        "refinement_source": source_identity(),
        "config": cfg.to_dict(),
        "arms": ["composition", "profile"],
        "comparison": "The two arms differ only in the additional profile loss.",
        "scope": "REAL_CACHE_DEVELOPMENT_NOT_WATER_NOT_INDEPENDENT_TEST",
        "selection_rule": "minimize_x_mean_pct_with_per_case_and_aggregate_guards",
        "fallback": "retain_parent_when_no_eligible_improvement",
        "amplitude_frozen": True,
        "monitor_targets_used_for_training": False,
    }
    return data, parent, identity


def register(cache_path, parent_work, work, cfg):
    cfg.validate(); work = Path(work).resolve()
    data, parent, identity = context(cache_path, parent_work, work, cfg)
    work.mkdir(parents=True, exist_ok=True)
    io.write(work / "protocol.json", identity)
    print("Protocol recorded:", work / "protocol.json", flush=True)
    print("Parent selected step:", parent.pointer["step"], "Additional updates per arm:", cfg.updates)
    print("No medical arrays were opened.")
    return identity


def open_protocol(cache_path, parent_work, work):
    work = Path(work).resolve()
    protocol = io.read(work / "protocol.json")
    cfg = RefinementConfig(**protocol["config"]).validate()
    data, parent, expected = context(cache_path, parent_work, work, cfg)
    require(protocol == expected, "The experiment protocol has changed. Use a separate experiment directory.")
    return data, parent, protocol, cfg


def prepare(cache_path, parent_work, work, device="cpu", max_records=32, allow_environment_change=False):
    data, parent, protocol, cfg = open_protocol(cache_path, parent_work, work)
    work = Path(work).resolve(); out = work / "frozen.local"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    io.seed_all(cfg.seed)
    base_identity = {"protocol_sha256": io.digest(protocol), "parent": parent.identity}
    out.mkdir(parents=True, exist_ok=True)
    with io.single_writer(out):
        if (out / "COMPLETE.json").exists():
            c = io.verify_finish(out, "frozen_magnitude_cache_completed")
            require(c["protocol_sha256"] == io.digest(protocol), "Frozen cache protocol differs.")
            print("Frozen cache verified; no transport inference.", flush=True)
            return out
        io.write(out / "contract.json", base_identity)
        env = io.environment(device)
        envfile = out / "first_environment.json"
        if envfile.exists() and io.read(envfile) != env:
            if not allow_environment_change:
                raise RuntimeError("Cache environment changed. Reuse the old device, or explicitly pass --allow-cache-environment-change. Completed records will be preserved.")
            io.write(out / "environment_events.local" / (io.stamp() + ".json"),
                     {"environment": env, "reason": "Explicit record-wise cache continuation; not bitwise equivalence."})
        else:
            io.write(envfile, env)

        model = parent.load_model(device)
        head = CoefficientHead.from_parent(model).cpu()
        initial = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        init_path = out / "initial_coefficient.pt"
        expected_digest = state_digest(initial)
        if init_path.exists():
            require(state_digest(torch.load(init_path, weights_only=True, map_location="cpu")) == expected_digest,
                    "Initial coefficient weights changed.")
        else:
            save_weights(init_path, initial)
        io.write(out / "initial_coefficient.json", {
            "sha256": io.sha(init_path), "tensor_sha256": expected_digest,
            "channels": parent.cfg.channels, "trainable_parameters": sum(p.numel() for p in head.parameters()),
            "parent_total_parameters": sum(p.numel() for p in model.parameters()),
            "parent_checkpoint": parent.pointer,
        })
        ids = parent.selection["train_ids"] + parent.selection["monitor_ids"]
        new_count = 0
        for j, sid in enumerate(ids, 1):
            row = data.by_id[sid]
            folder = out / "records.local" / sid
            meta_path = folder / "record.json"; array_path = folder / "arrays.npz"
            if meta_path.exists():
                m = io.read(meta_path)
                require(m["protocol_sha256"] == io.digest(protocol) and m["source_arrays_sha256"] == row["arrays_sha256"]
                        and m["arrays_sha256"] == io.sha(array_path), "Cached record changed.")
                continue
            if array_path.exists():
                # An array saved without its receipt is not silently trusted or overwritten.
                raise RuntimeError("An unsealed array archive exists: " + str(array_path) + ". Preserve it and inspect the interrupted write.")
            item = data.load(sid, allow_validation=(row["split"] == "validation"))
            begin = time.monotonic()
            if row["split"] == "train":
                with torch.no_grad():
                    condition = torch.tensor(item["condition"][None], device=device)
                    z = model.predict(condition, parent.cfg.ode_steps, parent.cfg.query_chunk)
                    logits = model.sign_logits(condition)[0, 0].cpu().numpy().astype(np.float32)
                a = z["magnitude"].astype(np.float32)
                s0 = np.tanh(logits.astype(np.float64)/2).astype(np.float32)
                require(np.allclose(s0, z["sign"], rtol=1e-5, atol=2e-6), "Sign reconstruction differs.")
                y = (item["target"].astype(float) / item["factor"]).astype(np.float32)
                raw = ((item["base"].astype(float) + data.norm["residual_rms_model_scale"] * a.astype(float) * s0)
                       / item["factor"]).astype(np.float32)
                previous = np.maximum(raw, 0).astype(np.float32)
                provenance = {"transport_recomputed": True, "new_frozen_baseline_prediction": False}
            else:
                saved, link = parent.prediction(sid)
                a = saved["magnitude_normalized"].astype(np.float32)
                s0 = saved["soft_sign"].astype(np.float32)
                logits = np.zeros_like(s0)  # Not used in loss scaling; validation is not training data.
                y = item["original"]["target_stored"].astype(np.float32)
                raw = saved["raw_prediction_stored"].astype(np.float32)
                previous = saved["prediction_stored"].astype(np.float32)
                provenance = {"transport_recomputed": False, "saved_parent_prediction": link}
            arrays = {"condition": item["condition"], "base": item["base"], "target": item["target"],
                      "target_stored": y, "residual": item["residual"], "magnitude": a,
                      "initial_logits": logits, "parent_sign": s0, "parent_raw": raw,
                      "parent_prediction": previous, "factor": np.asarray(item["factor"], dtype=np.float64)}
            require(np.isfinite(a).all() and (a >= 0).all(), "Invalid cached magnitude.")
            folder.mkdir(parents=True, exist_ok=True)
            save_npz(array_path, arrays)
            io.write(meta_path, {"sample_id": sid, "case_id": row["case_id"], "split": row["split"],
                "protocol_sha256": io.digest(protocol), "arrays_sha256": io.sha(array_path),
                "source_arrays_sha256": row["arrays_sha256"], "source_record_sha256": row["record_meta_sha256"],
                "environment": env, "seconds": time.monotonic()-begin, **provenance})
            new_count += 1
            print(f"FROZEN CACHE {j}/{len(ids)} saved; split={row['split']}", flush=True)
            if max_records is not None and new_count >= max_records and j < len(ids):
                print("Cache paused safely. Repeat prepare to continue.", flush=True)
                return out

        del model
        # Fit loss units once using the unchanged initialization and TRAIN records only.
        values = {k: [] for k in ("sign_bce", "reconstruction_mse", "profile_relative", "profile_absolute_mse")}
        sign_num, sign_den = 0., 0.
        for sid in parent.selection["train_ids"]:
            with np.load(out / "records.local" / sid / "arrays.npz", allow_pickle=False) as z:
                tensors = {k: torch.from_numpy(z[k].copy())[None] for k in
                           ("initial_logits", "magnitude", "residual", "base", "target")}
            with torch.no_grad():
                loss = components(tensors["initial_logits"], tensors["magnitude"], tensors["residual"],
                                  tensors["base"], tensors["target"], data.norm["residual_rms_model_scale"], cfg)
            sign_num += float(loss["sign_numerator"].sum()); sign_den += float(loss["sign_denominator"].sum())
            for key in values:
                values[key].append(float(loss[key].mean()))
        raw_scales = {k: float(np.mean(v)) for k, v in values.items()}
        raw_scales["sign_bce"] = sign_num / max(sign_den, 1e-12)
        scales = {k: max(v, cfg.normalization_floor) for k, v in raw_scales.items()}
        io.write(out / "normalizers.json", {
            "fitted_on": "parent_train192_only_at_unchanged_initialization",
            "raw_values": raw_scales, "denominators": scales,
            "floor": cfg.normalization_floor, "train_ids_sha256": io.digest(parent.selection["train_ids"]),
        })
        io.write(out / "summary.json", {"n_train": 192, "n_monitor": 40,
            "train_magnitude_inference": "One inference per training record, from the frozen parent checkpoint.",
            "validation_transport_inference": False, "test_arrays_opened": 0,
            "parent_model_changed": False, "residual_scale": data.norm["residual_rms_model_scale"],
            "initial_coefficient_tensor_sha256": expected_digest,
            "used_input_hashes": {**parent.train_receipt.used, **parent.eval_receipt.used, **data.snapshot}})
        parent.recheck()
        io.finish(out, "frozen_magnitude_cache_completed", protocol_sha256=io.digest(protocol), n_records=len(ids))
        print("FROZEN CACHE COMPLETE:", out, flush=True)
        return out


class FrozenData:
    def __init__(self, work):
        self.work = Path(work).resolve(); self.root = self.work / "frozen.local"
        self.protocol = io.read(self.work / "protocol.json")
        self.cfg = RefinementConfig(**self.protocol["config"]).validate()
        require(self.protocol["refinement_source"] == source_identity(), "Refinement source changed.")
        require(self.protocol["parent"]["source_code"] == io.source_identity(), "Parent cwfr source changed.")
        self.receipt = Receipt(self.root, "frozen_magnitude_cache_completed")
        require(self.receipt.complete["protocol_sha256"] == io.digest(self.protocol), "Frozen/protocol linkage differs.")
        self.selection = self.protocol["parent"]["selection"]
        self.train_ids = self.selection["train_ids"]; self.monitor_ids = self.selection["monitor_ids"]
        self.scales = self.receipt.json("normalizers.json")["denominators"]
        self.summary = self.receipt.json("summary.json")
        self.scale = self.summary["residual_scale"]
        self.initial_meta = self.receipt.json("initial_coefficient.json")
        path = self.receipt.file("initial_coefficient.pt")
        require(io.sha(path) == self.initial_meta["sha256"], "Initial coefficient archive differs.")
        self.initial_state = torch.load(path, weights_only=True, map_location="cpu")
        require(state_digest(self.initial_state) == self.initial_meta["tensor_sha256"], "Initial coefficient tensors differ.")
        self.opened = {"train": set(), "validation": set()}; self.memory = {}

    def load(self, sid, allow_monitor=False):
        is_train = sid in self.train_ids
        if not is_train and (not allow_monitor or sid not in self.monitor_ids):
            raise PermissionError("Only training records and explicitly authorized monitor40 are available.")
        if sid in self.memory:
            return self.memory[sid]
        prefix = "records.local/" + sid + "/"
        meta = self.receipt.json(prefix + "record.json")
        path = self.receipt.file(prefix + "arrays.npz")
        require(meta["sample_id"] == sid and meta["protocol_sha256"] == io.digest(self.protocol), "Frozen record linkage differs.")
        require(io.sha(path) == meta["arrays_sha256"], "Frozen record bytes differ.")
        expected_split = "train" if is_train else "validation"
        require(meta["split"] == expected_split, "Frozen split differs.")
        with np.load(path, allow_pickle=False) as z:
            record = {k: z[k].copy() for k in z.files}
        record.update(sample_id=sid, case_id=meta["case_id"], split=meta["split"])
        self.memory[sid] = record; self.opened[expected_split].add(sid)
        return record
