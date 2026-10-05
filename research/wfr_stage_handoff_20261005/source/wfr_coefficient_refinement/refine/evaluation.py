"""Lock both selections before reading validation600, then score saved magnitudes."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from cwfr import common as io
from cwfr.metrics import score, aggregate
from .integrity import Receipt, require, save_npz
from .cache import FrozenData, open_protocol
from .config import ARMS
from .model import CoefficientHead
from .train import factor_metrics


def default_run(frozen, arm):
    return frozen.work / "runs.local" / (arm + "_seed" + str(frozen.cfg.seed))


def finalize(work, composition_run=None, profile_run=None):
    frozen = FrozenData(work)
    roots = {"composition": Path(composition_run) if composition_run else default_run(frozen, "composition"),
             "profile": Path(profile_run) if profile_run else default_run(frozen, "profile")}
    arms = {}; identities = []
    for arm in ARMS:
        root = roots[arm].resolve(); receipt = Receipt(root, "coefficient_training_completed")
        c = receipt.json("contract.json"); selected = receipt.json("selected.json"); last = receipt.json("last.json")
        require(c["identity"]["arm"] == arm and c["identity"]["protocol_sha256"] == io.digest(frozen.protocol), "Training arm linkage differs.")
        require(receipt.complete["updates"] == frozen.cfg.updates, "Complete the declared training budget for both arms first.")
        require(receipt.complete["contract_sha256"] == io.digest(c), "Training receipt contract differs.")
        for pointer in (selected["pointer"], last):
            require(io.sha(receipt.file(pointer["path"])) == pointer["sha256"], "Selected/last checkpoint changed.")
        identities.append(c["identity"])
        arms[arm] = {"run": str(root), "contract_sha256": io.digest(c),
                     "ledger_sha256": receipt.complete["files_sha256"],
                     "selected": selected, "last": last,
                     "summary": receipt.json("training_summary.json"), "environment": c["environment"]}
    for key in ("initial_tensor_sha256", "schedule_sha256", "frozen_cache_ledger", "extra_updates"):
        require(identities[0][key] == identities[1][key], "The A/B experiment differs on " + key)
    result = {"protocol_sha256": io.digest(frozen.protocol), "arms": arms,
              "same_training_environment": arms["composition"]["environment"] == arms["profile"]["environment"],
              "selection_locked_before_new_validation600": True,
              "validation_is_independent_test": False,
              "last_checkpoint_role": "Fixed-budget secondary comparison; not a replacement chosen after validation."}
    io.write(frozen.work / "PAIR_FROZEN.json", result)
    print("Both selections are locked:", frozen.work / "PAIR_FROZEN.json", flush=True)
    for arm in ARMS:
        print(arm, arms[arm]["selected"]["kind"], "selected extra step=", arms[arm]["selected"]["step"])
    return result


def load_locked_heads(frozen, pair, device):
    result = {}
    for arm in ARMS:
        record = pair["arms"][arm]; root = Path(record["run"])
        receipt = Receipt(root, "coefficient_training_completed")
        require(receipt.complete["files_sha256"] == record["ledger_sha256"], "Locked training ledger changed.")
        contract = receipt.json("contract.json")
        require(io.digest(contract) == record["contract_sha256"], "Locked training contract changed.")
        models = {}
        for role in ("selected", "last"):
            pointer = record["selected"]["pointer"] if role == "selected" else record["last"]
            receipt.file(pointer["path"])
            ck = io.load_checkpoint(root, pointer)
            require(ck["contract_sha256"] == io.digest(contract), "Checkpoint ancestry differs.")
            head = CoefficientHead(frozen.initial_meta["channels"]).to(device)
            head.load_state_dict(ck["model"], strict=True); head.eval()
            for parameter in head.parameters():
                parameter.requires_grad_(False)
            models[role] = head
        result[arm] = models
    return result


def evaluate(cache_path, parent_work, work, device="cpu", out_override=None, max_records=100):
    data, parent, protocol, cfg = open_protocol(cache_path, parent_work, work)
    frozen = FrozenData(work)
    pair = io.read(Path(work) / "PAIR_FROZEN.json")
    require(pair["protocol_sha256"] == io.digest(protocol), "Locked pair uses another protocol.")
    ids = parent.selection["validation_ids"]
    out = Path(out_override).resolve() if out_override else Path(work) / "evaluations.local" / "locked_pair_val600"
    io.separate(out, frozen.root, parent.work, data.path, data.old, Path(work) / "runs.local")
    identity = {"protocol_sha256": io.digest(protocol), "pair_sha256": io.digest(pair),
                "parent_identity": parent.identity, "validation_ids": ids,
                "primary": "both_preselected_arms", "secondary": "both_fixed_budget_last_checkpoints",
                "transport_recomputed": False, "scope": "DEVELOPMENT_VALIDATION600_NOT_INDEPENDENT_TEST"}
    io.seed_all(cfg.seed); env = io.environment(device)
    out.mkdir(parents=True, exist_ok=True)
    with io.single_writer(out):
        cp = out / "contract.json"
        if cp.exists():
            contract = io.read(cp)
            require(contract["identity"] == identity, "Evaluation identity changed.")
            if (out / "COMPLETE.json").exists():
                io.verify_finish(out, "coefficient_pair_validation_completed")
                print("Saved paired evaluation verified; no inference.", flush=True)
                return out
            if contract["environment"] != env:
                raise RuntimeError("Partial evaluation environment changed. Use --out with a new evaluation directory; the fixed pair is unchanged.")
        else:
            require(not any(out.iterdir()), "Nonempty evaluation without a contract.")
            contract = {"identity": identity, "environment": env}; io.write(cp, contract)
        heads = load_locked_heads(frozen, pair, device)
        rows = []; diagnostic_rows = []; new_count = 0
        for j, sid in enumerate(ids, 1):
            item = data.load(sid, allow_validation=True)
            saved, provenance = parent.prediction(sid)
            folder = out / "records.local" / sid
            arrfile = folder / "arrays.npz"; metafile = folder / "record.json"
            if metafile.exists():
                meta = io.read(metafile)
                require(meta["contract_sha256"] == io.digest(contract) and meta["arrays_sha256"] == io.sha(arrfile)
                        and meta["parent_prediction"] == provenance and meta["sample_id"] == sid
                        and meta["case_id"] == item["row"]["case_id"], "Saved evaluation record changed.")
            else:
                if arrfile.exists():
                    raise RuntimeError("Unsealed evaluation array exists; preserve it and inspect: " + str(arrfile))
                condition = torch.as_tensor(item["condition"][None], device=device)
                a = saved["magnitude_normalized"].astype(float)
                outputs = {}; record_scores = {}; record_diags = {}
                for arm in ARMS:
                    for role in ("selected", "last"):
                        key = arm + "_" + role
                        fallback = role == "selected" and pair["arms"][arm]["selected"]["kind"] == "parent_fallback"
                        if fallback:
                            coefficient = saved["soft_sign"].copy()
                            raw = saved["raw_prediction_stored"].copy()
                            pred = saved["prediction_stored"].copy()
                        else:
                            with torch.no_grad():
                                coefficient = heads[arm][role].coefficient(condition)[0].cpu().numpy().astype(np.float32)
                            raw = ((item["base"].astype(float) + frozen.scale*a*coefficient.astype(float))
                                   / item["factor"]).astype(np.float32)
                            pred = np.maximum(raw, 0).astype(np.float32)
                        outputs[key + "_prediction_stored"] = pred
                        outputs[key + "_raw_prediction_stored"] = raw
                        outputs[key + "_coefficient"] = coefficient
                        s = score(pred, item["original"]["target_stored"], item["factor"])
                        s["raw_rmse_stored_units"] = score(raw, item["original"]["target_stored"], item["factor"])["rmse_stored_units"]
                        record_scores[key] = s
                        record_diags[key] = factor_metrics(item["residual"], a, coefficient)
                save_npz(arrfile, outputs)
                meta = {"sample_id": sid, "case_id": item["row"]["case_id"],
                        "source_arrays_sha256": item["row"]["arrays_sha256"],
                        "source_record_sha256": item["row"]["record_meta_sha256"],
                        "contract_sha256": io.digest(contract), "parent_prediction": provenance,
                        "arrays_sha256": io.sha(arrfile), "metrics": record_scores, "factors": record_diags,
                        "new_transport_inference": False, "target_used_in_head": False}
                io.write(metafile, meta); new_count += 1
            for method, stats in meta["metrics"].items():
                rows.append({"method": method, "sample_id": sid, "case_id": meta["case_id"], **stats})
                diagnostic_rows.append({"method": method, "sample_id": sid, "case_id": meta["case_id"], **meta["factors"][method]})
            if j % 25 == 0 or j == 1:
                print(f"COEFFICIENT EVALUATION {j}/{len(ids)} saved/verified", flush=True)
            if max_records is not None and new_count >= max_records and j < len(ids):
                print("Evaluation paused safely. Repeat the same evaluate command.", flush=True)
                return out
        frame, cases, summary = aggregate(rows)
        frame.to_csv(out / "metrics_records.private.csv", index=False)
        cases.to_csv(out / "metrics_cases.private.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        pd.DataFrame(diagnostic_rows).to_csv(out / "factor_records.private.csv", index=False)
        parent.recheck()
        io.finish(out, "coefficient_pair_validation_completed", contract_sha256=io.digest(contract),
                  n_records=len(ids), n_cases=2, new_transport_inference=False, test_arrays_opened=0)
        print("PAIRED EVALUATION COMPLETE:", out, flush=True)
        print(summary.to_string(index=False), flush=True)
        return out
