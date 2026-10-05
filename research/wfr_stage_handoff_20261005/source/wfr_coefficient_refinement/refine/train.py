"""Train two coefficient refinements from the same saved initialization."""
from pathlib import Path
import copy
import time
import numpy as np
import pandas as pd
import torch
from cwfr import common as io
from cwfr.metrics import score
from .cache import FrozenData
from .config import ARMS
from .integrity import require, state_digest
from .model import CoefficientHead
from .losses import make_batch, objective
from .selection import assess, improves_primary


def summarize_rows(rows):
    frame = pd.DataFrame(rows)
    keys = [k for k in frame.columns if k not in ("sample_id", "case_id")]
    numeric = frame.groupby("case_id", sort=True)[keys].mean(numeric_only=True)
    clean = lambda row: {k: float(v) if np.isfinite(v) else None for k, v in row.items()}
    cases = {str(k): clean(v.to_dict()) for k, v in numeric.iterrows()}
    mean = clean(numeric.mean().to_dict())
    mean.update(n_records=len(frame), n_cases=len(numeric))
    return mean, cases, frame


def factor_metrics(u, a, coefficient):
    active = np.abs(u) > .01
    correct = np.sign(coefficient) == np.sign(u)
    positive = active & (u > 0); negative = active & (u < 0)
    rho = np.abs(u); denominator = float(rho[active].sum())
    return {
        "active_sign_accuracy": float(correct[active].mean()) if active.any() else None,
        "positive_recall": float(correct[positive].mean()) if positive.any() else None,
        "negative_recall": float(correct[negative].mean()) if negative.any() else None,
        "magnitude_weighted_sign_error": float((rho[active] * (~correct[active])).sum()/denominator) if denominator else None,
        "coefficient_abs_mean_active": float(np.abs(coefficient[active]).mean()) if active.any() else None,
        "retained_magnitude_fraction": float((a*np.abs(coefficient)).sum()/a.sum()) if a.sum() else None,
        "residual_rmse_normalized": float(np.sqrt(np.mean((a*coefficient-u)**2))),
    }


@torch.no_grad()
def evaluate_head(head, frozen, ids, device, reference=False):
    if head is not None:
        head.eval()
    rows = []
    for sid in ids:
        rec = frozen.load(sid, allow_monitor=True)
        if reference:
            coefficient = rec["parent_sign"].astype(float)
            raw = rec["parent_raw"]
            clipped = rec["parent_prediction"]
        else:
            condition = torch.as_tensor(rec["condition"][None], device=device)
            coefficient = head.coefficient(condition)[0].cpu().numpy().astype(float)
            raw = ((rec["base"].astype(float) + frozen.scale*rec["magnitude"].astype(float)*coefficient)
                   / float(rec["factor"])).astype(np.float32)
            clipped = np.maximum(raw, 0).astype(np.float32)
        stats = score(clipped, rec["target_stored"], float(rec["factor"]))
        stats["raw_rmse_stored_units"] = score(raw, rec["target_stored"], float(rec["factor"]))["rmse_stored_units"]
        stats.update(factor_metrics(rec["residual"], rec["magnitude"], coefficient))
        rows.append({"sample_id": sid, "case_id": rec["case_id"], **stats})
    return summarize_rows(rows)


def make_schedule(ids, cfg):
    rng = np.random.default_rng(cfg.seed)
    index = rng.integers(0, len(ids), size=(cfg.updates, cfg.batch_size))
    return [[ids[int(k)] for k in batch] for batch in index]


def _save(run, head, optimizer, state, contract):
    payload = {"model": head.state_dict(), "optimizer": optimizer.state_dict(),
               "rng": io.rng_state(), "state": copy.deepcopy(state),
               "step": state["step"], "contract_sha256": io.digest(contract)}
    pointer = io.save_checkpoint(run, payload)
    io.write(run / "last.json", pointer, replace=True)
    return pointer


def train(work, arm, device="cpu", max_updates=128, run_override=None):
    if arm not in ARMS:
        raise ValueError("Choose composition or profile.")
    frozen = FrozenData(work); cfg = frozen.cfg
    run = Path(run_override).resolve() if run_override else frozen.work / "runs.local" / (arm + "_seed" + str(cfg.seed))
    io.separate(run, frozen.root)
    require(run.is_relative_to(frozen.work / "runs.local"), "Run must be under this experiment's runs.local directory.")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    io.seed_all(cfg.seed)
    schedule = make_schedule(frozen.train_ids, cfg)
    identity = {"arm": arm, "protocol_sha256": io.digest(frozen.protocol),
                "frozen_cache_ledger": frozen.receipt.complete["files_sha256"],
                "initial_tensor_sha256": frozen.initial_meta["tensor_sha256"],
                "schedule_sha256": io.digest(schedule), "extra_updates": cfg.updates,
                "parent_total_updates": 384, "only_coefficient_parameters_optimized": True}
    run.mkdir(parents=True, exist_ok=True)
    with io.single_writer(run):
        env = io.environment(device)
        cp = run / "contract.json"
        if cp.exists():
            contract = io.read(cp)
            require(contract["identity"] == identity, "Run science/data identity changed. Do not overwrite its contract.")
            if (run / "COMPLETE.json").exists():
                io.verify_finish(run, "coefficient_training_completed")
                print("Completed coefficient run verified; no retraining:", run, flush=True)
                return run
            if contract["environment"] != env:
                raise RuntimeError("Training environment changed. Use the explicit fork command; keep the original run.")
        else:
            require(not any(run.iterdir()), "Nonempty run directory without a contract.")
            contract = {"identity": identity, "environment": env, "parent_run": None}
            io.write(cp, contract)
        io.write(run / "batch_schedule.private.json", schedule)
        head = CoefficientHead(frozen.initial_meta["channels"]).to(device)
        head.load_state_dict(frozen.initial_state, strict=True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for sid in frozen.train_ids:
            frozen.load(sid)
        for sid in frozen.monitor_ids:
            frozen.load(sid, allow_monitor=True)
        if (run / "last.json").exists():
            ck = io.load_checkpoint(run, io.read(run / "last.json"))
            require(ck["contract_sha256"] == io.digest(contract), "Checkpoint contract differs.")
            head.load_state_dict(ck["model"], strict=True); optimizer.load_state_dict(ck["optimizer"])
            for value in optimizer.state.values():
                for k, v in value.items():
                    if torch.is_tensor(v):
                        value[k] = v.to(device)
            state = ck["state"]
            selected_path = run / "selected.json"
            selection_record = io.read(selected_path) if selected_path.exists() else None
            if selection_record is None or selection_record["step"] != state["selected_step"]:
                require(state["selected_step"] == ck["step"],
                        "Selection receipt cannot be recovered from the last checkpoint. Preserve the run and inspect it.")
                io.write(selected_path, {
                    "kind": "parent_fallback" if ck["step"] == 0 else "accepted_refinement",
                    "pointer": io.read(run / "last.json"), "step": ck["step"]}, replace=True)
            io.restore_rng(ck["rng"])
            print(f"RESUME {arm}: next additional update={state['step']+1}", flush=True)
        else:
            require(state_digest(head.state_dict()) == frozen.initial_meta["tensor_sha256"], "Initialization is not the selected parent coefficient.")
            reference, reference_cases, frame = evaluate_head(None, frozen, frozen.monitor_ids, device, reference=True)
            initial, initial_cases, initial_frame = evaluate_head(head, frozen, frozen.monitor_ids, device)
            # Compare the reconstructed raw outputs, not merely a rounded overall RMSE.
            max_difference = 0.
            with torch.no_grad():
                for sid in frozen.monitor_ids:
                    rec = frozen.load(sid, allow_monitor=True)
                    s = head.coefficient(torch.as_tensor(rec["condition"][None], device=device))[0].cpu().numpy().astype(float)
                    pred = (rec["base"].astype(float) + frozen.scale*rec["magnitude"].astype(float)*s)/float(rec["factor"])
                    err = np.abs(pred-rec["parent_raw"].astype(float))
                    require(np.all(err <= 1e-9 + 1e-6*np.abs(rec["parent_raw"])),
                            "The coefficient initialization no longer reconstructs the saved parent. Review the numerical/environment difference.")
                    max_difference = max(max_difference, float(err.max()))
            state = {"step": 0, "reference": reference, "reference_cases": reference_cases,
                     "selected_metrics": reference, "selected_step": 0,
                     "loss_history": [], "monitor_history": [], "training_seconds": 0.,
                     "initial_raw_max_abs_difference": max_difference}
            pointer = _save(run, head, optimizer, state, contract)
            io.write(run / "selected.json", {"kind": "parent_fallback", "pointer": pointer, "step": 0})
            io.write(run / "initial_monitor.json", {"saved_parent_reference": reference,
                     "fresh_head": initial, "raw_max_abs_difference": max_difference,
                     "note": "The saved parent remains the fixed selection reference."})
            frame.to_csv(run / "reference_monitor_records.private.csv", index=False)
            print(f"START {arm}: parent x={reference['x_mean_pct']:.6g}, train={len(frozen.train_ids)} monitor={len(frozen.monitor_ids)}", flush=True)
        mark = time.monotonic(); count = 0
        try:
            while state["step"] < cfg.updates:
                head.train()
                records = [frozen.load(sid) for sid in schedule[state["step"]]]
                batch = make_batch(records, device, frozen.scale)
                optimizer.zero_grad(set_to_none=True)
                loss, terms = objective(head(batch["condition"]), batch, frozen.scales, cfg, arm)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss; no optimizer step was taken.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
                optimizer.step(); state["step"] += 1; count += 1
                terms.update(step=state["step"], gradient_norm=float(norm))
                state["loss_history"].append(terms)
                monitor_now = state["step"] % cfg.monitor_every == 0 or state["step"] == cfg.updates
                if monitor_now:
                    summary, cases, frame = evaluate_head(head, frozen, frozen.monitor_ids, device)
                    check = assess(summary, cases, state["reference"], state["reference_cases"], cfg)
                    selected = check["eligible"] and improves_primary(summary, state["selected_metrics"], cfg)
                    if selected:
                        state["selected_metrics"] = summary; state["selected_step"] = state["step"]
                    history = {"step": state["step"], **summary, "eligible": check["eligible"],
                               "selected_now": selected, "guard_failures": ";".join(check["failures"])}
                    state["monitor_history"].append(history)
                    state["training_seconds"] += time.monotonic()-mark; mark = time.monotonic()
                    pointer = _save(run, head, optimizer, state, contract)
                    if selected:
                        io.write(run / "selected.json", {"kind": "accepted_refinement", "pointer": pointer,
                                 "step": state["step"]}, replace=True)
                    frame.to_csv(run / (f"monitor_step_{state['step']:06d}.private.csv"), index=False)
                    pd.DataFrame(state["monitor_history"]).to_csv(run / "monitor_history.csv", index=False)
                    pd.DataFrame(state["loss_history"]).to_csv(run / "loss_history.csv", index=False)
                    print(f"UPDATE {state['step']}/{cfg.updates} {arm}: loss={terms['total']:.5g}, "
                          f"x={summary['x_mean_pct']:.6g}, RMSE={summary['rmse_stored_units']:.8g}, "
                          f"eligible={check['eligible']}, selected={selected}", flush=True)
                elif state["step"] % cfg.save_every == 0:
                    state["training_seconds"] += time.monotonic()-mark; mark = time.monotonic()
                    _save(run, head, optimizer, state, contract)
                if max_updates is not None and count >= max_updates and state["step"] < cfg.updates:
                    state["training_seconds"] += time.monotonic()-mark; mark = time.monotonic()
                    _save(run, head, optimizer, state, contract)
                    io.write(run / "status.json", {"status": "paused", "additional_updates": state["step"]}, replace=True)
                    print("PAUSED safely. Repeat the same train command.", flush=True)
                    return run
            state["training_seconds"] += time.monotonic()-mark
            # Ensure the fixed-budget last checkpoint is retained, even when it fails the gate.
            _save(run, head, optimizer, state, contract)
            selected = io.read(run / "selected.json")
            report = {"arm": arm, "status": "accepted_refinement" if selected["step"] > 0 else "no_eligible_refinement_parent_retained",
                      "parent_updates": 384, "additional_updates_executed": cfg.updates,
                      "selected_additional_step": selected["step"], "cumulative_updates_at_selected_state": 384+selected["step"],
                      "total_experiment_updates": 384+cfg.updates,
                      "selected_monitor": state["selected_metrics"], "parent_monitor": state["reference"],
                      "coefficient_parameters": sum(p.numel() for p in head.parameters()),
                      "training_seconds_accumulated": state["training_seconds"],
                      "training_cases": 6, "training_records": len(frozen.train_ids), "monitor_records": len(frozen.monitor_ids),
                      "parent_magnitude_updated": False, "parent_initialization_sha256": frozen.initial_meta["tensor_sha256"],
                      "schedule_sha256": io.digest(schedule), "guard_relative_tolerance": cfg.guard_relative_tolerance,
                      "scope": "DEVELOPMENT_MONITOR_SELECTION_NOT_TEST_OR_CLINICAL_ACCEPTANCE"}
            io.write(run / "training_summary.json", report)
            frozen.receipt.recheck()
            io.finish(run, "coefficient_training_completed", contract_sha256=io.digest(contract), updates=cfg.updates)
            print("TRAINING COMPLETE:", run, flush=True)
            print(report, flush=True)
            return run
        except BaseException as exc:
            io.write(run / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                     "note": "Resume only the last successfully saved checkpoint."}, replace=True)
            raise


def fork(work, parent_run, out, device):
    """Continue an unfinished arm in a new environment, with explicit retained ancestry."""
    frozen = FrozenData(work); parent_run = Path(parent_run).resolve(); out = Path(out).resolve()
    io.separate(out, parent_run, frozen.root)
    require(out.is_relative_to(frozen.work / "runs.local"), "Fork destination must be inside runs.local.")
    require(not out.exists() or not any(out.iterdir()), "Fork destination is not empty.")
    old = io.read(parent_run / "contract.json")
    require(old["identity"]["protocol_sha256"] == io.digest(frozen.protocol), "Parent protocol differs.")
    pointer = io.read(parent_run / "last.json"); last = io.load_checkpoint(parent_run, pointer)
    require(last["contract_sha256"] == io.digest(old), "Fork checkpoint linkage differs.")
    require(last["step"] < frozen.cfg.updates, "A completed arm is not an unfinished continuation.")
    selected = io.read(parent_run / "selected.json")
    best = io.load_checkpoint(parent_run, selected["pointer"])
    require(best["contract_sha256"] == io.digest(old), "Selected checkpoint linkage differs.")
    io.seed_all(frozen.cfg.seed); env = io.environment(device)
    new = {"identity": old["identity"], "environment": env,
           "parent_run": {"path": str(parent_run), "contract_sha256": io.digest(old),
                          "last": pointer, "kind": "explicit_non_bitwise_environment_continuation"}}
    out.mkdir(parents=True, exist_ok=True); io.write(out / "contract.json", new)
    io.write(out / "batch_schedule.private.json", make_schedule(frozen.train_ids, frozen.cfg))
    for saved in (last, best):
        saved["contract_sha256"] = io.digest(new)
        if old["environment"] != env:
            saved["rng"]["cuda"] = io.rng_state()["cuda"]
    lp = io.save_checkpoint(out, last); bp = io.save_checkpoint(out, best)
    io.write(out / "last.json", lp)
    io.write(out / "selected.json", {**selected, "pointer": bp})
    io.write(out / "resume_notice.json", {"message": "CPU/NumPy RNG and optimizer are retained; CUDA RNG is explicitly reset on environment change.",
                                         "counter_includes_parent_arm_updates": True})
    print("Fork created, not trained:", out, flush=True)
    return out
