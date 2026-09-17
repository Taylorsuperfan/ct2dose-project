# Original supporting code — read only, NOT a Run-All notebook

Do not execute these top-level fragments without their original state; operational code uses explicit binding instead.

### Original notebook cell 5

```python
# ============================================================
# Final Phase9G settings
# ============================================================

CONFIG = {
    "seed": 42,
    "batch_size": 2,
    "num_workers": 2,

    # Euler sampling steps for base model
    "euler_steps": 30,

    # Phase9G deployable core calibration
    "core_scale": 0.990,
    "core_thr": 0.70,
    "tau": 0.04,

    # Professor/Card17-style percentage error threshold
    "threshold_frac": 0.01,

    # Evaluation amount
    # Use 300 if you want same as earlier professor-style eval.
    "max_eval_batches": 300,

    # Figure samples
    "num_auto_examples": 20,
}

print(json.dumps(CONFIG, indent=2))
```

### Original notebook cell 25

```python
# ============================================================
# Use exact Phase9D-plus metadata
# ============================================================

PHASE9D_PLUS_CKPT = CHECKPOINT_DIR / "ct2dose_phase9d_plus_stronger_axis_refinement_seed42_best.pt"

ckpt9d_meta = torch.load(PHASE9D_PLUS_CKPT, map_location="cpu")

print("PHASE9D_PLUS_CKPT:", PHASE9D_PLUS_CKPT)
print("Keys:", ckpt9d_meta.keys())

recorded_base = ckpt9d_meta["base_checkpoint"]
BASE_CKPT = Path(recorded_base)

if not BASE_CKPT.exists():
    BASE_CKPT = CHECKPOINT_DIR / Path(recorded_base).name

if not BASE_CKPT.exists():
    raise FileNotFoundError(f"Recorded base checkpoint not found: {recorded_base}")

PHASE9D_PLUS_CONFIG = ckpt9d_meta["phase9d_plus_config"]

print("\nUsing BASE_CKPT:")
print(BASE_CKPT)

print("\nPhase9D-plus config:")
for k, v in PHASE9D_PLUS_CONFIG.items():
    print(k, ":", v)

# Use exact euler steps from the successful Phase9D-plus run
CONFIG["euler_steps"] = int(PHASE9D_PLUS_CONFIG.get("euler_steps", 10))

print("\nFinal euler_steps used for reproduction:", CONFIG["euler_steps"])
```

### Original notebook cell 26

```python
base_ckpt_meta = torch.load(BASE_CKPT, map_location="cpu")
BASE_CONFIG = base_ckpt_meta.get("config", {})

BASE_CH = int(BASE_CONFIG.get("base_ch", 24))
DOSE_SCALE = float(BASE_CONFIG.get("dose_scale", 1000.0))

print("BASE_CKPT:", BASE_CKPT)
print("BASE_CH:", BASE_CH)
print("DOSE_SCALE:", DOSE_SCALE)

print("\nBase config:")
for k, v in BASE_CONFIG.items():
    print(k, ":", v)
```

### Original notebook cell 27

```python
base_model = ConditionalUNetFlow3D(
    in_ch=3,
    out_ch=1,
    base_ch=BASE_CH,
).to(device)

base_ckpt = load_state_dict_flexible(
    base_model,
    BASE_CKPT,
    device,
    strict_final=True,
)

base_model.eval()


model_9d_plus = MultiplicativeAdditiveRefineHead3D(
    in_ch=7,
    base_ch=16,
    log_scale_bound=float(PHASE9D_PLUS_CONFIG["log_scale_bound"]),  # 0.2
    additive_scale=float(PHASE9D_PLUS_CONFIG["additive_scale"]),    # 0.012
).to(device)

phase9d_ckpt = load_state_dict_flexible(
    model_9d_plus,
    PHASE9D_PLUS_CKPT,
    device,
    strict_final=True,
)

model_9d_plus.eval()

print("Models loaded cleanly.")
```

### Original notebook cell 50

```python
# ============================================================
# Phase10C: Direction-aware bounded refinement
# ============================================================

from pathlib import Path
import time
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

PHASE10C_NAME = "ct2dose_phase10c_direction_aware_bounded_refinement"

PHASE10C_DIR = PHASE9G_FINAL_DIR / PHASE10C_NAME
PHASE10C_DIR.mkdir(parents=True, exist_ok=True)

PHASE10C_CKPT_DIR = CHECKPOINT_DIR
PHASE10C_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Smoke-test config first.
# If it runs and looks promising, increase:
#   epochs: 3
#   max_train_batches: 1200
#   eval_max_batches: 300
# ------------------------------------------------------------

PHASE10C_CONFIG = {
    "seed": 42,

    # smoke-test training
    "epochs": 2,
    "lr": 5e-5,
    "weight_decay": 1e-5,
    "max_train_batches": 1000,
    "eval_max_batches": 200,

    # Phase9G inference base
    "euler_steps": int(CONFIG["euler_steps"]),

    # bounded correction
    "delta_scale": 0.010,
    "core_protect_strength": 0.75,
    "tail_protect_strength": 0.35,

    # region/profile thresholds
    "threshold_frac": float(CONFIG.get("threshold_frac", 0.01)),
    "main_frac": 0.30,
    "core_frac": 0.70,
    "shoulder_low": 0.30,
    "shoulder_high": 0.80,

    # losses
    "lambda_voxel": 0.6,
    "lambda_along": 4.0,
    "lambda_along_slope": 1.5,
    "lambda_perp_no_worse": 5.0,
    "lambda_core_no_worse": 3.0,
    "lambda_residual_l1": 0.08,
    "lambda_smooth": 0.06,

    # no-worse margins
    "perp_margin_abs": 0.0005,
    "core_margin_abs": 0.0005,

    # model
    "base_ch": 16,

    # Card17 reference for sample491
    "card17_mixed_along_x": 2.774812,
    "card17_mixed_perp_y": 1.614564,
}

torch.manual_seed(PHASE10C_CONFIG["seed"])
np.random.seed(PHASE10C_CONFIG["seed"])

print("PHASE10C_DIR:", PHASE10C_DIR)
print(json.dumps(PHASE10C_CONFIG, indent=2))
```

### Original notebook cell 66

```python
# ============================================================
# Phase10D train / eval loops
# ============================================================

def average_stats(stats_list):
    if len(stats_list) == 0:
        return {}

    keys = stats_list[0].keys()
    return {k: float(np.mean([s[k] for s in stats_list])) for k in keys}


def run_phase10d_train_epoch(epoch):
    phase10d_model.train()
    base_model.eval()
    model_9d_plus.eval()

    stats_list = []
    t0 = time.time()

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= PHASE10D_CONFIG["max_train_batches"]:
            break

        ct = batch["ct"].to(device, non_blocking=True)
        gt = batch["dose"].to(device, non_blocking=True)

        with torch.no_grad():
            preds9 = predict_phase9g(
                ct,
                steps=PHASE10D_CONFIG["euler_steps"],
            )
            phase9g_pred = preds9["phase9g_pred"]

        refined, aux = phase10d_model(ct, phase9g_pred)

        loss, stats = phase10d_loss(
            refined,
            phase9g_pred,
            gt,
            aux,
        )

        optimizer10d.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(phase10d_model.parameters(), max_norm=1.0)
        optimizer10d.step()

        stats_list.append(stats)

        if (batch_idx + 1) % 50 == 0:
            avg = average_stats(stats_list[-50:])
            print(
                f"epoch {epoch} batch {batch_idx+1} | "
                f"loss={avg['total']:.6f} | "
                f"along={avg['along_loss']:.6f} | "
                f"log={avg['along_log']:.6f} | "
                f"falloff={avg['falloff_log']:.6f} | "
                f"perpNW={avg['perp_noworse']:.6f} | "
                f"res={avg['residual_l1']:.6f}"
            )

    out = average_stats(stats_list)
    out["elapsed_min"] = (time.time() - t0) / 60.0

    return out


@torch.no_grad()
def evaluate_phase10d_profile_metrics(loader, max_batches=200):
    phase10d_model.eval()
    base_model.eval()
    model_9d_plus.eval()

    rows = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        ct = batch["ct"].to(device, non_blocking=True)
        gt = batch["dose"].to(device, non_blocking=True)

        preds9 = predict_phase9g(
            ct,
            steps=PHASE10D_CONFIG["euler_steps"],
        )

        phase9g_pred = preds9["phase9g_pred"]
        refined, aux = phase10d_model(ct, phase9g_pred)

        B = ct.shape[0]

        for b in range(B):
            p9_profiles = extract_axis_profiles(phase9g_pred[b], gt[b])
            p10_profiles = extract_axis_profiles(refined[b], gt[b])

            for axis in ["x", "y", "z"]:
                gt_p = p9_profiles[axis]["gt"]

                p9_p = p9_profiles[axis]["pred"]
                p10_p = p10_profiles[axis]["pred"]

                _, p9_mean, p9_max, valid_count, thr = percentage_error_curve(
                    p9_p,
                    gt_p,
                    threshold_frac=PHASE10D_CONFIG["threshold_frac"],
                )

                _, p10_mean, p10_max, _, _ = percentage_error_curve(
                    p10_p,
                    gt_p,
                    threshold_frac=PHASE10D_CONFIG["threshold_frac"],
                )

                rows.append({
                    "batch_idx": batch_idx,
                    "sample_local_idx": b,
                    "axis": axis,
                    "phase9g_mean_pct": p9_mean,
                    "phase10d_mean_pct": p10_mean,
                    "phase9g_max_pct": p9_max,
                    "phase10d_max_pct": p10_max,
                    "delta_mean_pct": p10_mean - p9_mean,
                    "valid_count": valid_count,
                    "threshold": thr,
                })

        if (batch_idx + 1) % 20 == 0:
            print(f"eval processed batches: {batch_idx+1}")

    casewise_df = pd.DataFrame(rows)

    summary_df = (
        casewise_df
        .groupby("axis")
        .agg(
            phase9g_mean_pct=("phase9g_mean_pct", "mean"),
            phase10d_mean_pct=("phase10d_mean_pct", "mean"),
            delta_mean_pct=("delta_mean_pct", "mean"),
            phase9g_max_pct=("phase9g_max_pct", "mean"),
            phase10d_max_pct=("phase10d_max_pct", "mean"),
        )
        .reset_index()
    )

    return casewise_df, summary_df
```

### Original notebook cell 67

```python
# ============================================================
# Train Phase10D-strict
# ============================================================

training_rows_10d = []
best_score_10d = float("inf")
best_epoch_10d = None

best_ckpt_path_10d = PHASE10D_STRICT_CKPT_DIR / f"{PHASE10D_STRICT_NAME}_seed42_best.pt"
latest_ckpt_path_10d = PHASE10D_STRICT_CKPT_DIR / f"{PHASE10D_STRICT_NAME}_seed42_latest.pt"

for epoch in range(1, PHASE10D_CONFIG["epochs"] + 1):
    print("=" * 100)
    print(f"Phase10D-strict Epoch {epoch}/{PHASE10D_CONFIG['epochs']}")

    train_stats = run_phase10d_train_epoch(epoch)

    val_casewise, val_summary = evaluate_phase10d_profile_metrics(
        val_loader,
        max_batches=PHASE10D_CONFIG["eval_max_batches"],
    )

    display(val_summary)

    x_row = val_summary[val_summary["axis"] == "x"].iloc[0]
    y_row = val_summary[val_summary["axis"] == "y"].iloc[0]
    z_row = val_summary[val_summary["axis"] == "z"].iloc[0]

    # Select best on strict_val2:
    # minimize x, strongly penalize y/z degradation relative to Phase9G.
    score = (
        x_row["phase10d_mean_pct"]
        + 4.0 * max(0.0, y_row["phase10d_mean_pct"] - y_row["phase9g_mean_pct"])
        + 4.0 * max(0.0, z_row["phase10d_mean_pct"] - z_row["phase9g_mean_pct"])
    )

    row = {
        "epoch": epoch,
        **{f"train_{k}": v for k, v in train_stats.items()},
        "score": float(score),
    }

    for _, r in val_summary.iterrows():
        axis = r["axis"]
        row[f"strict_val_{axis}_phase9g_mean_pct"] = r["phase9g_mean_pct"]
        row[f"strict_val_{axis}_phase10d_mean_pct"] = r["phase10d_mean_pct"]
        row[f"strict_val_{axis}_delta_mean_pct"] = r["delta_mean_pct"]

    training_rows_10d.append(row)

    log_df_10d = pd.DataFrame(training_rows_10d)
    display(log_df_10d.tail(1))

    torch.save(
        {
            "model_state_dict": phase10d_model.state_dict(),
            "optimizer_state_dict": optimizer10d.state_dict(),
            "config": PHASE10D_CONFIG,
            "epoch": epoch,
            "row": row,
            "split_protocol": "strict_train6_val2_test2_refinement_head",
        },
        latest_ckpt_path_10d,
    )

    if score < best_score_10d:
        best_score_10d = float(score)
        best_epoch_10d = epoch

        torch.save(
            {
                "model_state_dict": phase10d_model.state_dict(),
                "optimizer_state_dict": optimizer10d.state_dict(),
                "config": PHASE10D_CONFIG,
                "epoch": epoch,
                "row": row,
                "best_score": best_score_10d,
                "split_protocol": "strict_train6_val2_test2_refinement_head",
            },
            best_ckpt_path_10d,
        )

        print(f"New best saved: epoch={epoch}, score={best_score_10d:.6f}")

training_log_df_10d = pd.DataFrame(training_rows_10d)

training_log_csv_10d = PHASE10D_STRICT_DIR / f"{PHASE10D_STRICT_NAME}_training_log.csv"
training_log_df_10d.to_csv(training_log_csv_10d, index=False)

print("Finished Phase10D-strict.")
print("Best epoch:", best_epoch_10d)
print("Best score:", best_score_10d)
print("Saved:", training_log_csv_10d)
print("Best ckpt:", best_ckpt_path_10d)
```

### Original notebook cell 68

```python
# ============================================================
# Load best Phase10D-strict checkpoint
# ============================================================

best_ckpt_10d = torch.load(best_ckpt_path_10d, map_location=device)

phase10d_model.load_state_dict(best_ckpt_10d["model_state_dict"], strict=True)
phase10d_model.eval()

print("Loaded best Phase10D-strict checkpoint:")
print(best_ckpt_path_10d)
print("Best epoch:", best_ckpt_10d.get("epoch"))
print("Best score:", best_ckpt_10d.get("best_score"))
```

### Original notebook cell 70

```python
# ============================================================
# Create stratified eval subsets for strict val/test
# ============================================================

from pathlib import Path
import json
import random
from collections import defaultdict, Counter
import pandas as pd

STRICT_EVAL_SUBSET_DIR = STRICT_SPLIT_DIR / "eval_subsets"
STRICT_EVAL_SUBSET_DIR.mkdir(parents=True, exist_ok=True)


def load_records_from_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "records" in data:
            data = data["records"]
        elif "data" in data:
            data = data["data"]
        else:
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

    if not isinstance(data, list):
        raise ValueError(f"Could not parse records from {json_path}")

    return data


def infer_case_id_from_record(rec):
    if isinstance(rec, dict):
        for key in ["case_id", "patient_id", "case", "caseid"]:
            if key in rec:
                return str(rec[key])

        for v in rec.values():
            if isinstance(v, str):
                for part in Path(v).parts:
                    if part.isdigit() and len(part) > 12:
                        return str(part)

            if isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, str):
                        for part in Path(vv).parts:
                            if part.isdigit() and len(part) > 12:
                                return str(part)

    return "unknown"


def make_stratified_subset_json(
    src_json,
    out_json,
    n_per_case=300,
    seed=42,
):
    records = load_records_from_json(src_json)

    groups = defaultdict(list)
    for r in records:
        cid = infer_case_id_from_record(r)
        groups[cid].append(r)

    rng = random.Random(seed)

    subset_records = []
    rows = []

    for cid, recs in sorted(groups.items()):
        recs_copy = list(recs)
        rng.shuffle(recs_copy)

        chosen = recs_copy[:min(n_per_case, len(recs_copy))]
        subset_records.extend(chosen)

        rows.append({
            "case_id": cid,
            "n_available": len(recs),
            "n_selected": len(chosen),
        })

    # Keep deterministic ordering after sampling
    rng.shuffle(subset_records)

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w") as f:
        json.dump(subset_records, f, indent=2)

    summary_df = pd.DataFrame(rows)

    print("Saved subset:", out_json)
    print("Total selected:", len(subset_records))
    display(summary_df)

    return out_json, summary_df


# For quick but representative validation:
# 300 per case = 600 records = 300 batches if batch_size=2.
N_PER_CASE_EVAL = 300

STRICT_VAL2_SUBSET_JSON, strict_val_subset_summary = make_stratified_subset_json(
    STRICT_VAL2_JSON,
    STRICT_EVAL_SUBSET_DIR / f"val2_stratified_{N_PER_CASE_EVAL}_per_case.json",
    n_per_case=N_PER_CASE_EVAL,
    seed=42,
)

STRICT_TEST2_SUBSET_JSON, strict_test_subset_summary = make_stratified_subset_json(
    STRICT_TEST2_JSON,
    STRICT_EVAL_SUBSET_DIR / f"test2_stratified_{N_PER_CASE_EVAL}_per_case.json",
    n_per_case=N_PER_CASE_EVAL,
    seed=42,
)
```

### Original notebook cell 73

```python
# ============================================================
# Fast stratified strict validation evaluation
# ============================================================

strict_val_casewise_10d, strict_val_summary_10d = evaluate_phase10d_profile_metrics(
    strict_val_subset_loader,
    max_batches=None,
)

strict_val_casewise_csv = PHASE10D_STRICT_DIR / f"phase10d_strict_val2_stratified_{N_PER_CASE_EVAL}_per_case_casewise_axis_metrics.csv"
strict_val_summary_csv = PHASE10D_STRICT_DIR / f"phase10d_strict_val2_stratified_{N_PER_CASE_EVAL}_per_case_summary.csv"

strict_val_casewise_10d.to_csv(strict_val_casewise_csv, index=False)
strict_val_summary_10d.to_csv(strict_val_summary_csv, index=False)

print("Saved:", strict_val_casewise_csv)
print("Saved:", strict_val_summary_csv)
display(strict_val_summary_10d)
```

### Original notebook cell 94

```python
# ============================================================
# Phase10D-strict config
# train = strict_train6
# select = strict_val2
# final test = strict_test2
# ============================================================

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

PHASE10D_STRICT_NAME = "ct2dose_phase10d_falloff_aware_direction_refinement_strict_train6_val2"

PHASE10D_STRICT_DIR = PHASE9G_FINAL_DIR / PHASE10D_STRICT_NAME
PHASE10D_STRICT_DIR.mkdir(parents=True, exist_ok=True)

PHASE10D_STRICT_CKPT_DIR = CHECKPOINT_DIR
PHASE10D_STRICT_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Start with a manageable strict run.
# If time allows, set eval_max_batches=None for full strict_val2 selection.
PHASE10D_CONFIG = {
    "seed": 42,

    # training
    "epochs": 2,
    "lr": 4e-5,
    "weight_decay": 1e-5,
    "max_train_batches": 800,

    # For quick strict rerun use 300 or 500.
    # For formal strict selection use None.
    "eval_max_batches": 300,

    "euler_steps": int(CONFIG["euler_steps"]),

    # bounded correction
    "delta_scale": 0.008,
    "core_protect_strength": 0.80,
    "tail_protect_strength": 0.25,

    # profile thresholds
    "threshold_frac": float(CONFIG.get("threshold_frac", 0.01)),
    "main_frac": 0.30,
    "core_frac": 0.70,
    "shoulder_low": 0.30,
    "shoulder_high": 0.80,
    "falloff_low_frac": 0.05,
    "falloff_high_frac": 0.45,

    # losses
    "lambda_voxel": 0.4,
    "lambda_along": 2.0,
    "lambda_along_slope": 1.0,
    "lambda_along_log": 2.5,
    "lambda_falloff_log": 4.0,
    "lambda_falloff_no_worse": 4.0,
    "lambda_perp_no_worse": 6.0,
    "lambda_core_no_worse": 3.0,
    "lambda_residual_l1": 0.10,
    "lambda_smooth": 0.08,

    # margins
    "perp_margin_abs": 0.0004,
    "core_margin_abs": 0.0004,
    "falloff_margin_log": 0.015,

    # model
    "base_ch": 16,

    # references
    "card17_mixed_along_x": 2.774812,
    "card17_mixed_perp_y": 1.614564,

    # strict split metadata
    "strict_train_json": str(STRICT_TRAIN6_JSON),
    "strict_val_json": str(STRICT_VAL2_JSON),
    "strict_test_json": str(STRICT_TEST2_JSON),
    "note": (
        "Strict refinement-head rerun. Phase10D head is trained on strict_train6, "
        "selected on strict_val2, and evaluated on strict_test2. "
        "Base Phase9G pipeline is reused from previous checkpoint."
    ),
}

torch.manual_seed(PHASE10D_CONFIG["seed"])
np.random.seed(PHASE10D_CONFIG["seed"])

print("PHASE10D_STRICT_DIR:", PHASE10D_STRICT_DIR)
print(json.dumps(PHASE10D_CONFIG, indent=2))
```

### Original notebook cell 96

```python
# ============================================================
# Temporarily use strict loaders for Phase10D-strict
# ============================================================

old_train_loader = train_loader if "train_loader" in globals() else None
old_val_loader = val_loader if "val_loader" in globals() else None

train_loader = strict_train_loader
val_loader = strict_val_loader

print("Now using strict loaders:")
print("train_loader -> strict_train_loader:", len(train_loader.dataset))
print("val_loader   -> strict_val_loader:", len(val_loader.dataset))
```
