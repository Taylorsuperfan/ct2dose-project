"""Paired scores from unchanged saved predictions, with clear primary/secondary roles."""
from pathlib import Path
import numpy as np
import pandas as pd
from cwfr import common as io
from cwfr.metrics import score, aggregate
from cwfr.evaluate import _previous_eval
from .cache import open_protocol
from .integrity import Receipt, require
from .config import ARMS

NAMES = {
    "composition_selected": "Composition-supervised coefficient refinement",
    "profile_selected": "Profile-supervised coefficient refinement",
    "composition_last": "Composition refinement: fixed-budget last state (secondary)",
    "profile_last": "Profile refinement: fixed-budget last state (secondary)",
    "parent_wfr": "Previous conditional WFR magnitude and grid coefficient",
    "previous_positive_negative_flows": "Previous separate positive/negative correction flows",
    "previous_direct_signed_flow": "Previous directly learned signed correction",
    "previous_final_system": "Previous final Practical dose system",
    "shared_calibrated_predictor": "Shared calibrated dose predictor",
}


def compare(cache_path, parent_work, work, hjd_path, direct_path, evaluation=None, out=None):
    data, parent, protocol, cfg = open_protocol(cache_path, parent_work, work)
    work = Path(work).resolve()
    pair = io.read(work / "PAIR_FROZEN.json")
    require(pair["protocol_sha256"] == io.digest(protocol), "Pair protocol mismatch.")
    evaluation = Path(evaluation).resolve() if evaluation else work / "evaluations.local/locked_pair_val600"
    receipt = Receipt(evaluation, "coefficient_pair_validation_completed")
    ec = receipt.json("contract.json")
    require(ec["identity"]["pair_sha256"] == io.digest(pair) and
            ec["identity"]["parent_identity"] == parent.identity, "Paired evaluation ancestry differs.")
    require(ec["identity"]["validation_ids"] == parent.selection["validation_ids"], "Evaluation cohort or order differs.")
    require(receipt.complete["contract_sha256"] == io.digest(ec), "Evaluation completion linkage differs.")
    old_contract = io.read(data.old / "contract.local.json")
    old_status = io.read(data.old / "run_status.json")
    require(io.digest(old_contract) == data.contract["baseline_contract_sha256"] and
            old_status["status"] == "completed_validation_inference" and old_status["n_records"] == 600,
            "The Practical final baseline is not the saved full600 evaluation.")
    hjd, hc = _previous_eval(hjd_path, data, "new_hjd_rf")
    direct, dc = _previous_eval(direct_path, data, "new_residual_rf")
    out = Path(out).resolve() if out else work / "comparisons.local/locked_pair"
    io.separate(out, evaluation, data.path, data.old, parent.work, hjd, direct, work / "runs.local")
    identity = {"protocol_sha256": io.digest(protocol), "pair_sha256": io.digest(pair),
                "evaluation_ledger": receipt.complete["files_sha256"],
                "hjd_contract": io.digest(hc), "direct_contract": io.digest(dc)}
    out.mkdir(parents=True, exist_ok=True)
    with io.single_writer(out):
        io.write(out / "contract.json", identity)
        if (out / "COMPLETE.json").exists():
            io.verify_finish(out, "coefficient_comparison_completed")
            print("Existing comparison verified; no model inference.")
            return out
        rows = []
        for j, source_row in enumerate(data.val, 1):
            sid = source_row["sample_id"]; item = data.load(sid, allow_validation=True)
            prefix = "records.local/" + sid + "/"
            meta = receipt.json(prefix + "record.json"); path = receipt.file(prefix + "arrays.npz")
            parent_arrays, provenance = parent.prediction(sid)
            require(meta["sample_id"] == sid and meta["case_id"] == source_row["case_id"] and
                    meta["source_arrays_sha256"] == source_row["arrays_sha256"] and
                    meta["source_record_sha256"] == source_row["record_meta_sha256"] and
                    meta["parent_prediction"] == provenance and meta["contract_sha256"] == io.digest(ec),
                    "Prediction/source pairing mismatch.")
            require(io.sha(path) == meta["arrays_sha256"], "Prediction bytes changed.")
            with np.load(path, allow_pickle=False) as z:
                predictions = {arm + "_" + role: z[arm + "_" + role + "_prediction_stored"].copy()
                               for arm in ARMS for role in ("selected", "last")}
            predictions.update(parent_wfr=parent_arrays["prediction_stored"],
                previous_final_system=item["original"]["phase10d_strict"],
                shared_calibrated_predictor=np.maximum(item["original"]["phase9g"], 0))
            for root, contract, name in [(hjd, hc, "previous_positive_negative_flows"),
                                        (direct, dc, "previous_direct_signed_flow")]:
                folder = root / "records.local" / sid; m = io.read(folder / "record.json")
                require(m["sample_id"] == sid and m["case_id"] == source_row["case_id"] and
                        m["contract_sha256"] == io.digest(contract) and
                        m["arrays_sha256"] == io.sha(folder / "arrays.npz"), "Previous saved method mismatch.")
                with np.load(folder / "arrays.npz", allow_pickle=False) as z:
                    predictions[name] = z["prediction_stored"].copy()
            for method, prediction in predictions.items():
                rows.append({"method": method, "sample_id": sid, "case_id": source_row["case_id"],
                             **score(prediction, item["original"]["target_stored"], item["factor"])})
            if j % 100 == 0:
                print(f"PAIRED SAVED-ARRAY COMPARISON {j}/600", flush=True)
        frame, cases, summary = aggregate(rows)
        summary.insert(1, "display_name", summary.method.map(NAMES))
        for arm in ARMS:
            if pair["arms"][arm]["selected"]["kind"] == "parent_fallback":
                summary.loc[summary.method == arm + "_selected", "display_name"] += " [no eligible update; parent retained]"
        primary = summary[~summary.method.str.endswith("_last")].copy()
        summary.to_csv(out / "all_results.csv", index=False)
        primary.to_csv(out / "comparison.csv", index=False)
        summary[summary.method.str.endswith("_last")].to_csv(out / "fixed_budget_last.csv", index=False)
        frame.to_csv(out / "paired_records.private.csv", index=False)
        cases.to_csv(out / "case_metrics.private.csv", index=False)
        metrics = [k for k in cases.columns if k not in ("method", "case_id")]
        deltas = []
        for method in ("composition_selected", "profile_selected"):
            left = cases[cases.method == method].set_index("case_id")
            for baseline in ("parent_wfr", "previous_final_system", "previous_positive_negative_flows",
                             "previous_direct_signed_flow", "composition_selected"):
                if method == baseline:
                    continue
                right = cases[cases.method == baseline].set_index("case_id")
                require(set(left.index) == set(right.index), "Case sets differ.")
                for case in left.index:
                    deltas.append({"method": method, "baseline": baseline, "case_id": case,
                                   **{k + "_delta": float(left.loc[case, k]-right.loc[case, k]) for k in metrics}})
        pd.DataFrame(deltas).to_csv(out / "paired_case_deltas.private.csv", index=False)
        budgets = []
        for arm in ARMS:
            s = pair["arms"][arm]["summary"]
            budgets.append({"method": arm + "_selected", "parent_optimizer_updates": 384,
                            "additional_updates_executed": s["additional_updates_executed"],
                            "selected_additional_step": s["selected_additional_step"],
                            "coefficient_parameters_optimized": s["coefficient_parameters"],
                            "parent_transport_frozen": True, "same_batch_schedule": True,
                            "training_records": 192, "monitor_records": 40,
                            "cached_transport_cost_excluded_from_head_timing": True,
                            "head_training_seconds": s["training_seconds_accumulated"],
                            "selection_status": s["status"]})
        pd.DataFrame(budgets).to_csv(out / "compute_scope.csv", index=False)
        write_report(out, primary, pair)
        parent.recheck(); receipt.recheck()
        io.finish(out, "coefficient_comparison_completed", n_records=600, n_cases=2,
                  n_primary_methods=len(primary), n_secondary_methods=2, test_arrays_opened=0,
                  independent_test=False)
        print(primary.to_string(index=False), flush=True)
        print("REPORT:", out / "report.md", flush=True)
        return out


def write_report(out, summary, pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out); folder = out / "figures"; folder.mkdir(exist_ok=True)
    specs = [("rmse_stored_units", "01_whole_cube_rmse", "Whole-cube dose RMSE"),
             ("x_mean_pct", "02_x_profile_percentage", "Array-x profile mean percentage error"),
             ("x_profile_rmse_stored_units", "03_x_profile_absolute", "Array-x profile absolute RMSE")]
    for metric, filename, title in specs:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        names = summary.display_name.str.replace(" [no eligible update; parent retained]", " [parent retained]", regex=False)
        ax.barh(names, summary[metric]); ax.invert_yaxis()
        ax.set_title(title); ax.set_xlabel("Percent" if metric.endswith("_pct") else "Stored dose units (not verified Gy)")
        maxval = float(summary[metric].max())
        ax.set_xlim(0, maxval*1.19)
        for index, value in enumerate(summary[metric]):
            ax.text(value+maxval*.008, index, f"{value:.6g}", va="center", fontsize=9)
        fig.tight_layout(); fig.savefig(folder/(filename+".png"), dpi=150); plt.close(fig)
    lines = ["# Frozen-magnitude coefficient refinement", "",
             "The two new arms start from the same selected parent coefficient weights and use the same frozen magnitudes, data, batches, optimizer and extra update budget. They differ in one additional profile loss.", "",
             "The checkpoint rule was locked before this run. It minimizes array-x percentage error subject to declared per-case and aggregate guards. If no eligible improvement exists, the primary output retains the saved parent; that is not a successful refinement.", "",
             "These 600 cubes come from two development cases and include the 40 selection records. Historical upstream exposure remains. This is not an independent patient test, a water-reference experiment, or clinical validation.", "",
             "## Primary comparison", "", summary.to_markdown(index=False, floatfmt=".8g"), "",
             "## Selection outcomes", ""]
    for arm in ARMS:
        entry = pair["arms"][arm]
        lines.append(f"- {arm}: {entry['selected']['kind']}; selected additional step {entry['selected']['step']}.")
    lines += ["", "## How to read this report", "",
              "Read whole-volume and profile changes together. No scalar success score hides a trade-off. The fixed-budget last checkpoints are in fixed_budget_last.csv as predeclared secondary results, including failed gates. They must not replace the selected rows after inspecting validation.", "",
              "The old Practical baseline is the unchanged saved Phase10D-strict output. Old positive/negative and direct signed-flow outputs are also unchanged. Metrics are recomputed from serialized predictions using the same implementation.", "",
              "Array axes are indexing conventions, not verified beam directions. All coefficients are bounded signed correction factors, not calibrated class probabilities. Neither oracle predictions nor oracle labels enter this table.", "",
              "The current study changes the coefficient objective and selection policy while preserving WFR transport. A/B attribution concerns the added profile term; improvements versus older systems do not isolate WFR geometry itself.", ""]
    for _, filename, _ in specs:
        lines.append(f"![{filename}](figures/{filename}.png)\n")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
