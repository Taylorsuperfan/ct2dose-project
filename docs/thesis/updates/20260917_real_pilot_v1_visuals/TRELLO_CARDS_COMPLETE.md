# 完整Trello卡片：按类别复制

本文件是草稿，未自动创建、修改、移动或上传附件。Done=执行/记录完成，不是科学全面胜出。

看板：`https://trello.com/b/lUhyMDIZ/conditional-flow-matching-project`

## 已有历史卡片处理

#105保留Done的旧审计/synthetic证据。#106与#108可在追加实际结项说明后移Done，保留synthetic范围，不能用真实结果覆盖旧身份。#107用于剩余暴露/患者分组问题（下方G02），#109用于本次发布（下方G01）。#58等不相关卡片不动。

如相同真实任务卡已经手工建好，请更新原卡，不重复创建。图片从附件包中选择指定PNG；不要复制几百行日志或患者/病例标识。

# Done

## D01 — [REAL-V1] Recover Phase10D-strict and verify val600

操作：Create only if this real recovery card does not already exist.

### Title
```text
[REAL-V1] Recover Phase10D-strict and verify val600
```

### Description
```text
Scope: recover the original final Practical system, not a synthetic or Phase3 replay.

Completed: the original definitions were recovered; RF base, Phase9D-plus and Phase10D-strict states were strictly bound; original sampling/calibration behavior was retained. Smoke4 and the same 600-record real validation evaluation completed.

Historical aggregate check: maximum absolute mean-profile difference was 0.0007223514 percentage points. This is aggregate consistency, not bitwise or historical record-order identity.

Evidence stays on private Drive: inspection hashes, run01_val600, source snapshots and historical_summary_delta.csv. No new final-test arrays were evaluated. Inherited upstream-exposure and patient-independence limitations remain open; see the lineage card.

Done = runnable recovery and documented validation comparison, not clinical or fully blind certification.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
Original source and three model-state bindings recorded
Smoke4 and val600 reports saved
Historical aggregate differences retained without tuning to match
Upstream exposure and historical identity limitations disclosed
```

## D02 — [REAL-V1] Train ordinary residual RF and spatial HJD

操作：Create only if the real training card does not already exist.

### Title
```text
[REAL-V1] Train ordinary residual RF and spatial HJD
```

### Description
```text
Task: r = real GT dose - frozen Phase9G prediction. This reference is a learned old predictor, NOT water dose.

Completed: 192 real train records across six cases; train-only scaling; signed decomposition and zero-overlap target audit; separate 12-record learnability checks; fresh initialization for the two formal pilots.

Each pilot: seed17, batch2, 384 optimizer updates. Best checkpoint selected using equal-case whole-volume RMSE on 40 validation records. Correction ODE: 8 Euler steps; HJD uses 4,096 particles per branch.

Trainable correction parameters: ordinary RF 340,545; HJD 104,144. These exclude the shared frozen upstream and do not establish inference speed.

Checkpoints, optimizer/RNG state, logs, config and source identity stay on Drive. This is a small192-record pilot, not exhaustive full-data training; CUDA determinism is not guaranteed.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
192-record selection and train-only scaling recorded
Both real signed-target training runs completed
Best/last checkpoints and optimizer/RNG history retained
Small-run weights not reused in pilot
Budget and source/normalization identities recorded
```

## D03 — [REAL-V1] Complete val600 comparison and document metric trade-off

操作：Create only if the real comparison card does not already exist.

### Title
```text
[REAL-V1] Complete val600 comparison and document metric trade-off
```

### Description
```text
Compared two new corrections and four old-system stages on the same600 validation records, two cases; old predictions were reused.

Relative to Phase10D-strict:
Ordinary RF: whole-volume RMSE -4.20%, MAE -7.43%; absolute x-profile RMSE +3.45%; mean x percentage error 6.674168% -> 7.895002%.
HJD: whole-volume RMSE -3.47%, MAE -4.99%; absolute x-profile RMSE +2.34%; mean x percentage error -> 7.268619%.

Both new models have higher aggregate absolute x/y/z profile RMSE than Phase10D. HJD mitigates the x regression relative to ordinary RF but is not an overall winner. Regression is not solely a percentage-denominator effect; its spatial cause has not been isolated.

Scope: equal-case record averages; seed17; monitor40 overlaps val600. Stored numerical units and legacy array axes, not certified beam/mm/Gy. Not water, a blind final test, or significance evidence. Freeze V1; do not reselect a checkpoint to make it win.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
Both new methods evaluated on exactly600 validation records
Full-precision global and profile metrics reviewed
Per-case differences retained privately
Benefits and regressions reported together
V1 results frozen; no overall-superiority claim
```

附件：`01_volume_rmse.png`, `03_x_percentage_error.png`, `05_relative_changes_vs_phase10d.png`, `06_global_profile_tradeoff.png`

## D04 — [REAL-V1-VIS] Generate seven aggregate comparison figures

操作：Create new visualization-generation card; source/figure production is complete, Drive/publication is separate.

### Title
```text
[REAL-V1-VIS] Generate seven aggregate comparison figures
```

### Description
```text
Completed in the delivered visual package: seven figures, each PNG + SVG, generated from the user-reported high-precision V1 aggregate snapshot. No training, checkpoint loading, medical-array access or model inference.

Figures: global RMSE; global MAE; mean x percentage error; grouped x/y/z absolute profile RMSE; relative changes vs Phase10D; global/x trade-off scatter; correction-only parameter count.

Bars begin at zero. Figure06 is an explicit zoomed five-system detail; upstream Phase5E remains in figures01-04. Figure05 uses100*(new/old-1), not percentage points or a combined score. No invented error bars, boxplots or statistical significance.

Fourteen visualization software checks pass. Plot sources, approved numeric columns, SVG/PNG, figure captions and hashes are supplied. Counts of parameters are separately user-reported, not timed.

Done covers source/snapshot-figure generation only. User-side Drive CSV verification, board attachment and GitHub push remain in the publication card.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
Seven PNG/SVG pairs generated from declared aggregate input
Source provenance and numeric table retained
Whole-volume gains and profile regressions both shown
No private case IDs or medical images in the snapshot figures
Fourteen visualization software tests passed locally
```

附件：`05_relative_changes_vs_phase10d.png`, `06_global_profile_tradeoff.png`

# Doing

## G01 — [REAL-V1] Verify figures and publish reviewed source/results

操作：Update existing publication card #109; move to Doing until actually verified.

### Title
```text
[REAL-V1] Verify figures and publish reviewed source/results
```

### Description
```text
Publication scope: the unchanged Phase10D recovery and Phase9G real-training packages, existing V1 records, the new read-only visualization source, and the reviewed aggregate figure gallery.

Remaining: run visual Notebook V0-V3 against the saved Drive CSV; check values/hashes/captions; review repository conflicts and all staged files; confirm sharing permission; commit/push and verify local/remote IDs. Figure generation does not mean publication has occurred.

Use branch thesis/phase9g-real-residual-pilot. Preserve github=GitHub and origin=GitLab. Use the NEW tools/pilot_v1_visual_release/check_release.py --staged for the expanded allowlist, not the old staged-only checker.

Exclude trained weights, CT/dose/prediction arrays, case/patient manifests, private reports, credentials and executed notebook outputs. PNG/SVG allowed here are only the enumerated aggregate charts.

Attach the reviewed figure05/06 and actual commit URL after completion. Do not overwrite old releases/manifests or make V2 performance claims. Move Done only after verification.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
V0-V3 verified the saved Drive report and generated a new figure folder
Snapshot and Drive-derived chart values/captions reviewed
Local import conflicts reconciled without overwriting evidence
New combined release checker and manual staged review passed
Commit pushed to github and local/remote hashes match
Actual commit link and reviewed PNGs attached to cards
```

## G02 — [LINEAGE] Document remaining historical exposure and patient grouping

操作：Update/scope existing #107; runnable recovery already belongs to D01.

### Title
```text
[LINEAGE] Document remaining historical exposure and patient grouping
```

### Description
```text
Runnable Phase10D-strict recovery and val600 aggregate consistency are complete in the recovery card. This card concerns the separate unresolved historical evidence.

Remaining: document base/refinement training and selection exposure; identify reused parent checkpoints and their data history; retain current case-group checks and unresolved patient mapping. Do not interpret head-level strict naming as fully blind end-to-end evidence.

The present600-record comparison is development evidence. No new final-test evaluation or search/training is required merely to update this card.

Close only when the remaining lineage question is resolved with evidence, or explicitly scoped as an acknowledged limitation. Do not relabel recovery as failed and do not erase exposure history.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
Parent checkpoint/data-exposure evidence documented
Case vs patient grouping status stated separately
Final-test eligibility not inferred from strict filename
Outstanding limitations written in the experiment registry
```

# To Do

## T01 — [REAL-V2] Predefine profile-aware training and selection

操作：Create only if an equivalent V2 design card does not exist.

### Title
```text
[REAL-V2] Predefine profile-aware training and selection
```

### Description
```text
Planned; V2 training has NOT started.

Question: retain global improvements while avoiding regression in the main legacy-x profile relative to the old system.

Before training: freeze actual array-axis/profile conventions, primary x metric, global/lateral guardrails, loss coefficients, tolerances, seeds, checkpoint-selection rule and compute budget. Apply comparable final-dose profile supervision to ordinary residual RF and HJD; record differentiable endpoint cost.

Profile losses act on reconstructed dose, not log(signed residual). Keep the same frozen upstream and data policy unless a separately documented design change is agreed. Keep V1 files/results unchanged.

Do not select thresholds after seeing val600 to claim V1 wins. Development remains development; physical-reference definition and independent final-test eligibility are separate unresolved questions. No promised improvement or completed V2 claim.
```

### Checklist（每行一项；按实际完成状态勾选）
```text
Primary objective and array-axis mapping written before training
Global/lateral guardrails and tolerances fixed
Comparable loss/selection protocol fixed for both models
Seed/data/compute budget and uncertainty plan stated
V1 frozen and V2 stored as a new version
```
