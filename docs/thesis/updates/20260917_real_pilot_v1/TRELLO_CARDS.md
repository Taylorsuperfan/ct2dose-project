# Trello 操作与可复制内容

本文件是草稿；没有执行Trello写操作。Done表示任务执行及记录完成，不表示方法全面胜出。

## 已有卡片调整

- #105: 保留旧数值审计/fixture历史记录。
- #106: 原synthetic extension集成任务已完成，可加结项评论后移Done；不改写成真实试验。
- #108: 五组synthetic执行已完成，可加范围说明后移Done；保留synthetic身份。
- #107: 原标题是完整baseline lineage认证；仍保留Doing，说明“可运行旧系统已恢复，完整历史/患者独立性未认证”，链接新D1卡；不要将该历史独立性任务直接全部勾Done。
- #109: 使用下方W1更新标题/说明并移Doing；提交push/hash验证后才Done。

## 原卡片追加评论（不删除历史说明）

#106/#108：Execution completed for the historical synthetic pilot. Retained as software/method evidence only. The real Phase9G residual V1 is recorded in separate REAL-V1 cards; no synthetic values enter its real-data comparison.

#107：Runnable Phase10D-strict recovery and val600 aggregate consistency are completed and recorded separately. This does not certify full historical training exposure, patient grouping or blind final-test eligibility. Retain the unresolved lineage scope.


## D1 — Done

标题：
```text
[REAL-V1] Recover Phase10D-strict and verify val600
```

Description：
```text
Scope: recover the original final Practical system, not a Phase3 replay.
Completed: original definitions recovered; RF base, Phase9D-plus and Phase10D-strict weights strictly loaded. Reused original Phase9G calibration and sampler. Smoke4 and 600 real validation records evaluated; no new test-array evaluation.
Evidence: old run01_val600 report and inspection/source hashes. Maximum aggregate historical difference: 0.0007223514 percentage points.
Limit: aggregate consistency is not bitwise/record-order identity or fully blind end-to-end certification. Historical upstream exposure remains disclosed.
Done means runnable recovery and documented validation check, not proof of clinical readiness.
```

Checklist（单独粘贴到 Add an item，一行一项）：
```text
Record source and checkpoint identity
Preserve old val600 report and historical delta
State upstream exposure and axis/unit limitations
```


## D2 — Done

标题：
```text
[REAL-V1] Train ordinary residual RF and spatial HJD
```

Description：
```text
Target: r = real GT - frozen Phase9G prediction; NOT water dose or a new artificial reference. Old Phase9G/Phase10D weights remain unchanged.
Completed: train-only scaling; 192 real training records across six cases; all selected records have both residual signs; HJD target reconstruction and target overlap errors were zero. Separate 12-record learnability checks, then fresh pilot initialization.
Both pilots: seed17, batch2, 384 optimizer updates. Best model selected by full-volume RMSE on the same40 validation records. RF:340545 parameters; HJD:104144. Correction Euler8; HJD4096 particles per branch. Best monitor epochs: RF4, HJD9.
Evidence: pilot_seed17_residual_rf / pilot_seed17_hjd_rf histories, best/last pointers and checkpoints in private Drive.
Limits: small pilot, not all12000 train records; no exact CUDA-determinism or final-test claim.
```

Checklist（单独粘贴到 Add an item，一行一项）：
```text
Keep cache identity and train-only normalization
Keep both complete run folders and checkpoint pointers
Record actual budgets and validation-only selection
```


## D3 — Done

标题：
```text
[REAL-V1] Compare on val600 and document global/profile trade-off
```

Description：
```text
Completed: both new methods compared with identical frozen old predictions on600 records from two validation cases; high-precision profile review completed.
Versus Phase10D-strict:
- ordinary RF: volume RMSE -4.20%, MAE -7.43%; x-profile absolute RMSE +3.45%; x mean error6.674168% ->7.895002% (+1.220834pp).
- HJD: volume RMSE -3.47%, MAE -4.99%; x-profile absolute RMSE +2.34%; x mean error6.674168% ->7.268619% (+0.594450pp).
Both new models have slightly higher x/y/z absolute profile RMSE. HJD reduces x regression vs ordinary RF but has not surpassed Phase10D's x result. Not solely a percentage-denominator effect; failure location/cause is not established.
Evidence: V1 report, comparison and paired per-case deltas, plus docs/thesis/updates/20260917_real_pilot_v1.
Limits: one seed, two cases; monitor40 overlaps val600; inherited exposure; legacy array axes, not verified beam/mm; stored units, not certified Gy. Development evidence only. V1 frozen.
```

Checklist（单独粘贴到 Add an item，一行一项）：
```text
Keep exact aggregate and paired per-case results
Report both improved global error and worsened profile
Freeze V1; do not overwrite checkpoint selection
```


## W1 — Doing

标题：
```text
[REAL-V1] Publish reviewed source and aggregate result note
```

Description：
```text
Update existing publication card #109 rather than creating a duplicate.
Scope: submit research/phase10d_recovered_comparison, research/phase9g_signed_real_v1 and the new 20260917 result note. Keep source releases byte-identical; prior status-at-delivery records remain historical.
Acceptance: verify local working tree and remote (`github`=GitHub, `origin`=GitLab); check proposed import for conflicts; verify hashes and inspect all staged content; no medical arrays, weights, case manifests, private reports, tokens or executed notebook outputs; commit/push on thesis/phase9g-real-residual-pilot; verify local and remote commit IDs; paste actual commit link here.
Status: user action pending. This handoff does not push GitHub or modify Trello. Close only after publication is verified. Confirm permission before making research public.
```

Checklist（单独粘贴到 Add an item，一行一项）：
```text
Import exact source and result-note files without overwriting
Review file list, permissions and staged diff
Commit and push to github; verify remote hash
Attach actual commit link and reviewed result-note path
```


## P1 — To Do

标题：
```text
[REAL-V2] Predefine profile-aware objective and checkpoint selection
```

Description：
```text
Planned only; no V2 run yet.
Question: can reconstructed-dose profile supervision and aligned checkpoint selection retain global improvements without x-profile regression?
Before training: preserve V1; declare actual array axes and GT-peak training/evaluation convention; specify primary x absolute/percentage metric, global/lateral guardrails, loss weights and tolerances; apply a comparable protocol to ordinary residual RF and HJD; record endpoint-rollout cost, seeds and update budget.
Do not choose weights/checkpoints after looking at val600 to declare V1 a winner. Evaluate final dose, not log of signed residual. Keep the same learned-reference task unless a separately agreed target is documented.
Development data remains development data. Independent final-test eligibility and physical water-reference definition are not solved by this task.
```

Checklist（单独粘贴到 Add an item，一行一项）：
```text
Write V2 protocol and acceptance criteria before running
Define axes, losses, selection and no-worse tolerances
Preserve V1 and predeclare training budget/seeds
```
