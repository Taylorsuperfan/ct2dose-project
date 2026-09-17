# Trello cards for a reader who has not seen the code

草稿：只改面向读者的表述，未自动修改看板。按标题查找同一任务后更新原卡，不重复创建。已完成指执行/记录完成，不意味着方法整体胜出。下面每张卡都能独立阅读；内部代码标识放在METHOD_NAMES_CN_EN.md，不用它们代替研究任务标题。

建议：已有发布卡#109对应“Review and publish...”；已有历史数据暴露卡#107对应“Document which data...”。这些是对先前映射的延续，不是本次对看板状态的重新核验。历史synthetic卡保留原身份，不能改成这次真实比较。


# Done


## Restore the final dose-prediction system from the master practical

**Description**

Purpose
Re-establish the previous final system as the reference for this study, rather than replacing it with a newly written approximation.

Completed work
Recovered the original model and inference definitions, matched the saved weights to all required components, and ran the complete system on 600 validation cubes from two cases. Original preprocessing, sampling and calibration were retained.

Result
The largest difference between the newly calculated and archived mean dose-profile errors was approximately 0.000722 percentage points. This is agreement of aggregate results, not proof that every historical voxel or record ordering is identical.

Limitations
The earlier training history of the reused components still needs to be reported. This recovery does not establish an independent final test or clinical validity. No new final-test arrays were evaluated.

Technical traceability
Original final-system identifier: phase10d_strict. Exact source, weights and execution evidence remain in the existing recovery records.

**Checklist**

- [ ] Recovered definitions and saved weights are linked to the correct components
- [ ] Completed validation results and historical differences are retained
- [ ] Aggregate agreement is distinguished from exact historical identity
- [ ] Remaining historical training-exposure limits are recorded


## Train two approaches for correcting the existing dose predictions

**Description**

Research question
Can learning the remaining prediction error improve an existing computed-tomography-to-dose predictor?

Target
For each paired training sample, subtract the frozen calibrated prediction from the ground-truth dose. The resulting correction may be positive or negative. The starting prediction is learned from data; it is not a dose-to-water field.

Approach 1: directly learned signed correction
Use rectified flow matching to predict one correction field containing both positive and negative values, then add it to the same fixed starting prediction.

Approach 2: separate positive/negative corrections
Use Hahn–Jordan decomposition to split the correction into two nonnegative components. Learn two spatial flow fields and each component's total magnitude, subtract the negative component from the positive one, and add the result to the fixed starting prediction. Include a loss discouraging unnecessary overlap.

Completed training
Used 192 cubes from six training cases, 384 optimizer updates per approach, batch size two and one seed. Selected the saved weights using whole-cube root-mean-square dose error on 40 validation cubes. Separate small-data checks did not supply trained weights to the main runs.

Records
Saved training settings, source versions, model weights, optimizer state, random-generator state and logs. This is a small development experiment, not full-data training.

**Checklist**

- [ ] Both approaches use the same fixed calibrated predictor and training records
- [ ] Data scaling was fitted on selected training cubes only
- [ ] Positive and negative correction targets reconstruct the original residual
- [ ] Both training runs completed and their states/logs are retained
- [ ] Selection data and actual optimizer-update budget are documented


## Compare dose accuracy: whole-volume gains versus dose-profile regressions

**Description**

Completed comparison
Evaluated the two newly trained correction approaches and the previous system on the same 600 validation cubes from two cases, using unchanged dose metrics and line-profile definitions.

Relative to the previous final system
Directly learned signed correction: whole-cube root-mean-square error decreased by 4.20%, while absolute x-profile root-mean-square error increased by 3.45%.
Separate positive/negative corrections: whole-cube root-mean-square error decreased by 3.47%, while absolute x-profile root-mean-square error increased by 2.34%.

The mean x-profile percentage error was 6.674% for the previous final system, 7.895% for the new direct approach and 7.269% for the new separated approach.

Interpretation
Both approaches improve overall dose error but have not improved the previous system's main x-profile result. The separated approach mitigates the x regression relative to the direct approach. All three absolute line-profile errors are slightly higher for both new approaches than for the previous final system, so there is no overall superiority claim.

Limits
A dose profile is an array-aligned line through the target-dose peak, not an independently verified physical beam direction. The 40 model-selection cubes are part of the 600 validation cubes. One training seed and two development cases do not establish significance or patient-level generalization. The physical location and sole cause of the regression remain unresolved.

**Checklist**

- [ ] Same600 validation cubes and metric definitions used for all methods
- [ ] Full-precision volume and profile errors reviewed together
- [ ] Case-level changes retained privately without treating cubes as independent patients
- [ ] Improvements and regressions both included in the conclusion
- [ ] Completed results preserved without retrospective model reselection


## Prepare comparison figures that explain the methods without reading the code

**Description**

Completed presentation work
Generated seven numerical comparison figures from the reported aggregate result table. The figures now use descriptive method names: the previous final dose-prediction system, the shared calibrated predictor, directly learned signed corrections, and separately learned positive/negative corrections.

Figures
Whole-cube root-mean-square error; mean absolute error; x-profile percentage error; absolute profile errors in three array directions; relative changes from the previous final system; whole-volume/profile trade-off; and trainable correction-network size.

Accuracy of presentation
Model identifiers and version tags are explained in a separate code-to-name table. Captions define the error measures, numerical units and validation scope. Bars start at zero; the scatter plot is explicitly zoomed. No error bars, confidence intervals or significance results were invented.

Scope
Only labels, captions and presentation documents changed. The numerical input table, trained models and original results were not changed. Generated figures are based on the supplied aggregate snapshot, not new model inference. Uploading them to Trello or GitHub is a separate task.

**Checklist**

- [ ] All seven figure pairs generated with descriptive method names
- [ ] Metric definitions and shared-model relationships explained
- [ ] Original numerical table copied without alteration
- [ ] Both performance improvements and profile regressions shown
- [ ] Internal identifiers retained only for technical traceability


# Doing


## Review and publish the experiment code, figures and result summary

**Description**

Purpose
Make the completed experiment understandable and reproducible for a reader who has not followed the development notebooks.

Remaining work
Read the revised method overview and figure captions; compare the plots with the saved result table; review the files to be committed; confirm sharing permission; publish the code and approved aggregate figures; then record the actual commit link in this card.

Preserve
Existing training code, evaluation rules, saved weights, numerical results and historical version records. Descriptive display names must not rename program identifiers or modify the experiment.

Exclude from the repository and card attachments
Medical arrays, trained weights, prediction caches, patient/case manifests, private record-level reports, credentials and old notebooks with saved outputs. Aggregate figures also require appropriate sharing permission.

Completion
Move this card to Done only after the files have been reviewed, the remote commit is verified and the intended figures/links are attached. No publication action has been performed by generating these drafts.

**Checklist**

- [ ] Reader-facing method names and captions checked against the original identifiers
- [ ] Actual result-table values and all figure units reviewed
- [ ] Only intended source/documents/aggregate figures staged
- [ ] No private medical data, credentials or trained weights included
- [ ] Actual remote commit and figure links recorded on the card


## Document which data the previous models encountered during development

**Description**

Why this remains separate
The previous system has been restored and evaluated successfully. That does not answer whether all of its reused components avoided the present validation cases throughout earlier training and tuning.

Remaining evidence
Record the training and model-selection data used by the reused components. Distinguish case grouping from verified patient grouping. State which information is known and which remains unavailable.

Interpretation
The current comparison is development evidence. A corrected split for the final refinement head does not make the entire historical system a fully independent test.

Completion
Resolve the outstanding questions where evidence is available, and clearly record any acknowledged limitations. Do not erase past exposure or classify the completed system recovery as failed. This task does not require another model search or training run merely to update the card.

**Checklist**

- [ ] Training and selection exposure of reused components documented where available
- [ ] Case independence and patient independence reported separately
- [ ] Unavailable historical information explicitly marked
- [ ] Current comparison consistently described as development evidence


# To Do


## Design the next experiment to improve dose profiles without losing overall accuracy

**Description**

Research question
Can direct signed-correction learning and separate positive/negative correction learning retain their whole-volume improvements while avoiding the x-profile regression?

Before training
Define the actual array directions and the line-profile extraction rule. Predefine the main profile metric, acceptable changes in whole-volume and other-direction errors, loss weights, saved-model selection rule, data, seeds and compute budget.

Method comparison
Use comparable final-dose profile supervision for both approaches. Account for the cost of integrating the learned flow when computing endpoint losses. Apply profile terms to reconstructed dose, not the logarithm of a signed correction.

Status
Planned only. No new training or improved result is claimed. Keep the completed experiment and its results unchanged. The missing physical water-reference definition and independent final-test eligibility remain separate issues.

**Checklist**

- [ ] Profile objective and array-axis conventions agreed before training
- [ ] Allowed volume and other-profile changes defined
- [ ] Comparable loss and saved-model selection rules set for both approaches
- [ ] Data, seeds, training budget and uncertainty plan recorded
- [ ] Completed experiment preserved and new run clearly separated
