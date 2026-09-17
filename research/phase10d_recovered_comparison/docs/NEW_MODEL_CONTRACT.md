# 新方法如何与原Phase10D比较

本包没有训练新HJD，也没有重新定义reference。已有的新模型必须在相同真实cohort输出绝对dose。

## Python接口
```python
from p10recover.external import produce_external, compare_external

# your_predictor(raw_ct) must use only legitimate inference-time information.
# It returns absolute dose in the same numerical units as the stored target .npy.
# For a residual model, add its justified reference INSIDE the model wrapper.
# This is not permission to construct a reference from validation ground truth.
manifest = produce_external(
    baseline_run=VAL600_OUT,
    out=OUT_ROOT / "new_method_predictions.local",
    method="HJD_real_v1",
    predict_native=your_predictor,
    training_provenance="Record actual training/validation cases, seed and selection protocol here",
    producer_source_sha256=your_verified_source_hash,
    checkpoint_sha256=your_verified_checkpoint_hash,
)
compare_external(
    VAL600_OUT, manifest, OUT_ROOT / "old_vs_new_comparison.local"
)
```
以上`your_*`是待接入的真实模型对象，不是本包偷偷创建的模型。没有它们就不执行此部分。
所有legacy比较代码均可独立运行。

## 已有prediction文件的CLI
`python -m p10recover compare-external --baseline-run PATH --predictions predictions.local.json --out NEW_PATH`

JSON要求：`method`, `data_scope=real_validation`, `cohort_sha256`, `prediction_kind=absolute_dose`,
`units=stored_target_units`, `training_provenance`, `producer_source_sha256`, `checkpoint_sha256`, `records`。
每条有`sample_id`, `ct_sha256`, `prediction_path`, `prediction_sha256`。
cohort模板在旧比较结果的`cohort_for_external_predictions.local.json`中。
必须每个样本恰好一条；错误cohort、CT hash、缺失记录、错误数值单位声明会被拒绝。
声明不是证明：仍需审阅生产代码、split和训练暴露。不可把synthetic结果标real骗过检查。
新模型即使允许负dose，评分不自动裁剪；报告negative fraction，避免改变模型输出掩盖问题。

## 比较语义
新模型vs原Phase10D是系统比较。普通residual RF vs HJD应使用相同合法reference/target。
HJD无/有overlap是内部消融。绝不能将最近synthetic HJD的RMSE与此处real RMSE横向排名。
