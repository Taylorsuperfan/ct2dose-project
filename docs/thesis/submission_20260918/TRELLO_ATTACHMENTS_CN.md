# 给已修改好的卡片添加图片

卡片文字已经完成，不要再重复新建或重写。下表依据本次读取到的看板标题与编号。发布任务现在是#106，训练暴露任务是#113，后续设计是#58；不要套用旧草稿中的#109/#107映射。

只使用 `docs/thesis/meeting_20260918/figures/` 中的新版完整名称图片。旧图和新图数值相同，但旧版只有内部缩写，不适合作为本次主要说明。

| List | 当前卡片 | 建议附件 |
|---|---|---|
| Done | #105 恢复原最终系统 | 不必配当前新方法对比图；引用已保存的恢复说明/历史汇总差值 |
| Done | #110 训练两种修正方法 | 图07（参数规模），并注明它不证明训练收敛或速度 |
| Done | #111 比较整体剂量与沿线剂量误差 | 必选图05、06；补充01和03时成对加入，04可补三轴绝对误差 |
| Done | #112 整理可读比较图 | 七张PNG全集；SVG保存在仓库，无需同卡再上传一遍 |
| Doing | #106 源码与报告发布 | 实际commit链接和会议README链接；无须反复贴图片 |
| Doing | #113 历史训练数据暴露 | 不附性能图，图不能证明患者独立性 |
| To Do | #58 下一轮profile目标设计 | 可选图05，必须标为当前结果提供的动机，不是新一轮结果 |

最低操作量：#111加05/06；#110可加07；#112作为完整图集；其余用文字和实际链接即可。

## 图片与意义

| 文件 | 解释 |
|---|---|
| 01_whole_cube_squared_error.png | 各方法整个立方体的root-mean-square error（RMSE）；不是MSE |
| 02_whole_cube_absolute_error.png | 整个立方体的mean absolute error |
| 03_x_dose_line_percentage_error.png | 原数组x方向剂量线的平均百分比误差 |
| 04_dose_line_errors.png | 三个原数组方向的绝对剂量线RMSE |
| 05_changes_from_previous_system.png | 每个指标相对原最终系统的百分比变化；负为改善，正为退步 |
| 06_volume_and_profile_accuracy.png | 横轴全局误差、纵轴x方向误差；左下更好，坐标明确放大 |
| 07_correction_model_size.png | 只比较新增修正网络的可训练参数，不含共用上游，不表示速度 |

## Trello操作

先解压附件包，不是上传ZIP后要求导师解压。
打开卡片 → Add → Attachment → 选择本机PNG → Insert。
在卡片评论中粘贴下面相应文字。需要封面时选Cover；建议#111使用06、#112使用05，采用保留标题和卡片信息的半封面。图表读数请打开完整附件，封面可能裁切，不是正式读数界面。
从电脑上传会在Trello生成一份副本；确认图表共享权限。不要直接粘贴sandbox:/下载链接、file://链接或Colab本地路径给导师。
不要把训练日志warning截图、旧缩写图例、患者图像、case identifiers、模型权重或可推断未完成训练已经完成的图附上。

## #105 — Restore the final dose-prediction system from the master practical

可粘贴评论：

```text
Recovery evidence is provided by the saved inspection and historical aggregate comparison. The new-model comparison figures are not evidence of exact historical reproduction.
```

## #110 — Train two approaches for correcting the existing dose predictions

建议附件：`07_correction_model_size.png`。

可粘贴评论：

```text
Trainable parameters of the two newly trained correction models: 340,545 for directly learned signed corrections and 104,144 for separate positive/negative corrections. Both methods use the same frozen upstream predictor. This comparison excludes the upstream parameters and does not measure training completion, inference speed or prediction quality.
```

## #111 — Compare dose accuracy: whole-volume gains versus dose-profile regressions

建议附件：`05_changes_from_previous_system.png`, `06_volume_and_profile_accuracy.png`。

可粘贴评论：

```text
The two figures summarize the completed comparison on the same 600 validation cubes from two cases. In figure 05, changes are calculated relative to the previous final system separately for each metric: negative means lower error; positive means higher error. These are relative percentages, not percentage-point differences. In figure 06, lower and further left is better. Both new methods improve whole-cube error but worsen the main x-direction dose-profile result. These are development results, not an independent final test; no uncertainty intervals were estimated.
```

## #112 — Prepare comparison figures that explain the methods without reading the code

建议附件：`01_whole_cube_squared_error.png`, `02_whole_cube_absolute_error.png`, `03_x_dose_line_percentage_error.png`, `04_dose_line_errors.png`, `05_changes_from_previous_system.png`, `06_volume_and_profile_accuracy.png`, `07_correction_model_size.png`。

可粘贴评论：

```text
This gallery contains the seven descriptive-name figures generated from the reported aggregate result table. They show the whole-cube and dose-line measures together. Only presentation names and captions changed; numerical inputs and trained models were preserved. Exact model identifiers, metric definitions and limitations are explained in the linked experiment overview.
```

## #106 — Review and publish the experiment code, figures and result summary

可粘贴评论：

```text
After publication, add the actual commit URL and the reader-facing experiment overview. Move this card to Done only after source review, push verification and attachment checks. Generating a ZIP is not publication.
```

## #113 — Document which data the previous models encountered during development

可粘贴评论：

```text
Aggregate performance figures cannot resolve historical training exposure or verified patient independence. Keep this card focused on the available lineage evidence and documented limitations, without attaching case/patient identifiers.
```

## #58 — Design the next experiment to improve dose profiles without losing overall accuracy

可粘贴评论：

```text
Motivation from the completed experiment, NOT a result of the planned experiment. The next design will align reconstructed-dose profile supervision and saved-model selection while preserving the completed experiment unchanged. No new performance improvement is claimed.
```

## 核对记录

这些图片复用上一份已生成的汇总图，不重新训练或推理；输入来自用户已报告高精度汇总，不是本次从Drive下载的独立CSV。当前600条来自两个开发case，包含40条选模记录；使用存储数值单位，数组方向未验证为物理beam方向。图片不能制造显著性或独立患者证明。

参考界面说明：
- Atlassian, Add an attachment to a card: https://support.atlassian.com/trello/docs/adding-attachments-to-cards/
- Atlassian, Add a card cover: https://support.atlassian.com/trello/docs/what-is-a-card-cover
