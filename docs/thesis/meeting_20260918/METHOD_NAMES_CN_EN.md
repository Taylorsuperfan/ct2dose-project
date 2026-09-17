# 研究名称与内部代码标识的对应 / Reader-facing names and code identifiers

本文件只更改展示名称，不重命名代码、checkpoint、CSV中的method值或已完成的run。下列描述性名称不是新算法名称。需要追溯代码时查看右侧标识。

| 中文展示名称 | English display name | Original code identifier |
|---|---|---|
| 后续修正之前的剂量预测器 | Dose predictor before refinement | `phase5e_rf` |
| 加入学习到的缩放与偏移修正的预测器 | Predictor with learned scale/offset correction | `phase9d_plus` |
| 各修正方法共用的校准剂量预测器 | Shared calibrated dose predictor | `phase9g` |
| Practical阶段的最终剂量预测系统 | Previous final dose-prediction system | `phase10d_strict` |
| 新方法：直接学习正负剂量修正 | New: directly learned signed correction | `new_residual_rf` |
| 新方法：分别学习正、负修正分量 | New: separate positive/negative corrections | `new_hjd_rf` |

## 首次出现时说明，不要求读者先看代码

- **RF — Rectified Flow**：这里指整流流模型，不是随机森林。新方法使用flow matching学习修正场；旧模型也是flow-based，所以不能用“新模型 vs RF”来指代本次所有比较。
- **HJD — Hahn–Jordan decomposition**：把有正负值的修正场分成两个非负分量，再用正分量减去负分量的幅度。它不是“正剂量与负剂量”的临床分类，也不是两个分量相加。
- **V1**：本轮实验的内部版本标签，不是方法名称。标题改为“首次真实数据残差修正比较”或“Completed real-data correction experiment”；版本号可留在技术元信息。
- **Phase9G / Phase10D-strict**：内部历史阶段编号。对外用本表的功能描述；代码对应关系保留在此处。
- **GT — Ground truth**：配对数据提供的目标剂量数组。对外写“ground-truth dose”或“target dose”，不默认为临床测量。
- **Residual**：`target dose − fixed starting prediction`，即需要增加或减少的剂量修正。
- **Mass head**：预测正、负修正分量各自总量的网络。该总量是体素数值的和，不是身体质量或已验证的物理能量。
- **Overlap loss**：惩罚两个预测分量在同一位置同时有较大幅度的损失；不是已证明互斥的保证。
- **Checkpoint**：保存的模型权重和可能包含的训练状态。Trello标题可写“restore the trained model”；文件名保留原样。
- **val600**：用于本次比较的600个验证立方块，来自两个病例，不是600个患者。
- **Profile**：沿指定数组方向的一条剂量曲线。图中array x/y/z对应旧评价器的W/H/D，不自动代表物理射束方向。

## Short figure labels versus scientific names

**New: directly learned signed correction** means *rectified flow matching with direct signed-residual prediction*. This is not conventional direct regression.

**New: separate positive/negative corrections** means *rectified flow matching with Hahn–Jordan decomposition of the signed residual*, including learned component magnitudes and a penalty on component overlap. The two spatial flow fields share condition features; this description does not claim two entirely independent full networks.

**Previous final dose-prediction system** means the recovered *Phase10D-strict* system, including its fixed upstream components. “Strict” refers to the corrected refinement-head split, not full historical data independence.
