# 可视化决策追加 — 2026-09-17

Observation: V1已完成，全部模型级高精度指标已提供；RMSE/MAE改善但所有方向绝对profile RMSE相对Phase10D略升。
Decision: 添加只读绘图层，不修改V1源码、不重训、不重新选模。保持固定模型顺序、明确单位与参考方法。采用7张单坐标轴图，PNG+SVG。
Interpretation: 相对变化图和散点图帮助展示取舍，不能制造整体赢家或显著性。
Limitations: 快照来自用户控制台汇总，非本次模型重评。两个开发case；seed17；monitor40包含在val600；没有独立误差棒依据。
Next: Colab核对实际CSV；人工检查图注；按权限发布源码/汇总；把真实commit填写Trello。V2仍先设计，不宣称训练完成。
