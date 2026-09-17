# V1 可视化更新（不修改实验）

这是 `20260917_real_pilot_v1` 之后的新增可视化层。原两套训练/恢复源码、旧报告、旧registry和交接hash均不覆盖。

**[直接打开7图图廊](gallery/README.md)**

核心图：
- [全局RMSE](gallery/figures/01_volume_rmse.png)
- [x平均百分误差](gallery/figures/03_x_percentage_error.png)
- [相对旧系统的改善与退步](gallery/figures/05_relative_changes_vs_phase10d.png)
- [全局/profile取舍](gallery/figures/06_global_profile_tradeoff.png)

绘图代码：`research/phase9g_v1_visuals/pg9viz.py`。图像输入是用户已报告高精度汇总的冻结快照；来源记录见gallery/source_provenance.json。不是再次访问Drive或重新推理；Colab V1–V3用于用户侧核对实际CSV后另存图像。

不得称为masked-water、独立测试、临床验证或统计显著胜出。当前仅模型级均值，没有逐记录分布或重复seed的不确定性估计。

当前状态：源码与快照图已生成并测试。Drive核对、实际Trello编辑和GitHub上传尚需用户执行。
