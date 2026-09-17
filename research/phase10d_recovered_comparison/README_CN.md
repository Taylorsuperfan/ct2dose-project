# Phase10D-strict：已找到原源码的恢复与统一比较

## 这次结论
上传的 `ct2dose_phase9g_final_figures_seed42 (4)(1).ipynb` 有145个物理单元格，内部不止Phase9G：
- 第63格：`FalloffAwareBoundedRefineHead3D`；
- 第64–66格：Phase10D损失、预测包装、训练/评价函数；
- 第67–68格：明确的Phase10D-strict训练、best保存/加载；
- 第94/96格：strict配置、切换strict loaders。
原文件是交互式追加/重跑形成的，严格配置在页面顺序中晚于训练块。因此不要Run All原Notebook。

本包逐字抽取30个定义并记录hash；没有从指标倒推或猜一个能加载的网络。
尚未执行你的实际checkpoint加载和医学数组推理；Colab运行后才产生这些证据。
原训练初始化、操作顺序、上游暴露历史不因抽取源码自动得到证明。

## 实际恢复链
Phase9D-plus checkpoint的`base_checkpoint`指向：
`ct2dose_phase5e_along_falloff_from_phase5bplus_seed42_lr5e6_ep3_g0p02_af0p01_sl0p005_best.pt`。
它不是Notebook开头临时填写的`phase5e_strong...`。
链为：该Phase5E RF -> Phase9D-plus -> prediction-based core scalar 0.990 -> Phase10D-strict。
三份权重全部严格加载；以Phase9D-plus和Phase10D配置一致的10步为准，不沿用最早的30步。

## comparison含义
默认实际比较：
1. `phase5e_rf`：旧最终系统所依赖的RF base；不是Phase3 Optuna。
2. `phase9d_plus`：乘性/加性head后。
3. `phase9g`：再加0.990 core calibration。
4. `phase10d_strict`：最终strict refinement head后。

这是同一真实validation cohort的旧系统组件比较与恢复验证，不是已经完成新的HJD与旧系统比较。
`external.produce_external` + `compare-external`提供完整的新方法预测接入和统一打分代码：
必须同一cohort和CT hash，输出绝对dose，数值单位与原target文件一致，提供训练来源及源码/权重hash。
不把synthetic HJD结果混进真实数据表；不制造水参考，不自动用旧模型预测替换导师的reference定义。

## 快速步骤
打开 `notebooks/phase10d_recovered_colab.ipynb`，按S0–S6。
无需旧Friday工具、last_result、任务注册或反复扫描目录。Notebook自带代码。
S0恢复源码和测试；S1挂载路径；S2严格绑定三份旧权重；S3仅4条real validation smoke；
S4明确开启后评价原保存的600条validation subset；S5查看新旧历史摘要差值和图。
不读取test JSON或test arrays。不能仅因名字strict就声称整个旧系统未见过validation。

## 原行为必须保留
- CT归一化：若min<-10或max>10，`clip((ct+1024)/2048,0,1)`；否则不变。
- Dose：原函数按`max<0.1`决定乘以checkpoint dose_scale（记录为1000）。保留并逐记录记factor，不当作可靠物理单位识别器。
- 原采样：CT初始化；每步time为`(i+0.5)/10`，state用Euler式更新并clip>=0。这不是RK2中点法。
- Phase10C/10D feature及training loss的x是array D；报告提取x是array W，且训练取几何中心、报告取GT峰值直线。保留原逻辑并标注，不静默修正。
- profile error是GT峰值线上有效点的local percentage error，阈值>=该条GT profile峰值1%，eps=1e-8在model scale。
- 保存的数字单位称stored/model scale，不臆称Gy/mm；array方向不自动等于已验证beam方向。

## 文件
`p10recover/legacy_defs.py`：原定义；`pipeline.py`：明确绑定checkpoint/config；
`data.py`：只读validation划分；`evaluate.py`：可逐记录恢复的统一推理/评价；
`external.py`：真实新方法预测的接入/共同cohort评分；`tests/`：隔离测试；
`records/`：源码来源、历史匿名聚合数值、决策记录；`docs/`：操作与方法说明。

## 科学边界
不声称clinical readiness；无人工reference实验；无训练命令；旧checkpoint保持不动。
历史CSV只作为ARCHIVED参考，不能当成新推理结果。aggregate接近不能唯一证明所有historical records完全相同。
