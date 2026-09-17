# V1 只读比较可视化

本目录不改变原训练源码、checkpoint或评价指标。读取已经计算好的六方法、600条/2case汇总CSV；不加载PyTorch、医学数组或checkpoint，不执行新训练和推理。

## 最短使用路径
- 直接看交接包 `docs/thesis/updates/20260917_real_pilot_v1_visuals/gallery/README.md`，已生成七张图。
- 需要在Drive实际CSV上核对并重画：Colab打开本目录 `notebooks/phase9g_v1_visuals_colab.ipynb`，依次V0–V3。

## 本地重画（独立新输出目录）
```bash
python3 -m unittest discover -s research/phase9g_v1_visuals/tests -v
python3 research/phase9g_v1_visuals/pg9viz.py --csv research/phase9g_v1_visuals/data/comparison_reported.csv --out /tmp/pg9_v1_visual_review --source-kind user_reported_aggregate
```
本地需numpy、matplotlib。不要覆盖版本包已保存的图或训练源码。

## 输入与统计约定
内嵌data/comparison_reported.csv来自用户高精度控制台汇总的既有交接快照；不是重新取得600条原始误差。本工具保留模型级必需列，不导出原CSV其他字段。
条形图从0开始；图6是明示局部放大坐标的散点图。图5是各指标独立的相对变化百分数，不是百分点，不是多指标总分。没有逐记录/多seed数据，故不生成误差棒、箱线图或p值。两个case不是600名患者。
参数图只表示新增修正网络，不表示全系统资源或推理速度。

## 存储
新输出含PNG、SVG、数据白名单CSV、相对变化表、HTML/Markdown图廊、运行环境和hash。输出已有且源/代码不同会停止。Drive只验证三个小文件，不加载医学数组。

## 实验结论
全体素RMSE下降与x/y/z绝对profile RMSE退步均显示；V1不声明全面胜出。图像不定位误差发生区，也不证明显著性。真实Phase9G误差修正，不是water-reference或盲final-test。
