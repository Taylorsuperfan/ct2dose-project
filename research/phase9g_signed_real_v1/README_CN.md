# 真实 Phase9G 残差修正：普通 RF 与 HJD

**本轮定义：真实 GT − 已冻结的 Phase9G。不是 water reference、物理输运或盲 final-test 实验。**
用户已确认先试这项受控实验。原始导师目标中的 masked dose-to-water 不被重命名或替代。

打开 `notebooks/phase9g_signed_real_colab.ipynb`，依次 P0–P7。Notebook 已内嵌完整运行代码；ZIP 用于本地审阅和提交源码。

## 不重做的事
不搜索旧源码，不更换旧权重，不重跑旧 600 条网络推理，不生成 synthetic/reference 数据，不打开 test JSON 或 test arrays。
`p10recover/` 与已恢复并评价的上一版逐字节一致。现有旧结果只读取。

## 本轮比较
- frozen Phase9G：不修正；直接复用旧预测。
- original Phase10D-strict：原 Practical 最终系统；直接复用旧预测。
- ordinary signed residual RF：新训练；CT、Phase9G 为条件。
- HJD spatial two-flow + learned masses：新训练；同样的真实目标与条件。
旧 Phase5E/Phase9D-plus 结果保留在大表中作为历史系统组件，不重新训练。

## 第一轮明确是 pilot
按记录标识 hash 预选 **6 cases × 32 = 192 条真实 train records**，不是全部12,000条。
先做 **12条train-only、每case2条** 的可学习性检查；两个模型各128次优化更新。
再从新随机初始化开始各 **12 epochs × 32 updates、batch2 = 384 updates**。
模型选择使用val600内预先固定的40条（每case20）；最终报告仍在同一val600上。
这是开发验证，不是独立测试。原上游训练暴露仍须披露；旧10D与新模型训练预算并不相同。

## 目录
- `pg9learn/cache.py`：真实192条训练缓存、直接引用旧val600、train-only尺度。
- `pg9learn/models.py`：普通RF、HJD及全部损失。
- `pg9learn/train.py`：训练、checkpoint、resume、monitor曲线。
- `pg9learn/compare.py`：新模型600条预测、原评价、paired deltas、图表。
- `p10recover/`：原样保留的恢复代码及30个原函数/类定义。
- `tests/`：旧24项与新14项软件测试。临时测试数组不是本轮训练数据。
- `records/`：方法决定与任务模板；不包含新的真实实验结果。

本地测试不代表实际Drive/GPU训练成功，更不保证HJD优于旧系统。
