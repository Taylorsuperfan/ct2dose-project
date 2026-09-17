# 操作步骤：从已恢复旧系统开始训练真实残差

## 先固定两条路径
原数据 `/content/drive/MyDrive/rectified_flow_ct2dose`。
旧已完成结果 `/content/drive/MyDrive/ct2dose_thesis/phase10d_recovered_v1/run01_val600`。
不要把smoke4目录填成val600，不重跑旧S3/S4。

## P0：打开新Notebook，恢复完整源码
新Notebook已内嵌原样的p10recover和新增pg9learn。不需要先运行旧Friday notebook。
恢复30个原定义核验；运行14项新增测试（完整包另有原24项测试）。
不升级已有PyTorch；缺少轻量包才安装。测试使用临时小数组，不是你的医学数据或新训练结果。

## P1：挂载与路径
确认旧600条status为completed_validation_inference。新输出放在phase9g_signed_real_v1下。
当前GPU可用就使用；无需原A100才能读取缓存，但续训有环境一致性检查。
RUN_ID=pilot_seed17；CACHE_ID=train192_seed42；这两个不要复用到另一配置。

## P2：加载已恢复的冻结旧系统（仅训练缓存尚缺时）
首次将TRUST_OWN_CHECKPOINTS=True。它严格读取已知三份旧文件，配置与旧600条的contract对应。
只是为192条train产生Phase9G；不训练旧模型，也不再次评价600条旧网络。
缓存READY已存在时直接跳过装载。原定义不符时停止，不绕过source/strict检查。

## P3：准备并读取真实缓存
首次设置PREPARE_REAL_CACHE=True；以后READY存在时可保持False只读缓存。
训练从6个cases各选32条，按ID哈希，与target/误差无关；实际模型只拟合这192条，不称全12000训练。
对这192条产生固定Phase9G预测，然后读取真实GT、计算差值。
验证集引用旧600条已保存的ct/predictions/target，不重复写600份数组也不重新预测。
输出train residual audit、共同尺度、初始正负mass、源manifest/旧pipeline/source hashes。
不生成或强迫负值，不生成water；真实残差全零时停止。
Cache会将所需数组一次载入CPU RAM，避免每个batch通过Drive读小文件。
如果挂载很慢，参考进度；不要据GPU型号估算Drive读写时间，也不要同时重启第二个writer。

## P4：12条真实train-only可学习性检查
设置RUN_SMALL=True。两种新模型分别优化128步；监控仍为这些12条train，不看validation选模型。
保存每个小实验的checkpoint、history、训练曲线和training_summary。
重点看 learned_beyond_zero_correction，及best_monitor.rmse是否小于base_rmse。
False是如实的诊断结果，不是程序错误。尤其HJD没有改善时先停该分支，保留输出；不要改baseline或用oracle mass。
这一步不能证明泛化。后面pilot重新随机初始化，不从这个小实验续训。

## P5a/P5b：普通RF和HJD的正式pilot
分别设置RUN_RF_PILOT、RUN_HJD_PILOT=True。
每组192train、384updates、batch2；所有原模型冻结。
每epoch在固定40validation records（2cases各20）评价；以equal-case dose RMSE选best。
HJD为两个空间flow+learned mass+endpoint辅助loss，不是假装双通道回归是两条RF。
它可能更慢、更不准确，不能承诺胜出。loss总数值不与普通RF直接比较。
不在这个run里临时改变loss、epochs、seed、particles。修改则另起版本/run。

## P6：最终同一600条比较
两个pilot都完成后RUN_COMPARE600=True。
只推理两种新模型；旧四阶段结果直接读取已完成文件。
新模型绝对dose主指标clamp0，raw也另存；signed residual不在ODE中clamp。
保存每record/case native单位和原GT-peak profile指标、raw HJD质量、抵消、符号、越界率。
最终表包含原Phase9G、Phase10D等与两个新方法；没有方法输出就停止，不补零伪造对照。

## P7：查看报告、打包小型报告
报告中的负delta表示误差下降；profile百分比指标的差用百分点。
不要根据最好的样本或某个小数位宣布general superiority。
可打包只含小型CSV/JSON/Markdown（不含医学数组、图片、checkpoint）的结果供核对；仍按私有材料处理。

## 断线后
1. 确認旧进程不再运行，不允许两台VM同时写同一个目录。
2. P0、P1。缓存已完成时P2可跳过，P3只读已有缓存。
3. 运行同一未完成的P4/P5格；代码从已校验last.json指向的checkpoint恢复。
4. last.json/best.json是指针，真实权重位于checkpoints/step...pt；没有固定last.pt并非没保存。
5. 源码、缓存、配置或训练环境不匹配时拒绝精确续训。不要删检查；另建明确的新run或恢复相同环境。
6. 已完成的训练只核验best，不重复训练；旧evaluation环境不同时使用新的EVAL_TAG创建独立新评价。
保存间隔16updates，可能丢失最后未落盘更新；Drive读回hash不等于云端绝不丢失保证。

## 这轮完成标准
真实192train缓存+目标分解可解释；新训练有checkpoint和history；两种新方法同val600输出与旧基线paired比较；
报告标记真实旧预测残差修正，不写water reference、不宣称final test独立。
实际运行状态更新卡片；本地软件测试通过不等于新真实实验完成。
