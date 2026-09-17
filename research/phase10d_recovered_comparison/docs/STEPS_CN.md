# 今晚操作步骤

## S0：打开新Notebook并恢复代码
直接上传`phase10d_recovered_colab.ipynb`到Colab；其中内嵌完整运行代码，不需要另传ZIP。
不再运行旧Friday的导出工具。S0会核对内嵌代码、原定义hash和24项测试。
已有torch不升级。缺numpy/pandas/matplotlib/pytest时只安装缺少的依赖。
CPU足够做恢复和检查；实际600条推理推荐GPU，但不要求A100。

## S1：挂载Drive并设置原目录和新的输出目录
原目录 `/content/drive/MyDrive/rectified_flow_ct2dose` 必须已存在，不创建空替代。
新输出 `/content/drive/MyDrive/ct2dose_thesis/phase10d_recovered_v1`。
设置`RUN_TAG="run01"`，SMOKE_OUT和VAL600_OUT分别为两个新目录。
以后只读报告不需要重跑推理。并行进程不得使用同一个run目录。

## S2：严格加载
读完路径和来源后，只有你确认是自己的原训练文件才设置`TRUST_OWN_CHECKPOINTS=True`。
程序自动读取已知Phase9D-plus、Phase10D-strict文件，依据9D+ metadata取得精确base文件名。
`model_state_dict`逐key/shape/dtype核对，`strict=True`；不自动重命名、丢层、换父模型。
预期base 26条/763993元素；10D 18条/25697元素。实际以输出为准。
若受限加载报错，保留完整错误，不使用weights_only=False直接跳过。

## S3：4条真实validation smoke
把`RUN_SMOKE=True`，运行一次。每个当前validation case的前2条，selection在推理前冻结。
预测函数只接收CT，模型预测完成后才读取GT用于metrics。
保存四种旧链组件的预测、逐记录/逐case表、profiles、contract和源码快照。
当前train/val case IDs必须互斥，但不推断上游历史训练暴露已解决。

## S4：原600条validation subset
smoke通过且数值有限后设置`RUN_VAL600=True`。
只使用已经存在的`eval_subsets/val2_stratified_300_per_case.json`，验证它是当前val的子集且2case各300。
缺少则停止，不重新随机生成一个假装是原来的。
不执行旧Notebook中的test评价，不运行sample491（它原来来自已暴露历史test集合）。
本次进度按每条已保存记录打印；不要仅为了看到新输出重复启动进程。

## S5：查看和保存report
运行查看格，选SMOKE_OUT或VAL600_OUT。
600条时额外生成`historical_summary_delta.csv`，对照附件里已有validation摘要。
历史x：9G 7.020824 /10D 6.673486；y：2.228602/2.219377；z：2.282180/2.246739（百分误差）。
这些是报告数值，不是本次生成。必须以重新计算后的delta为准。
不通过调checkpoint/scale/轴向去强行接近历史数字。

## 中断后
停止/确认旧进程不再运行；同一环境、代码、数据、权重下重跑同一S3/S4。
每条完成记录的文件和输入hash通过就跳过。中途未完成那一条重算，不重算已完成数组。
环境或版本改变则拒绝混合；改新的RUN_TAG从独立目录运行，旧证据保留。
记录级保存会减少损失，不保证Drive断网前最后一次写入已同步。不要同时写同一run。

## S6：未来接入新的HJD或其他真实预测
阅读`NEW_MODEL_CONTRACT.md`。目前没有真实reference定义/已训练新模型时保持关闭。
不提交synthetic benchmark输出、不让评分脚本把缺失的方法补成零或伪造结果。
comparison源码今晚已经齐全，不等于新模型实验今晚也已完成。

## 今日停点
优先完成S0/S2/S3、提交源码与真实状态；时间足够完成S4。
源代码已经发现，不再开全盘搜索，不重新训练旧head。保留original checkpoint数值恢复待验标签。
