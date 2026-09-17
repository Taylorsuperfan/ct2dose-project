# V1 结果结论 / Result conclusion

## 中文

本轮已完成真实 Phase9G 残差修正比较，不是仅有软件测试。冻结旧 Phase9G，学习 r=GT-Phase9G，与原 Phase10D-strict 及旧组件在相同600条validation记录上比较。两种新方法使用192条train、384次更新、batch=2、seed=17；best权重由40条validation的全体素RMSE选择。

普通 residual RF 相比 Phase10D-strict：全体素RMSE降低4.20%，MAE降低7.43%，但x-profile绝对RMSE增加3.45%，x平均百分比误差从6.674168%升至7.895002%。

HJD 相比 Phase10D-strict：全体素RMSE降低3.47%，MAE降低4.99%，但x-profile绝对RMSE增加2.34%，x平均百分比误差升至7.268619%。HJD比普通residual RF更好地保留x-profile，但未超过原最终系统。两种新方法的三个方向绝对profile RMSE均略高于Phase10D；不能笼统写“侧向结构全面改善”。

Done表示完成实验执行与结果记录，不表示已证明新方法整体更优。不是water-reference实验；不是独立最终测试；轴向沿用旧数组定义；单位是存储数值单位。当前表为用户提供的汇总数值，本文没有重新运行模型。

## English

I recovered the original Phase10D-strict pipeline and evaluated ordinary signed residual RF and spatial HJD on the same 600-record validation cohort. The target was real GT minus frozen Phase9G prediction, not a water-dose reference. Each new method used 192 training records and 384 optimizer updates, with checkpoint selection based on full-volume RMSE on 40 validation records.

Relative to Phase10D-strict, ordinary residual RF and HJD reduced full-volume RMSE by 4.20% and 3.47%, respectively. However, their absolute x-profile RMSE increased by 3.45% and 2.34%, and mean x percentage error rose from 6.674168% to 7.895002% and 7.268619%. The profile regression is therefore not solely a percentage-denominator effect. HJD mitigates the x-profile regression compared with ordinary residual RF but has not improved the original system's main profile objective.

These are one-seed, two-case development results with inherited upstream-exposure limitations. V1 is frozen. The proposed next experiment will align final-dose profile losses and checkpoint selection with a predefined profile objective; no V2 performance claim is made.
