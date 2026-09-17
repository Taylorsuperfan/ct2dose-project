# 会议图序建议 / Figure sequence

01（全局RMSE）→03（x平均百分误差）→05（相对变化）→06（二维取舍）。02与04可补充，07作为新增网络资源附图。PNG可附Trello，SVG用于矢量排版。

解释05：左侧柱为相对Phase10D的误差下降，右侧柱为上升；每一行都是同一指标自身的相对变化，不能相加。图中+8.91%不是+8.91个百分点；对应HJD的x百分点差约+0.59445。

解释06：左侧意味着全局RMSE较低，下侧意味着x百分误差较低。新模型向左但向上；没有通过图形直接产生一个单一赢家。该图是放大散点视图，Phase5E在前四张全方法柱图中保留。

Suggested English explanation:
"The new residual RF and HJD corrections reduce whole-volume RMSE, but worsen the original x-profile error. The relative-change chart includes both improvements and regressions. The scatter plot makes the two-objective trade-off explicit. These are aggregate development results from two cases and one new-model seed, not evidence of an overall statistically significant winner."

原loss/选模没有profile项是解释方向，不由图形证明唯一原因。仅汇总数据不支持按体素/record画分布、箱线图或置信区间。
