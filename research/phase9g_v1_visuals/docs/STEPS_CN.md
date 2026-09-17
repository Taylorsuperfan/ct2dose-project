# V1 可视化：完整步骤

## 直接使用已经生成的图
交接包已包含7张PNG与7张SVG，位于repo_additions/docs/thesis/updates/20260917_real_pilot_v1_visuals/gallery/figures/。从用户已报告的高精度CSV生成，不是本次重新评价Drive模型。推荐会议主图01、03、05、06；07仅辅助说明修正网络参数，不代表速度。

## Colab V0–V3（核对实际Drive CSV并保存一套新图）
下载phase9g_v1_visuals_colab.ipynb，在Colab上传并打开。CPU即可；不用原P0–P7，不用checkpoint，不训练。

V0：执行工具恢复和14项轻量测试；仅安装缺少的numpy/matplotlib。不更改PyTorch。

V1：执行Drive挂载，核对下面3个小文件：comparison.csv、REPORT_READY.json、FILES.json。目录固定为本轮reports.local/20260916T212513229830Z。验证scope完成/600records/2cases，并核对比较CSV与已报告数字的显示精度。缺文件或不一致停止，不回退到人工数据，不改旧receipt。

V2：生成新目录：
```text
/content/drive/MyDrive/ct2dose_thesis/phase9g_signed_real_v1/visualizations.local/V1_<timestamp>/
```
输出figures/七对PNG/SVG、comparison_approved_columns.csv、relative_changes_vs_phase10d.csv、source_provenance.json、README.md、gallery.html、FIGURES_READY.json、FIGURE_FILES.json。

V3：显示四张核心图，生成旁边的ZIP并下载。若ZIP已存在，逐文件核对后复用，不覆盖；已完成图可验证后直接读。不要让下载ZIP进入带receipt的图文件夹。

## 断线与保存
已经写入Drive的图目录保留；重连可重新V0–V2生成另一套图，代价只是读小CSV和画图，不重跑网络。若同一runtime执行V2，会验证相同输出。源/代码发生变化则要求新的输出目录。不要覆盖旧图receipt、checkpoint或训练report。

## 图片应怎样阅读
图1/2展示全局RMSE/MAE下降；图3展示x百分误差升高；图4是全部方法的三个绝对profile指标；图5各行独立相对变化，负数左侧表示更低误差，正数右侧表示更高；图6散点越左下越好，但没有整体赢家声明；图7参数少不等于运行快。

相对百分变化100*(new/old-1)不同于百分点(new%-old%)。数值乘10^6只改变显示尺度，不证明Gy。600records/2cases/seed17，monitor40重叠，不伪造置信区间/显著性或患者数。

## 哪一套图上传GitHub
本交接包内的快照图已经有完整hash和来源，直接作为审核后的汇总附件。Drive重新生成的图作为独立私有核对证据，不覆盖快照图文件；若以后要发布新的原CSV图，放新日期目录并单独审查。当前需要公开许可，图不含case ID并不意味着研究自动获准公开。
