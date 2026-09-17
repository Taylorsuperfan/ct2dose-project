# 本地、GitHub、Trello

## 本地
解压本包放到原仓库`research/phase10d_recovered_comparison/`，不要覆盖原文件。
原上传Notebook和全部Drive运行结果保存为私有备份，不直接提交。公开包只含源定义/工具/测试和匿名历史汇总。
代码中的旧操作片段保存在Markdown供阅读，不是自动训练脚本。

## Terminal
在原`ct2dose-project`内：
```bash
git status -sb
git remote -v
git switch -c thesis/phase10d-recovered-comparison
python3 research/phase10d_recovered_comparison/scripts/public_check.py research/phase10d_recovered_comparison
git add -- research/phase10d_recovered_comparison
git diff --cached --name-only
git diff --cached --check
git commit -m "Recover original Phase10D definitions and add validation comparison"
git push -u github thesis/phase10d-recovered-comparison
git fetch github
git rev-parse HEAD
git rev-parse github/thesis/phase10d-recovered-comparison
```
分支已存在时`git switch thesis/phase10d-recovered-comparison`，不要重复-c。github是GitHub，origin是GitLab，不重命名。
检查staged内容后再commit。checkpoint、医学数组、case manifests、运行report、原带输出Notebook不提交。
`.gitignore`不移除已经跟踪的敏感内容；自动检查不替代人工审阅。

## Trello草稿（未自动创建或修改卡片）

### Done — 原Phase10D-strict源码已定位，comparison代码已整理
- 实际源码在名称仍为Phase9G的上传Notebook里；含Phase10D类、损失、strict训练保存代码。
- 30个定义按原文提取并记录cell/hash；本地软件测试通过，验证范围见verification。
- 这不表示用户的checkpoint数值复现已经完成。

### Doing — 严格加载3份旧权重并复现真实validation
- 依据9D+ metadata选base，不使用开头旧的strong路径。
- S2三份strict加载；S3四记录smoke；S4原600条val subset。
- 保存数值单位、实际sampler、轴约定、cohort/hash和source snapshot。
- 完成后记录commit、报告位置、历史摘要差值；test不重新评价。

### To Do — 同cohort新方法比较
- 真实signed target/reference定义确认后，训练/恢复新模型。
- 用相同CT、cohort、指标产生absolute dose；接入external接口。
- 历史/新结果分开，synthetic不混进real表，数据暴露如实披露。

### To Do — 源码提交与会议简报
- 审查公开文件，git push github，并核对commit hash。
- 汇报发现：源码未丢失，而是追加在旧命名Notebook内；原axis约定不一致，复现保留，后续单独修正。
- 只有实际跑完S3/S4后才将相应任务移到Done。
