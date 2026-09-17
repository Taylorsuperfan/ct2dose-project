# V1 可视化完整交接包

已包含此前完整V1交接的所有仓库文件（字节不变），再新增绘图源码、7张PNG+7张SVG、无输出独立Colab Notebook和本次按类别的完整Trello草稿。**无训练、无推理、无Trello/GitHub写操作。**

## 1 看图
打开 repo_additions/docs/thesis/updates/20260917_real_pilot_v1_visuals/gallery/gallery.html，或其中README.md。PNG用于Trello/PPT，SVG用于矢量排版。本包图来自用户已报告汇总；实际Drive核对运行新visual Notebook V0–V3。无需重复原训练。

## 2 导入原仓库
解压整个ct2dose_v1_visual_handoff_20260917到Downloads。不要把外层文件夹直接塞进仓库。Mac Terminal进入实际ct2dose-project后：
```bash
git status -sb
git diff --cached --name-only
git remote -v
if git show-ref --verify --quiet refs/heads/thesis/phase9g-real-residual-pilot; then
  git switch thesis/phase9g-real-residual-pilot
else
  git switch -c thesis/phase9g-real-residual-pilot
fi
python3 "$HOME/Downloads/ct2dose_v1_visual_handoff_20260917/install_into_repo.py" --repo "$PWD"
```
最后一条只检查。相同旧文件保留，新文件才导入；任何同路径不同内容均停止。已经导入旧交接也可以使用这次完整包。

通过后：
```bash
python3 "$HOME/Downloads/ct2dose_v1_visual_handoff_20260917/install_into_repo.py" --repo "$PWD" --apply
python3 tools/pilot_v1_visual_release/check_release.py
```
**使用新checker：旧tools/pilot_v1_release/check_release.py --staged不了解新增可视化路径，不能拿它拒绝新路径当模型错误；旧源码/manifest不修改。**

## 3 暂存并手工审阅
```bash
git add -- research/phase10d_recovered_comparison research/phase9g_signed_real_v1 research/phase9g_v1_visuals docs/thesis/updates/20260917_real_pilot_v1 docs/thesis/updates/20260917_real_pilot_v1_visuals tools/pilot_v1_release tools/pilot_v1_visual_release
python3 tools/pilot_v1_visual_release/check_release.py --staged
git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
```
核对图的来源、单位、数值和隐私权限。只允许清单内模型级汇总PNG/SVG，绝不是允许所有医学图像。不要git add .，不要删除旧checkpoints或用新文件覆盖实际训练过的改版源码。

已有真实数据或凭据若被跟踪，.gitignore不能清除历史；需单独处理，不直接推送。若全局ignore挡住图：用git check-ignore -v检查具体PNG路径；不要强制添加整个repo或全图目录。

## 4 提交
```bash
git commit -m "Add reproducible V1 aggregate figures and preserve global-profile trade-off"
git push -u github thesis/phase9g-real-residual-pilot
git fetch github
git rev-parse HEAD
git rev-parse github/thesis/phase9g-real-residual-pilot
git status -sb
```
保留github=GitHub，origin=GitLab。检查两个commit一致，在GitHub选对应分支浏览。此操作不自动合并main。认证或冲突时停止，不force-push/no hard reset。先审查公开许可，未确认保持private，且private也不上传医学数组与权重。

## 5 Trello
完整按Done/Doing/To Do分类文本位于repo_additions/docs/thesis/updates/20260917_real_pilot_v1_visuals/TRELLO_CARDS_COMPLETE.md。每张卡的title/description/checklist另有txt便于复制。
保留原#105/#106/#108的synthetic身份。#107记录剩余lineage，#109更新为发布任务不重复创建。Done不表示科学全面胜出。
打开卡片编辑Description；Checklist每行一项粘贴；Add→Attachment上传图05和06，可补01/03。不要把sandbox链接当Trello附件，先下载PNG。实际上传后记录当前commit，不提前标Done。

## 文件分区
GitHub：三份代码目录、无输出Notebook、测试、文档、模型级CSV与审核后PNG/SVG，全部有清单。
Drive：数据/权重/缓存/逐病例表/原始完整报告/执行Notebook/私有metadata。
此前交接和原registry全部保留历史语义；本次当前状态写进新增visuals目录，不能改旧RELEASE.json来消除pending。
