# 从已有源码到GitHub：本次唯一操作入口

这是提交文件的整合包，不是新实验版本。不训练，不推理，不重新画图，不自动修改Trello或GitHub。旧完整可视化交接包147个仓库文件和最新版导师阅读材料49个文件逐字节保留。新工具只补上统一导入、完整Git暂存核验和最新卡片附件映射。

## 一、下载解压

将 `dose_prediction_submission_20260918.zip` 解压到 `~/Downloads/dose_prediction_submission_20260918`。
里面：
- `repo_additions/`：应该进入原仓库的文件。
- `install_into_repo.py`：默认检查，只有`--apply`才复制缺少文件。
- `00_READ_ME_CN.md`：本说明。
- `submission_tests/`：本地交接工具测试，不需要为提交重新训练。

不要把外层交接包目录、整个Downloads或整个Google Drive拖入GitHub。
不需要以前那几份ZIP逐一安装，这份已包含它们。旧文件已在原仓库且相同会被保留，有不同会停止。

## 二、先给Trello附图

只用`repo_additions/docs/thesis/meeting_20260918/figures/`下新版PNG。对应表和英文评论：
`repo_additions/docs/thesis/submission_20260918/TRELLO_ATTACHMENTS_CN.md`。
卡片无需再建：#111图05/06；#110可选图07；#112七图全集；#58仅可选图05作为当前动机；#105/#106/#113不必硬配结果图。

## 三、Mac Terminal进入真实仓库

输入 `cd ` 后将Finder中的真实`ct2dose-project`拖入Terminal并回车。所有Git命令均在Mac运行，不在Colab。

```bash
pwd
git rev-parse --show-toplevel
git status -sb
git diff --cached --name-only
git remote -v
```

确认当前目录就是原Git仓库顶层；不要`git init`。已有无关暂存文件先审阅，不让本次commit混入。
保留`github=GitHub`、`origin=GitLab`，以实际`git remote -v`核对。不改remote、不猜仓库URL。
认证失败不要把密码/token贴进代码或聊天。

## 四、使用原先本轮分支

先读取GitHub远端状态：

```bash
git fetch github
BRANCH=thesis/phase9g-real-residual-pilot
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
elif git show-ref --verify --quiet "refs/remotes/github/$BRANCH"; then
  git switch --track -c "$BRANCH" "github/$BRANCH"
else
  git switch -c "$BRANCH"
fi
git status -sb
```

如果工作区变更妨碍切换，停止检查；不要force或reset --hard。
已存在同名分支不重复创建。若本地和远端分歧，先解决，不force-push。
新分支从当前分支出发，推送会包含当前分支未在远端的历史，下面会提醒检查；本工具不审计过去所有commit。

## 五、导入前检查，再应用

```bash
python3 "$HOME/Downloads/dose_prediction_submission_20260918/install_into_repo.py" --repo "$PWD"
```

预期看到新文件数、相同文件数和`CHECK ONLY`。冲突会先报告并停止，不复制冲突文件。
路径出现`(1)`等下载后缀时请改成实际路径，不猜测已执行了另一份包。

没有冲突才运行：

```bash
python3 "$HOME/Downloads/dose_prediction_submission_20260918/install_into_repo.py" --repo "$PWD" --apply
python3 tools/meeting_submission/check_submission.py
```

新检查器核对所有提交文件，继承旧的源码与Notebook payload检查，并承认新增`meeting_20260918`目录。不要继续使用旧checker的`--staged`对新增目录检查，也不要修改旧manifest来绕过限制。
此步不需要PyTorch或GPU，仅Python标准库。
`.DS_Store`与编译缓存可被工作区检查忽略，但若被暂存，仍会被完整暂存检查拒绝。

## 六、只暂存明确路径

```bash
git add -- \
  research/phase10d_recovered_comparison \
  research/phase9g_signed_real_v1 \
  research/phase9g_v1_visuals \
  docs/thesis/updates/20260917_real_pilot_v1 \
  docs/thesis/updates/20260917_real_pilot_v1_visuals \
  docs/thesis/meeting_20260918 \
  docs/thesis/submission_20260918 \
  tools/pilot_v1_release \
  tools/pilot_v1_visual_release \
  tools/meeting_submission

python3 tools/meeting_submission/check_submission.py --staged

git diff --cached --name-only
git diff --cached --stat
git diff --cached --check
git --no-pager diff --cached -- docs/thesis/meeting_20260918 docs/thesis/submission_20260918
```

完整暂存核验同时检查：每个清单文件已在Git索引中（包括此前提交过的相同文件），没有其他任务混入暂存，没有漏加被ignore的图片，暂存内容与预期相同。不是只检查工作区版本。
不要 `git add .`。文件不同不通过时不要直接换回旧训练源码；需要保留实际运行版本，核对来源。

如误暂存一个已确认不属于本次的文件：

```bash
git restore --staged -- "实际文件路径"
```

仅撤回该文件暂存，不删除工作区内容。不要省略`--staged`。
若明确一个PNG被全局规则挡住：

```bash
git check-ignore -v -- docs/thesis/meeting_20260918/figures/05_changes_from_previous_system.png
```

先确认是本包该图且hash检查通过，再仅对这个具体路径 `git add -f -- 路径`。不要强制添加全仓库。
`gitignore`不处理已跟踪文件，也不清除过去commit。若敏感内容已经进入历史，单独处理，不直接push。

## 七、commit并检查待推历史

如果差异不为空且审阅完毕：

```bash
git commit -m "Publish dose-correction comparison with readable figures and method descriptions"
git status -sb
git log --oneline --decorate --max-count=10
```

如果没有任何差异，可能同一包已经提交；核验后直接检查远端，不为了新建commit改动实验。
若分支在GitHub已存在，查看将被推送的历史：

```bash
git log --oneline github/thesis/phase9g-real-residual-pilot..HEAD
git diff --stat github/thesis/phase9g-real-residual-pilot..HEAD
```

若远端还没有此分支，上面两条不运行。确认当前分支的先前历史适合推送；不是只审查最后一次commit。

## 八、push与确认

```bash
git push -u github thesis/phase9g-real-residual-pilot
git fetch github
git rev-parse HEAD
git rev-parse github/thesis/phase9g-real-residual-pilot
git status -sb
```

两个commit哈希相同后，在GitHub切到`thesis/phase9g-real-residual-pilot`查看：
1. `docs/thesis/meeting_20260918/README.md`（导师阅读入口，完整方法名与新版图）；
2. 两个训练/恢复源码目录；
3. `docs/thesis/submission_20260918/ALL_FILES.txt`（本次逐文件清单）。

分支上传不等于合并main。仓库private时给导师适当访问权限，否则链接无法查看；未确认研究公开许可保持private。

## 九、最后更新发布卡

在当前#106发布卡添加真实commit链接（不是sandbox链接、/content路径或占位符）和README链接：

```text
Publication verified
Branch: thesis/phase9g-real-residual-pilot
Commit: <replace with the actual commit ID>
Commit URL: <copy the actual GitHub commit URL>
Reader entry: docs/thesis/meeting_20260918/README.md
Checks: planned files reviewed; local and remote commit IDs matched.
No medical arrays, trained weights or private case manifests were committed.
```

不要原样保留尖括号占位符。检查导师能访问，再把发布卡移Done。
其余进行中/待办任务不会因代码发布而自动完成，不重新重写你已经改好的卡片。

## 文件边界

上传：模型与评价/绘图源码、无输出Notebook、测试、方法/实验说明、允许共享的模型级汇总CSV和新版PNG/SVG。
不上传：CT/dose/预测数组、缓存、旧新权重、optimizer/RNG、case/patient manifests、原始带输出Notebook、导师私人材料、私有记录级HTML、token/密码。
即使仓库private也无需把医学数据与权重加入这个交接。
原始运行结果保留在Drive，尤其缓存仍引用原600条结果；不要上传完就删运行目录。

自动检查不等于隐私认证、研究公开许可或科学验证；不是模型重新训练测试。

## 官方操作参考
- https://support.atlassian.com/trello/docs/adding-attachments-to-cards/
- https://support.atlassian.com/trello/docs/what-is-a-card-cover
- https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github
- https://git-scm.com/docs/gitignore
- https://git-scm.com/docs/git-push
- https://git-scm.com/docs/git-commit
