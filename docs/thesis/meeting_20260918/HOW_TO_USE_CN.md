# 本次只改读者看到的名称和说明

## 不需要重跑训练或修改旧结果

已重新生成7张图，每张PNG和SVG各一份。使用的是上一份图表交接包中的高精度汇总CSV，字节保持一致。原来的模型源码、method标识、checkpoint、缓存、原始CSV、旧run、旧图表包和旧release manifest均不修改。

## Trello

打开同一研究任务的已有卡片，将标题和Description替换为TRELLO_CARDS.md中对应内容。每张卡也提供独立的title.txt、description.txt与checklist.txt，保存在trello/Done、trello/Doing或trello/To Do中。不要因为措辞修订重复创建任务。

新标题不再以V1、HJD、RF、Phase10D为读者入口，而是说明“做了什么、为什么做、结果是什么”。正文首次出现时解释rectified flow matching及Hahn–Jordan decomposition。严谨术语保留，不改成另一种方法。

实际模型恢复、训练与600条比较已经有用户运行证据，仍可列Done。图形文字修订和图生成已完成，但Drive核对、Trello附件上传、GitHub推送没有自动完成，所以发布任务仍按实际状态处理。剩余历史数据暴露问题保持Doing，下一轮profile目标设计保持To Do。

建议对comparison卡使用figures/05_changes_from_previous_system.png与figures/06_volume_and_profile_accuracy.png。旧附件可保留历史说明，但会议中应明确采用这套描述性图例。上传前确认研究共享权限。

## GitHub

不要给旧源码文件夹、checkpoint或CSV列做全局重命名。把这份文件夹作为一个新的文档目录，例如：

```
ct2dose-project/
  docs/thesis/meeting_20260918/
    README.md
    EXPERIMENT_OVERVIEW.md
    METHOD_NAMES_CN_EN.md
    TRELLO_CARDS.md
    HOW_TO_USE_CN.md
    method_names.json
    make_figures.py
    requirements.txt
    SOURCE_NOTES.json
    FILES.json
    VERIFICATION.md
    data/comparison_reported.csv
    figures/
    trello/
    tests/
```

若该目录已经存在不同内容，选择一个新的文档目录，不覆盖。先在当前研究分支中审阅新增目录，再作为“Clarify method names in meeting figures and task descriptions”这样的文档提交。

此前发布检查器的staged白名单只覆盖原交接目录，未必接受这个新文档目录。这不是模型损坏，也不需要改旧RELEASE.json或旧hash。单独审阅本目录的文件列表与内容；不要把旧检查器未覆盖的新文件宣称已由它验证。

本次不重复提供训练源码，因为它们没有任何变化。模型、旧报告、旧数据保留原路径；只在仓库说明中链接到这个新的阅读入口。

## 给导师看的顺序

先读README的研究问题与方法区别，再看图05和图06，最后查看具体数值或代码对应表。这样即使不打开代码，也能知道旧系统是哪一个、两个新方法差在哪、结果改善和退步各是什么。

本包没有修改真实Trello看板，没有push GitHub，没有读取医学数组或重新执行模型。
