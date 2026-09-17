# 本地、GitHub、Trello

## 本地代码目录
解压本包，将`phase9g_signed_real_v1/`放到原`ct2dose-project/research/`下面，作为新目录；不覆盖旧恢复代码。
原数据、私有manifests、缓存、checkpoint与真实图表留在Drive。Notebook用无输出的交付版本提交。

Mac Terminal，进入ct2dose-project后：
```bash
git status -sb
git remote -v
git switch -c thesis/phase9g-real-residual-pilot
python3 research/phase9g_signed_real_v1/scripts/public_check.py research/phase9g_signed_real_v1
git add -- research/phase9g_signed_real_v1
git diff --cached --name-only
git diff --cached --check
git commit -m "Add real Phase9G residual RF and HJD pilot"
git push -u github thesis/phase9g-real-residual-pilot
git fetch github
git rev-parse HEAD
git rev-parse github/thesis/phase9g-real-residual-pilot
```
分支已存在时用git switch而非-c；不reset --hard，不git add .。
GitHub remote=github，GitLab=origin，不重命名。提交前人工检查凭据和私有数据。
自动公开检查只作初筛；.gitignore不会移除已经tracked的敏感内容。
本次交付未执行用户仓库push或Trello写操作。

## 更新已有卡片，不重复建同义任务
### Done：原Phase10D-strict基线恢复与val600复核
依据：用户实际完成的600条验证及接近历史汇总的输出。逐体素历史身份未证明；原上游暴露限制保留。
### Doing：真实Phase9G残差训练缓存与可学习性检查
目标：r=GT−frozen Phase9G，不是water。192train，12条train-only sanity。
验收：缓存hash、train-only尺度、正负分解审计，RF/HJD各自small run history与checkpoint。
### To Do：普通残差RF / HJD受控pilot
验收：两模型从新初始化训练，记录192train/384updates/40valmonitor、全部loss/solver/参数/耗时；旧系统不改变。
### To Do：同val600新旧预测比较与报告
验收：两新模型各600条，原预测复用；相同评价、raw/clamped区分、paired per-case delta和HJD质量诊断。
不根据一部分结果将卡片移Done；不把600patches称600patients。
### To Do：代码提交和结果范围核对
验收：源码审阅、git commit/push/hash一致；实际实验未跑完时只说代码提交完成。
