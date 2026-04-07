
# CT-to-Dose Prediction: Regression vs Flow on Paired 2D/3D Cubes

This project studies the mapping from paired CT cubes to dose cubes on a `32×32×32` dataset, and compares regression-based and flow-based approaches under controlled 2D and 3D settings.

## Project goal

The goal is to learn a mapping

$$
\text{CT cube} \rightarrow \text{dose cube}
$$

and to understand how far conditional flow models can approach strong regression baselines.

---

## Data setup

The project uses paired `32×32×32` CT-dose cubes.

A patient-level split was created:
- first 8 case folders for training
- last 2 case folders for testing

This avoids train/test leakage across patients.

More details are documented in [`data/README.md`](data/README.md).

---

## Repository structure

```text
ct2dose-project/
├── README.md
├── meeting_summary_apr7.md
├── .gitignore
├── notebooks/
│   ├── 2d/
│   │   └── ct2dose_2d_baselines.ipynb
│   └── 3d/
│       └── ct2dose_3d_regression_and_flow.ipynb
├── data/
│   ├── README.md
│   └── splits/
│       ├── train_pairs_2d.json
│       ├── test_pairs_2d.json
│       ├── train_pairs_3d.json
│       └── test_pairs_3d.json
├── docs/
│   └── figures/
│       ├── regression_2d_best.png
│       ├── flow_2d_best.png
│       ├── regression_3d_best.png
│       ├── flow_3d_best.png
│       ├── flow_euler_check.png
│       ├── regression_3d_loss.png
│       └── flow_3d_loss.png
└── outputs/
```
---

## 2D results

In the 2D setup, the CT-to-dose mapping is clearly learnable.

The 2D U-Net regression baseline performs better than the 2D conditional flow baseline under the current setup.

### Best 2D regression example

![Best 2D regression example](docs/figures/regression_2d_best.png)

### Best 2D flow example

![Best 2D flow example](docs/figures/flow_2d_best.png)

---

## 3D results

In 3D, the CT-to-dose mapping is also learnable.

A strong 3D regression baseline was established and used as the main reference model. Several 3D conditional flow baselines were then tested under comparable settings.

### Best 3D regression baseline

**Setup**
- 64 training samples / 32 test samples
- 3D U-Net
- `base_ch = 24`

**Metrics**
- Train MSE: `0.000250`
- Test MSE: `0.000319`
- Train MAE: `0.010088`
- Test MAE: `0.010948`

![Best 3D regression example](docs/figures/regression_3d_best.png)

### Best 3D flow baseline

**Setup**
- 64 training samples / 32 test samples
- conditional 3D U-Net flow
- `base_ch = 24`
- `batch_size = 2`
- `lr = 3e-4`
- `50 epochs`
- `30 Euler steps`

**Metrics**
- Train MSE: `0.000404`
- Test MSE: `0.000600`
- Train MAE: `0.013144`
- Test MAE: `0.015693`

![Best 3D flow example](docs/figures/flow_3d_best.png)

---

## Additional analysis

### Euler step check

The same 3D flow checkpoint was re-evaluated with different Euler step counts:
- 30
- 50
- 100

Result: the metrics and visual outputs remained almost unchanged.

This suggests that the remaining gap between 3D flow and 3D regression is **not mainly caused by insufficient Euler sampling resolution**.

![Euler step check](docs/figures/flow_euler_check.png)

---

## Training curves

### 3D regression loss

![3D regression loss](docs/figures/regression_3d_loss.png)

### 3D flow loss

![3D flow loss](docs/figures/flow_3d_loss.png)

---

## Main conclusion

The current stage supports the following conclusions:

- the paired CT-dose pipeline works correctly
- the CT-to-dose mapping is learnable in both 2D and 3D
- regression is currently the stronger baseline
- conditional flow is feasible and improves substantially with a more stable training setup
- the remaining flow-vs-regression gap is more strongly related to optimization stability than to Euler sampling resolution

---

## Notes

The large raw cube data are **not** included in this repository.  
Only lightweight split files, notebooks, and representative figures are tracked here.
=======
# Rong AP rectified flow matching



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.umm.uni-heidelberg.de/medphys/juergen-hesser/rong-ap-rectified-flow-matching.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.umm.uni-heidelberg.de/medphys/juergen-hesser/rong-ap-rectified-flow-matching/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

