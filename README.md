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
