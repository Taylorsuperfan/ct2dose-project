## 1. Loss curve check

### 3D regression
- observation: The regression training loss decreases smoothly and almost monotonically.
- interpretation: The regression training appears stable, with no obvious sign of divergence or strong oscillation.

### 3D flow
- observation: The flow training loss also decreases clearly over time, but it is noisier than the regression loss curve.
- interpretation: The current best flow setup appears stable overall. Although the curve shows small fluctuations, there is no obvious divergence.

---

## 2. Normalization and scaling check

### CT input
- raw min / max / mean / std:
  - train: min = -1227.76, max = 2308.05, mean = -275.42, std = 522.17
  - test: min = -1225.59, max = 2506.50, mean = -295.22, std = 504.94
- normalized min / max / mean / std:
  - train: min = 0.0, max = 1.0, mean = 0.2965, std = 0.2066
  - test: min = 0.0, max = 1.0, mean = 0.2886, std = 0.1995

**Observation:**  
The normalized CT values fall into the expected [0, 1] range. Train and test statistics are very similar, which suggests no strong distribution mismatch. The histogram shows a noticeable peak near 0 and another concentration around 0.35–0.45, which is consistent with background/air and tissue-dominant structures. Some clipping occurs at the lower and upper bounds, but there is no sign that most values are collapsed to the extremes.

### Dose output
- raw min / max / mean / std:
  - train: min = 2.44e-08, max = 0.00426, mean = 4.67e-05, std = 6.97e-05
  - test: min = 2.18e-08, max = 0.00383, mean = 4.67e-05, std = 6.81e-05
- scaled min / max / mean / std:
  - train: min = 2.44e-05, max = 4.26, mean = 0.0467, std = 0.0697
  - test: min = 2.18e-05, max = 3.83, mean = 0.0467, std = 0.0681

**Observation:**  
The raw dose values are extremely small, so scaling by 1000 brings them into a numerically more convenient range. After scaling, the mean and standard deviation are of reasonable magnitude for optimization, and train/test statistics remain closely matched. However, the dose histogram is strongly right-skewed, with most voxels concentrated near zero and only a small fraction taking larger values.

---

## 3. Histogram check

### CT normalized histogram
- observation: The train and test CT histograms are very similar. The distribution is not uniform, but this is expected for CT data. The peaks near low values and around 0.4 appear physically plausible.

### Dose scaled histogram
- observation: The train and test dose histograms are also very similar. Most values remain near zero even after scaling, and the distribution has a long tail.

---

## 4. Preliminary conclusion

- The regression training appears stable.
- The flow training also appears stable overall, although it is noisier than regression.
- The current CT normalization seems reasonable.
- The current dose scaling by 1000 also seems numerically reasonable.
- There is no obvious train/test distribution mismatch.
- A possible challenge is that the dose distribution is highly imbalanced, since most voxels are close to zero.