# Data README

## Overview
This project uses paired `32×32×32` CT-dose cubes for CT-to-dose prediction experiments.

Each sample consists of:
- one **CT cube**
- one aligned **dose cube**

The task is to learn the mapping:

`CT cube -> dose cube`

---

## Data format
The working data are stored as paired cube files, typically in NumPy format.

Each cube has shape:

`32 x 32 x 32`

with:
- CT values representing the anatomical / material structure
- dose values representing the target dose distribution

---

## Pairing
The project assumes that each CT cube has a corresponding dose cube with the same filename or sample identity.

This pairing is required for:
- regression experiments
- flow-based experiments
- validation and targeted evaluation
- casewise comparison across models

---

## Dataset usage across phases

### Early pilot stage
Earlier pilot experiments used smaller subsets, such as:
- 64 training samples
- 32 test samples

These were mainly used for:
- feasibility checking
- pipeline debugging
- early qualitative analysis

### Current development stage
The larger development-stage 3D flow experiments use:
- **2000 training samples**
- **500 validation samples**

In terms of case-level grouping, the project uses:
- **6 training cases**
- **2 validation cases**
- **2 fully held-out test cases**

The held-out test set is intentionally kept untouched during the current development stage.

---

## Splits
Split files are stored in:

```text
data/splits/