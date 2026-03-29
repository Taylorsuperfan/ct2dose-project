# Data notes

This repository does not include the full raw CT-dose cube dataset.

## Included
- train/test split files for 2D and 3D experiments:
  - `splits/train_pairs_2d.json`
  - `splits/test_pairs_2d.json`
  - `splits/train_pairs_3d.json`
  - `splits/test_pairs_3d.json`

## Not included
- raw `.npy` cube files
- large intermediate outputs
- large checkpoints

## Data organization used in the experiments

The project assumes paired folders of the form:

- `input_cubes/`
- `output_cubes/`

where matching files share the same filename.

A patient-level split was used:
- first 8 case folders for training
- last 2 case folders for testing

## Notes

The raw dataset is intentionally excluded from version control to keep the repository lightweight and reproducible.