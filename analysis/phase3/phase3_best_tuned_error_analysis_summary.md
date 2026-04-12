# Best-Tuned Validation Error Analysis

- checkpoint: `lr5e4_base24_bs2_cont_to_50_best.pt`
- validation samples: `500`

## Validation-wide summary
- best validation samples are around `2.91% – 3.11%` global relative error
- worst validation sample global relative error: `11.86%`

## Typical case
- sample index: `264`
- along-beam mean percentage error: `2.71%`
- perpendicular mean percentage error: `2.67%`

## Worst case
- sample index: `50`
- global relative error: `11.86%`
- along-beam mean percentage error: `3.06%`
- perpendicular mean percentage error: `7.45%`

## Interpretation
- the tuned checkpoint improves the development-stage result further
- the main beam-shaped structure is learned very well overall
- the remaining difficulty is still strongest in harder perpendicular cross-sectional structure