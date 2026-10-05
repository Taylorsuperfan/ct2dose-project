# New conditional pilot decisions

- The completed grid-center codec checks verify representation, not new prediction.
- The 4096-draw target example motivates separating coupling supports from inference
  quadrature. It does not prove that a32768-source model will reduce real error.
- WFR magnitude transport plus a separately supervised grid sign is the main line.
  Historical HJD and direct signed RF results stay as frozen comparators; no parallel
  new HJD research program is started merely to satisfy comparison.
- The final Practical baseline is restored Phase10D-strict. It is not replaced by
  the early Phase3 model or a look-alike network.
- Main source is uniform numerical magnitude1 at each unit-box voxel center, not CT
  HU mass and not an artificial dose reference. This is a proposed computational
  source with the legacy error-correction target unchanged.
- Main inference uses trilinear magnitude deposition followed by soft grid sign.
  The hard-sign output is a separately reported sensitivity. No target-total rescaling.
- The first actual pilot retains384 updates and the previous monitor40 selection
  objective to expose changes without another unreported selection criterion.
- Whole-volume gains with profile losses remain a trade-off, not success on the
  main previous profile objective. A later profile-aware modification must get a
  new config, run name, precise loss definition and predeclared review criteria.
- This package's implementation, coupling approximations and source choice are
  not attributed to the supervisor or to the author code. No clinical claim.
