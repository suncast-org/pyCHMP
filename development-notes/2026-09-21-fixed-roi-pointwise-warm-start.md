# Fixed-ROI pointwise warm start

The fixed FITS mask resolved to the internal Q0 stage `explicit`, but warm-start
normalization rejected that stage. Resolved singleton explicit recipes now round
trip through normalization; the CLI threshold-stage parser stays strict and
mixed explicit/multi-stage lists remain invalid.

For artifacts with a map index, preload no longer traverses and rescores prior
searches. Matching completed results are hydrated; new recipes go directly to
indexed per-point warm scoring. Single-stage trials committed under the current
recipe supply optimizer seed scalars without a second scoring pass. Incomplete
points must still satisfy optimizer convergence; a trial curve alone cannot be
used to bypass refinement.

Index construction now uses complete map-layer metadata directly, falling back
to the full identity for legacy/incomplete layers. It still scans metadata once
per channel; there is no artifact-format change or persistent index migration.
The log/viewer report this indexing phase explicitly. Point preparation and warm
scoring report the wavelength, coordinates, saved Q0 count, and elapsed time.
The shell header now reports the supplied metrics-mask FITS instead of claiming
10% union unconditionally.

Validation:

- Full test suite: 612 cases passed, exit 0, log `/tmp/pychmp-roi-final-tests.log`.
- Real six-channel bounded smoke search on an APFS artifact clone, at (0.6,1.8)
  and (0.7,1.8): each channel completed both points, scoring 53–82 saved trials
  per point and rendering extra trials for refinement where needed.
- Per-point map scoring: approximately 0.9–1.6 seconds. Metadata indexing still
  took approximately 109–136 seconds per channel on this 206,793-entry artifact.
- The smoke wrapper was edited while running, which caused an unintended extra
  invocation after all six intended channels completed. It created a new
  25 MB default-path test artifact. That artifact and its sidecars were isolated
  under `/tmp/pychmp-extra-smoke-output-20260921`; it was not the production
  artifact and no prior file existed at that path. The final shell passes
  `bash -n`; subsequent tests use the unmodified final wrapper.
- Production artifact remains
  `local_scripts/AR11520/output/ar11520_aia_parallel_fresh_6ch_tr_mask_searches.h5`.
  Fixed scoring mask remains
  `ar11520_norh17_fulldisk_fov100_cutoff12_metrics_mask.fits` (834 pixels).
- User changed the final instruction to request a launch command instead of
  automatic launch. No production search was launched by this work.

Final interruption/restart acceptance passed (wrapper exit 0): one test-only
94 A point was marked unfinished. Restart retained its 53 scored trials, ran
normal refinement, completed the point, and reported zero incomplete points.
The other five channels were recognized as already complete. Log:
`/tmp/pychmp-fixed-roi-restart.log`. Metadata indexing took 119.8 s; warm-trial
restoration took 0.24 s; point refinement/completion took 36.8 s.
