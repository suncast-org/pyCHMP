# Ordinary restart preserves completed searches

An unchanged AR11520 six-channel restart was rescoring saved map-store trials.
For 131, 171, 193, and 211 A, slice-common diagnostics still carried the
`render_only_slice` marker from their original auxiliary rendering. Startup
therefore discarded their diagnostics despite saved searches. Restoring the PSF
kernel without its recipe changed `psf_source` and `resolved_psf`, creating a new
search identity. In addition, ordinary matching resumes did not enable the
stored-score preservation used by repair/expansion.

The adaptive launcher now resolves the active search before loading its
observation reference and overlays its saved recipe diagnostics, including exact
PSF provenance. A matching ordinary restart preserves stored trial scores and
hydrates the explicitly selected search. It does not eagerly promote/rescore
map-store curves; additional points obtain warm trials lazily.

After the existing recipe and geometry compatibility checks, a finished search
with matching adaptive bounds, steps, and seed exits before registration,
map-store indexing, or optimizer work. This requires a finalized inactive
lifecycle and completed, non-corrupt grid-point headers. Explicit recomputation,
new identity, targeted repair, expansion, and failed-point retry bypass this
completed-run shortcut. No artifact schema, signature algorithm, or stored data
migration was introduced.

## Validation (2026-09-21)

- Full test suite: 606 tests passed (exit 0). Nine new cases cover auxiliary
  placeholder/PSF restoration, finished versus pending/corrupt/unfinalized runs,
  changed bounds/seed, and avoiding eager map rescoring on matching resume.
- Ran the existing six-channel launcher with unchanged scientific settings on an
  APFS clone of the user's completed 8.4 GB artifact:

  ```sh
  SUNPY_CONFIGDIR=/tmp/pychmp-preflight-sunpy \
  MPLCONFIGDIR=/tmp/pychmp-preflight-mpl \
  local_scripts/AR11520/run_ar11520_aia_parallel_fresh_6ch_tr_mask_searches.sh \
    --artifact-h5 /tmp/pychmp-completed-resume.h5 \
    --log-dir /tmp/pychmp-resume-acceptance-all
  ```

- All six channels exited 0 and reported `Already completed`, retaining IDs
  e03b057b5a6952a8, 513ec885ae0a6a68, 3cae42c77bf0ce07, 0bb2997b355b519b,
  70b33372e7ddc049, and 178dc1b828805407 respectively.
- Total wall time: 32 seconds (12:41:58–12:42:30 UTC), approximately 5–6 seconds
  per Python process for startup/input/preflight checks. No point starts,
  warm-map scoring, resume preload, or optimizer work occurred.
- `cmp -s` between the original artifact and tested clone returned 0 after the
  launcher completed: the HDF5 artifact was byte-for-byte unchanged. Logs and
  sidecars were confined to the test paths. Original results were not modified.
- Full test log: `/tmp/pychmp-resume-full-tests.log`; acceptance output:
  `/tmp/pychmp-resume-all.log` and `/tmp/pychmp-resume-acceptance-all/`.

## Completed-run viewer follow-up

The completed-run shortcut originally returned before the viewer launch. It now
calls the existing launch/reuse helper before returning, respecting `--no-viewer`
and `PYCHMP_NO_AUTO_VIEWER`. Viewer streams are redirected to
`<artifact>.viewer.log`, with stdin disconnected, so the detached viewer cannot
keep the launcher's `tee` pipeline open after the scan process exits.

Validation: adaptive CLI tests passed (57 cases). The original six-channel
launcher with `--viewer` launched the viewer and reused running instances for
subsequent channels; all channels reported already completed and the launcher
exited 0. The final viewer process remained running after the launcher exited. Acceptance output is in
`/tmp/pychmp-completed-viewer-check.log`.
