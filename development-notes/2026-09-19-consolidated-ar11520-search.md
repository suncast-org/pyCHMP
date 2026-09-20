# Consolidated AR11520 search implementation

The launcher uses the existing `design/moddir-compatible-map-store` checkout.
This integration combines the following previously divergent work:

- `05708f6`: current map-store implementation, including auxiliary component identities, explicit masks, pinned observation time, model-refmap selector restoration, and EUV observation units inherited from main.
- `c0cb73b`: observer ephemeris metadata is no longer passed as renderer-relative longitude; compatible saved-FOV searches delegate observer restoration to gxrender.
- `5ddad33` and `2620e52`: accurate basin-expansion guidance, canonical pinned mask-stage recipes, warm rescoring, and recipe-drift guards.
- `db19d36`: archived incomplete-point finalization/resume, bracket-local Q0 refinement, multi-stage curve handling, PSF persistence/reconstruction, instrument resolution, and viewer refresh/read tolerance.
- Existing local work: parallel/exact/thread projection controls and projection-sensitive search/map identities.

Conflicts were resolved by keeping current channel-safe component storage, observation/template/TR-mask overrides, and explicit-mask display support while adding the historical resume and mask-stage behavior. The newer reusable-map provenance code supersedes the archive's legacy wavelength matching. Warm evaluation of recombined TR maps now takes the requested mask stage. Pinned search restoration includes canonical projection settings.

The projection options require the corresponding gxrender SDK/workflow changes in the adjacent `gximagecomputing` checkout. Both are imported by the local suncast Python environment.

Validation uses the full pyCHMP suite, the gxrender response-default tests, and the actual AR11520 six-channel launcher with one (a,b) point and bounded Q0 iteration settings on a disposable artifact. This validates execution and persistence; it is not a scientific convergence run. See the workspace AR11520 readiness note for results and the production launch command.
