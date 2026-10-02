# Shared AIA response and convolution-kernel persistence

## Implementation

Dynamic AIA responses previously bypassed the adapter cache when response_sav
was absent. Each rendering call could regenerate the response and contact JSOC.
The adaptive workflow now loads a validated response from artifact calibration
storage, or resolves it once and saves it. The renderer factory passes this
prebuilt payload to every point adapter, including across worker serialization.
The adapter also now caches dynamic responses for its own subsequent calls.

Layout: calibration/euv_responses/<setup key>/channels/<channel> stores one
response vector per channel. The temperature grid, units-related ds value,
structured dtype and source/correction metadata are shared at setup level.
Channel subsets reuse these same datasets; search IDs, ROI, metrics and a/b/Q0
are excluded from the storage key. Payload checksums are validated when loaded.
Different calibration setups are kept separate; existing values are never
silently replaced. No response or kernel arrays are added to search groups.

Convolution kernels already use slices/<channel>/common/psf_kernel. All six
fitted AIA channels had saved 201x201 kernels. Startup reads these arrays before
considering generation, independent of the selected search. Auxiliary 304 A
has only been rendered raw, so no convolution kernel has been needed for it.

## Production artifact recovery

Patched:
local_scripts/AR11520/output/ar11520_aia_parallel_fresh_6ch_tr_mask_searches.h5

Backup (APFS clone):
local_scripts/AR11520/output/ar11520_aia_parallel_fresh_6ch_tr_mask_searches.before_response_persistence_20260921.h5

One shared calibration setup contains seven channel response vectors (94, 131,
171, 193, 211, 304, 335). Canonical response SHA256:
781b4575c3ea996589e5a42f87bcbea5c3b4a61f0ff4e481e9fd701972cea9e6

The old artifact had neither response arrays nor response hashes. The corrected
response was recovered using the same model and evenorm_chiantifix settings.
Historical byte-for-byte equivalence cannot be certified. That limitation and
the original recipe metadata are retained in shared legacy_bindings_json.
Eleven previously unhashed search recipes were associated with this response,
with their existing search IDs retained. Past map identities were not rewritten.

Patch verification compared grid-point data/attributes, trial scores, map IDs,
and kernel arrays before/after; these scientific contents were unchanged. The
interrupted fixed-mask 94 A search still has 62 completed points and 13 trials
at its unfinished point (a=1.0,b=1.3).

## Validation

- 616 tests passed. New tests cover dynamic adapter caching, response round-trip,
  channel-subset sharing, corruption/replacement rejection, and explicit legacy
  binding without changing saved map/score data.
- Offline startup on four previously completed channels found the original
  search IDs and exited without rendering or network access.
- Actual native all-channel rendering for 94 and 131 A was tested with socket
  connections and response/PSF-generation functions forced to fail. Both renders
  succeeded using the saved response and kernels (4.9 and 3.0 seconds).
- Offline preflight against the patched production artifact found the existing
  fixed-mask search_72b4da2f13369cc9. The test exited at the matching check, before
  running the search; its wrapper's generic completion line is not a claim that
  the scientific search is finished.
- Production search has NOT been restarted.

Evidence is retained in local_scripts/AR11520/output/calibration_persistence_20260921/.
