# Free-mode selection overwritten during live refresh

**Resolved 2026-09-20.** Implemented: obsolete background results are discarded, search controls are not repopulated from another slice while a selection loads, and grid refreshes remap Free-mode coordinates. Explicitly chosen navigation modes override remembered startup/view state. Regression tests cover queued selection changes and grid growth.

The investigation notes below describe the original behavior.

Investigated without modifying executable code or opening the running artifact.
An isolated reproduction is in `/tmp/repro_pychmp_free_refresh.py`.

Two defects were reproduced using the current viewer methods:

1. A background metadata load captures a slice/search selection. If the user
   changes selection before completion, the callback still applies the older
   payload. `_schedule_background_scan_load` checks its load token but does not
   invalidate that token when a newer selection is deferred. The metadata
   callback does not check current selection before applying its result.
   `_refresh_search_controls` then uses the older payload's search list and
   replaces the user's search ID when it is absent from that list. Free mode
   remains selected throughout.
2. `_apply_slice_grid_metadata_from_payload` replaces the coordinate arrays
   without remapping the Free-mode selector indices to `_free_selection_ab`.
   Inserting coordinates before a selected point leaves the same indices
   referring to another point. Live-trial eligibility uses these indices, so
   the displayed curves/maps can disagree with the preserved coordinate anchor.

The harness demonstrated `chosen_search` becoming `active_search` while mode
remained `free`, and a coordinate anchor of `(1,0)` with selectors addressing
`(0,0)` after grid growth. This establishes code defects consistent with the
reported symptom; the exact route in the user's live viewer was not traced.

Fix after the current search, or in an isolated viewer-only copy:

- Reject stale load results when their artifact/slice/search selection no
  longer matches the requested UI selection; continue the latest queued load.
- Preserve Free-mode coordinate anchors and remap indices whenever the grid
  arrays change, before selectors or live-trial eligibility are refreshed.
- Test delayed loads across selection changes and grid growth in Free mode,
  including curve/map display. Keep Active-mode following intact.

Existing Free-mode refresh tests mock out selector refresh and do not exercise
the delayed-load completion path, so they do not cover the first reproduction.
