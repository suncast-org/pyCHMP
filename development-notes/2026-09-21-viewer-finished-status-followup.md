# Deferred: viewer incorrectly reports FINISHED

User requested this be addressed later; do not change the running search for it.

Screenshot from the fixed NoRH ROI run shows:
- Artifact: ar11520_aia_parallel_fresh_6ch_tr_mask_searches.h5
- Slice: EUV 94 A; search_72b4da2f13369cc9
- Green FINISHED banner, but 0/1 points computed and Pending: 1.
- Search status: in_progress; search active marker: no.
- Last phase: adaptive search failed.
- Search started: 2026-09-21T13:48:26Z.
- Search completed: 2026-09-20T20:10:23Z (before this search started).

Investigate lifecycle field provenance and viewer status precedence: a stale
completion timestamp may be inherited from slice-common or a previous search.
Also verify that ordinary resume reopens runner lifecycle and publishes a fresh
heartbeat during startup/indexing, and that old failure phases are cleared.
These are hypotheses, not yet verified against the running process.

Expected behavior: distinguish running/indexing, completed, failed, and interrupted
states. An inactive marker alone must not imply successful completion. Use the
selected search's own lifecycle; do not inherit another search's completed_at.
Add a regression case reproducing these conflicting fields and resume startup.
