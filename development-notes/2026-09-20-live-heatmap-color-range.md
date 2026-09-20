# Live heatmap range distorted by provisional point metrics

**Resolved 2026-09-20.** Implemented: incomplete points are drawn as outlines and excluded from color normalization. Linear/log and no-completed-point cases have regression coverage. Completed outliers are retained.

The investigation notes below describe the original behavior.

User screenshot: Active mode, EUV 211 Å, eta2, logarithmic heatmap,
192 computed points. Most cells appear uniformly purple while the colorbar
extends into the tens.

A subsequent read-only load of the same slice (include_maps=False) found 192
completed points with eta2 min/median/max approximately
0.43941844 / 0.47166237 / 0.90350427. One pending point then had eta2 0.50312682.
The artifact had advanced, so the exact provisional value at screenshot time
was not recovered.

Confirmed code behavior:

- `_draw_heatmap` puts pending records with finite metrics into
  `computed_records`, which supplies both colored patches and normalization.
- `resolve_heatmap_color_norm` uses the full minimum/maximum, including those
  provisional values.
- `trial_committed` refreshes live trial state without reloading grid metadata,
  so an earlier provisional grid value can persist while its live curve improves.

An isolated normalization check with the measured completed range plus an
illustrative pending value of 50 compresses completed cells into the bottom
15.2% of the log colormap. The completed range alone spans the full colormap.
This is a reproduced mechanism consistent with the screenshot, not proof of
the exact provisional value in that screenshot.

Proposed correction: derive automatic color limits from completed finite
points of the selected search. Display incomplete cells distinctly without
letting their provisional values determine those limits. Define a fallback
when there are no completed points; retain genuine completed outliers rather
than silently clipping them. Cover log/linear normalization and live refresh
in regression tests.

No executable code or artifact was changed during this investigation.
