Viewer refresh workflow (artifact-first)
=========================================

Status
------

Implemented refactor (2026-05-30): heartbeat is a wake-up signal; viewer reads live
state exclusively from the consolidated artifact.

Motivation
----------

During adaptive ``(a,b)`` searches the runner and viewer previously duplicated live
state in ``artifact.h5.refresh`` (active ``(a,b)``, q0 trial arrays, search ids).
That caused coordinate mismatches: trial curves came from the artifact while map
loads used stale heartbeat coordinates.

Design principle
----------------

**Runner writes truth → artifact. Heartbeat says "read again". Viewer reads artifact.**

Heartbeat contract (``<artifact>.h5.refresh``)
----------------------------------------------

Minimal JSON payload::

    {
      "phase": "trial 03 complete",
      "slice_key": "mw_2p873584ghz",
      "search_id": "search_a4c3736655362921"
    }

Fields
~~~~~~

``phase`` (required hint)
    Human-readable progress marker. Viewer uses substring matching to choose reload
    depth (see below). Examples: ``initialized``, ``point saved``, ``trial 03 complete``,
    ``scan complete``.

``slice_key`` (optional routing hint)
    Tells the viewer which slice the runner is working on (Active session auto-follow
    navigation). Not a substitute for artifact data.

``search_id`` (optional routing hint)
    Tells the viewer which search group is active. Not a substitute for artifact data.

Legacy fields (ignored by viewer after refactor)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``active_point``, ``pending_points``, ``live_trials`` — retained only for backward
compatibility in old runner sessions; the viewer must not merge them into display state.

Phase → viewer action
---------------------

+---------------------------+------------------------------------------+
| Phase contains            | Viewer action                            |
+===========================+==========================================+
| ``saved``, ``promoted``,  | Full ``load_scan_file()`` reload         |
| ``initialized``,          | (grid point landed, search promoted)     |
| ``resume``, ``loaded``,   |                                          |
| ``scan complete/failed/   |                                          |
| interrupted``             |                                          |
+---------------------------+------------------------------------------+
| ``trial``, ``active``,    | Lightweight sync:                        |
| other in-progress markers | ``load_live_trial_point()`` + UI refresh |
+---------------------------+------------------------------------------+
| (mtime change, empty)     | Lightweight sync                         |
+---------------------------+------------------------------------------+

Runner liveness
---------------

Detected by any of:

* Fresh ``.refresh`` mtime (within grace window)
* Runner PID in ``<artifact>.h5.log`` still running

Artifact sources for Active mode
--------------------------------

Trial curves, active cell, next trial, and per-trial maps all come from the search
group on the selected slice:

``searches/<search_id>/live_trial_point``
    attrs: ``a``, ``b``, ``q0``, ``trial_index``, ``metric_name``, ``updated_utc``
    datasets: ``fit_q0_trials``, ``fit_metric_trials``
    ``trial_history_json`` → ``raw_map_ref`` entries under ``/map_store/``

``searches/<search_id>`` lifecycle attrs
    ``in_progress``, ``active``, ``status`` — which search owns live work

Slice ``common``
    ``observed``, ``psf_kernel`` — same as saved grid points

Display pipeline (live == saved)
--------------------------------

1. User selects trial index (slider / click on q0 curve).
2. Resolve ``raw_map_ref`` from ``trial_history`` (live) or point record (saved).
3. Load raw map from ``map_store``; convolve with PSF from ``common``.
4. Build observed / modeled / residual in Selected Solution panel.

Heatmap semantics during a live search
--------------------------------------

The ``(a,b)`` heatmap uses **one fill color per cell**: the **best target-metric
value committed so far** for that grid point (the metric selected in the viewer
toolbar, typically ``eta2``). This applies to **finished and in-progress**
cells alike.

Partial / active cells (intended behaviour)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cell can be **live** (yellow star, runner working on it) and **already painted**
at the same time. The two overlays mean different things:

+------------------+-------------------------------------------------------------+
| Visual           | Meaning                                                     |
+==================+=============================================================+
| **Fill color**   | Best committed metric at that ``(a,b)`` so far              |
+------------------+-------------------------------------------------------------+
| **Gray dashed  | Grid point not finished yet (``pending`` / ``RUNNING``)       |
| outline**        |                                                             |
+------------------+-------------------------------------------------------------+
| **Yellow star**  | Active live cell: runner is here now; shown once at least   |
| (``active``)     | one trial is committed or a trial is streaming              |
+------------------+-------------------------------------------------------------+
| **Purple star**  | Assigned footprint cell not yet evaluated (expand/resume)   |
+------------------+-------------------------------------------------------------+

Rules implemented in ``viewer.py`` (``_draw_heatmap``):

* A cell is **filled** when its grid record has a **finite** value for the
  display metric — even if ``status`` is still ``pending`` (artifact
  ``RUNNING``).
* A cell stays **outline-only** when it is ``missing``, ``pending`` with **no**
  finite metric yet, or a ``live_pending`` ghost (active coordinates not yet
  present in the artifact).
* When a point finishes (``COMPLETED``), the fill remains the final best metric;
  the dashed pending outline and yellow active star move on to the next cell.

Trials panel
~~~~~~~~~~~~

For the active ``(a,b)``, the Q₀ curve shows **committed trial history** as soon
as trials land in the artifact, including while the point is still ``RUNNING``.
The footer line (``trial #N/M q0=… metric=… [best]``) tracks the current best
over committed trials, consistent with the heatmap fill.

Why this design
~~~~~~~~~~~~~~~

Earlier viewer builds treated all ``pending`` cells as empty (“No data yet”), which
hid useful partial curves during long Q₀ searches. Painting the **best-so-far**
metric gives immediate feedback as trials commit, without waiting for the full
point to complete.

Related tests: ``test_draw_trials_shows_partial_history_for_running_pending_point``
in ``tests/test_viewer_scan_state.py``.

Implementation checklist
------------------------

* [x] Simplify ``_ViewerRefreshHeartbeat._write_locked()`` to phase + routing hints
* [x] ``update_from_live_snapshot()`` sets phase only (no trial arrays in heartbeat)
* [x] Viewer ``_apply_refresh_signal_payload()`` — phase + routing hints only
* [x] Remove heartbeat overlay merge in ``_sync_live_trial_state_from_artifact()``
* [x] Populate ``_refresh_signal_active_point`` / ``_refresh_signal_live_trials`` from artifact only
* [x] Update tests for new contract

Related files
-------------

* ``examples/python/adaptive_ab_search_single_observation.py`` — runner, heartbeat
* ``src/pychmp/viewer.py`` — poll, sync, display
* ``src/pychmp/ab_scan_artifacts.py`` — ``write_live_trial_point``, ``load_live_trial_point``,
  ``load_live_trial_plot_payload``
* ``tests/test_viewer_scan_state.py`` — viewer contract tests
* ``tests/test_adaptive_single_observation_cli.py`` — heartbeat writer tests
