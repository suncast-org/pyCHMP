# Viewer badge redesign: Running while PID alive

**Branch:** `design/moddir-compatible-map-store`  
**Date:** 2026-10-02  
**Status:** implement with step 6 (Gelu-approved plan); precedence amended 2026-10-02

## Problem

After a search process dies or is killed, the badge can still look “live”
from stale heartbeat/refresh signals, while Incomplete/Interrupted are
hard to tell apart from Running. Conversely, a live PID must stay
Running even if status fields lag — **unless** the selected search has
already recorded an authoritative terminal outcome.

## Rules

1. **RUNNING** while the search runner PID (from `{artifact}.log`) is alive
   and targets this selection (or another slice — still Running with a
   “live scan slice” hint), **and** the selected search is not yet
   terminal. Fresh non-terminal refresh heartbeats may reinforce Running,
   but must not keep Running after the PID is gone.
2. **INCOMPLETE** only after the process is gone, when the selected
   search is not successfully complete and did not fail/abort. Applies for
   both empty (`total == 0`) and non-empty grids: leftover `active` /
   heartbeat markers alone must not become INTERRUPTED.
3. **INTERRUPTED** when the process is gone and phase/status indicates
   failure/abort/interrupt (including phrases like `adaptive search failed`).
4. **FINISHED** only for authoritative successful completion of the
   *selected* search (explicit `complete` / valid `completed_at`), never
   from an inactive marker alone or a stale `completed_at` before
   `started_at` (see FINISHED-banner fix).

## Precedence (PID alive vs terminal status)

**Authoritative complete / failed status wins over a still-alive PID.**

If the selected search already records successful completion (or a terminal
failure), show **FINISHED** / **INTERRUPTED** even when `os.kill(pid, 0)`
still succeeds. Operators care that the search finished; a delayed process
exit, cleanup hang, or orphaned PID must not keep the badge at RUNNING.

While the search is still non-terminal (`in_progress`, etc.), a live PID
keeps the badge at **RUNNING** (rule 1).

`PermissionError` / `EPERM` from `os.kill(pid, 0)` means the process exists
but is not signalable by this user — treat as **alive**, not dead.

## Non-goals

- Do not change Stored-maps / Active-Best lock overlays.
- Do not invent a separate FAILED badge; keep INTERRUPTED for failure.
