"""Atomic refresh v2 sidecar writer for pychmp-view routing hints."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

REFRESH_V2_VERSION = 2

REFRESH_EVENTS = frozenset(
    {
        "search_initialized",
        "point_assigned",
        "trial_committed",
        "point_completed",
        "point_failed",
        "search_completed",
    }
)


class RefreshSignalWriter:
    """Write lightweight refresh v2 payloads atomically."""

    def __init__(self, signal_path: Path) -> None:
        self._signal_path = Path(signal_path)
        self._lock = threading.Lock()
        self._sequence = 0
        self._slice_key: str | None = None
        self._search_id: str | None = None

    def set_routing(self, *, slice_key: str | None = None, search_id: str | None = None) -> None:
        with self._lock:
            if slice_key is not None:
                self._slice_key = None if not str(slice_key).strip() else str(slice_key).strip()
            if search_id is not None:
                self._search_id = None if not str(search_id).strip() else str(search_id).strip()

    def write_event(
        self,
        event: str,
        *,
        slice_key: str | None = None,
        search_id: str | None = None,
        point_id: str | None = None,
        trial_index: int | None = None,
        legacy_phase: str | None = None,
    ) -> None:
        event_name = str(event).strip()
        if event_name not in REFRESH_EVENTS:
            raise ValueError(f"unsupported refresh event: {event_name}")
        payload: dict[str, Any] = {
            "version": REFRESH_V2_VERSION,
            "event": event_name,
            "slice_key": slice_key if slice_key is not None else self._slice_key,
            "search_id": search_id if search_id is not None else self._search_id,
            "sequence": 0,
        }
        if legacy_phase is not None and str(legacy_phase).strip():
            payload["phase"] = str(legacy_phase).strip()
        if event_name == "trial_committed":
            if trial_index is None:
                raise ValueError("trial_committed requires trial_index")
            payload["trial_index"] = int(trial_index)
            if point_id is None:
                raise ValueError("trial_committed requires point_id")
            payload["point_id"] = str(point_id)
        elif event_name in {"point_assigned", "point_completed", "point_failed"}:
            if point_id is None:
                raise ValueError(f"{event_name} requires point_id")
            payload["point_id"] = str(point_id)
        elif event_name in {"search_initialized", "search_completed"}:
            pass
        with self._lock:
            self._sequence += 1
            payload["sequence"] = int(self._sequence)
            if payload.get("slice_key"):
                self._slice_key = str(payload["slice_key"])
            if payload.get("search_id"):
                self._search_id = str(payload["search_id"])
            self._write_atomic(payload)

    def _write_atomic(self, payload: dict[str, Any]) -> None:
        tmp_path = self._signal_path.with_suffix(self._signal_path.suffix + ".tmp")
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        try:
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(tmp_path, self._signal_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
