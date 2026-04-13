"""Task models for single-slice `(a, b)` scan execution.

These helpers stay slice-agnostic so the same execution model can later drive
microwave-frequency and EUV-channel searches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ABExecutionPolicy = Literal["serial", "process-pool"]


@dataclass(frozen=True)
class ABSliceTaskDescriptor:
    key: str
    domain: str
    label: str
    display_label: str


@dataclass(frozen=True)
class ABPointTask:
    slice_key: str
    slice_domain: str
    slice_label: str
    slice_display_label: str
    a: float
    b: float
    a_index: int
    b_index: int
    q0_min: float
    q0_max: float
    target_metric: str
    source_kind: str


def compile_rectangular_point_tasks(
    *,
    a_values: np.ndarray | list[float] | tuple[float, ...],
    b_values: np.ndarray | list[float] | tuple[float, ...],
    slice_descriptor: ABSliceTaskDescriptor,
    q0_min: float,
    q0_max: float,
    target_metric: str,
) -> list[ABPointTask]:
    a_arr = np.asarray(a_values, dtype=float)
    b_arr = np.asarray(b_values, dtype=float)
    tasks: list[ABPointTask] = []
    for i, a_value in enumerate(a_arr):
        for j, b_value in enumerate(b_arr):
            tasks.append(
                ABPointTask(
                    slice_key=str(slice_descriptor.key),
                    slice_domain=str(slice_descriptor.domain),
                    slice_label=str(slice_descriptor.label),
                    slice_display_label=str(slice_descriptor.display_label),
                    a=float(a_value),
                    b=float(b_value),
                    a_index=int(i),
                    b_index=int(j),
                    q0_min=float(q0_min),
                    q0_max=float(q0_max),
                    target_metric=str(target_metric),
                    source_kind="rectangular",
                )
            )
    return tasks


def compile_sparse_point_tasks(
    *,
    point_specs: list[tuple[float, float, float | None, float | None]],
    a_values: np.ndarray | list[float] | tuple[float, ...],
    b_values: np.ndarray | list[float] | tuple[float, ...],
    slice_descriptor: ABSliceTaskDescriptor,
    default_q0_min: float,
    default_q0_max: float,
    target_metric: str,
) -> list[ABPointTask]:
    a_arr = np.asarray(a_values, dtype=float)
    b_arr = np.asarray(b_values, dtype=float)
    a_lookup = {float(value): int(index) for index, value in enumerate(a_arr)}
    b_lookup = {float(value): int(index) for index, value in enumerate(b_arr)}

    tasks: list[ABPointTask] = []
    for a_value, b_value, point_q0_min, point_q0_max in point_specs:
        resolved_a = float(a_value)
        resolved_b = float(b_value)
        if resolved_a not in a_lookup:
            raise ValueError(f"sparse point a={resolved_a:.6g} is not present in a_values")
        if resolved_b not in b_lookup:
            raise ValueError(f"sparse point b={resolved_b:.6g} is not present in b_values")
        tasks.append(
            ABPointTask(
                slice_key=str(slice_descriptor.key),
                slice_domain=str(slice_descriptor.domain),
                slice_label=str(slice_descriptor.label),
                slice_display_label=str(slice_descriptor.display_label),
                a=resolved_a,
                b=resolved_b,
                a_index=int(a_lookup[resolved_a]),
                b_index=int(b_lookup[resolved_b]),
                q0_min=float(default_q0_min if point_q0_min is None else point_q0_min),
                q0_max=float(default_q0_max if point_q0_max is None else point_q0_max),
                target_metric=str(target_metric),
                source_kind="sparse",
            )
        )
    return tasks
