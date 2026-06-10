"""Pytest coverage of the §6.1 resource model (agent_specs/00_core_warehouse.md).

Every check here is a *measurement* compared against the closed-form/replayed
prediction — see warehouse_adopt/bench_resource.py for the machinery. CPU
checks are exact equalities; CUDA checks are formula bounds on
``torch.cuda.max_memory_allocated``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from warehouse_adopt.bench_resource import (
    TRAIN_SCHEDULE,
    compute_sizes,
    build_bench_warehouse,
    measure_config,
    predict_offline,
    predict_online,
    verify_config,
)

CUDA = torch.cuda.is_available()


def test_offline_counters_match_formula(tmp_path: Path) -> None:
    meas = measure_config("offline", None, "cpu", tmp_path / "wh")
    verify_config(meas)   # raises on any mismatch
    # Spot-check the headline numbers explicitly:
    W = meas.sizes.model_bytes
    T = len(TRAIN_SCHEDULE)
    assert meas.train.counters["pool_to_device_bytes"] == 2 * T * W   # ref + update per step
    assert meas.train.counters["disk_read_bytes"] == 2 * T * W
    assert meas.train.counters["disk_write_bytes"] == T * W           # eager writeback
    assert meas.pool_resident_bytes == 0                              # pool lives on disk


def test_online_cpu_counters_match_replay(tmp_path: Path) -> None:
    meas = measure_config("online", "cpu", "cpu", tmp_path / "wh")
    verify_config(meas)
    W = meas.sizes.model_bytes
    T = len(TRAIN_SCHEDULE)
    # Swap traffic never exceeds the offline bound of two full models/step,
    # and this schedule (with shared blocks between configs) beats it.
    assert meas.train.counters["pool_to_device_bytes"] < 2 * T * W
    assert meas.train.counters["device_to_pool_bytes"] == T * W       # full update writeback
    assert meas.train.counters["disk_read_bytes"] == 0                # pool prefetched
    assert meas.train.counters["disk_write_bytes"] == 0               # deferred to flush
    assert meas.pool_resident_bytes == meas.sizes.pool_bytes          # exactly N·W in host RAM


def test_predictions_are_engine_agnostic_in_totals(tmp_path: Path) -> None:
    """Same schedule ⇒ same persisted result; engines differ only in traffic."""
    wh = build_bench_warehouse(tmp_path / "wh")
    sizes = compute_sizes(wh)
    off, on = predict_offline(sizes), predict_online(sizes)
    # Identical logical work...
    assert off["train"]["assembles"] == on["train"]["assembles"]
    assert off["train"]["device_to_pool_bytes"] == on["train"]["device_to_pool_bytes"]
    # ...but offline always moves 2 full models/step while online moves only diffs:
    assert on["train"]["pool_to_device_bytes"] < off["train"]["pool_to_device_bytes"]
    # and total bytes that eventually reach disk agree (eager vs deferred):
    eager = off["train"]["disk_write_bytes"]
    deferred = on["flush"]["disk_write_bytes"]
    assert deferred <= eager   # flush collapses repeated writes to the same slice


@pytest.mark.skipif(not CUDA, reason="CUDA peak-memory checks need a GPU")
def test_cuda_peaks_offline(tmp_path: Path) -> None:
    meas = measure_config("offline", None, "cuda", tmp_path / "wh")
    verify_config(meas)   # asserts: inference < 1.9·W; training in [3.6, 4.85]·Wp


@pytest.mark.skipif(not CUDA, reason="CUDA peak-memory checks need a GPU")
def test_cuda_peaks_online_cpu_pool(tmp_path: Path) -> None:
    meas = measure_config("online", "cpu", "cuda", tmp_path / "wh")
    verify_config(meas)
    # Pool on host: VRAM bounds identical to offline; host pool exact.
    assert meas.pool_resident_bytes == meas.sizes.pool_bytes


@pytest.mark.skipif(not CUDA, reason="CUDA pool-residency checks need a GPU")
def test_cuda_peaks_online_cuda_pool(tmp_path: Path) -> None:
    meas = measure_config("online", "cuda", "cuda", tmp_path / "wh")
    verify_config(meas)   # bounds include the +N·W pool term
    # The pool itself is on the GPU and its size is exact:
    assert meas.pool_resident_bytes == meas.sizes.pool_bytes
    # And the measured engine-attributable VRAM proves the pool was resident:
    assert meas.train_total() >= meas.sizes.pool_bytes + 3 * meas.sizes.param_bytes
