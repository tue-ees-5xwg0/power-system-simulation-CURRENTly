"""Tests for power_system_simulation.pgm_runner.

These build a tiny synthetic network and load profiles in memory, persist
them to disk in PGM JSON + parquet formats, then exercise the runner.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from power_grid_model import AttributeType, ComponentType, DatasetType, LoadGenType, initialize_array
from power_grid_model.utils import json_serialize

from power_system_simulation.pgm_runner import PowerFlowRunner, ProfileLoadIdError, ProfileMismatchError

# ---------------------------------------------------------------------------
# Tiny synthetic grid:  source(10) -- node(1) -- line(3) -- node(2) -- sym_load(4)
# ---------------------------------------------------------------------------


def _build_tiny_input() -> dict:
    """Construct a minimal valid PGM input dataset in memory."""
    node = initialize_array(DatasetType.input, ComponentType.node, 2)
    node[AttributeType.id] = [1, 2]
    node[AttributeType.u_rated] = [10.5e3, 10.5e3]

    line = initialize_array(DatasetType.input, ComponentType.line, 1)
    line[AttributeType.id] = [3]
    line[AttributeType.from_node] = [1]
    line[AttributeType.to_node] = [2]
    line[AttributeType.from_status] = [1]
    line[AttributeType.to_status] = [1]
    line[AttributeType.r1] = [0.25]
    line[AttributeType.x1] = [0.2]
    line[AttributeType.c1] = [10e-6]
    line[AttributeType.tan1] = [0.0]
    line[AttributeType.i_n] = [1000]

    sym_load = initialize_array(DatasetType.input, ComponentType.sym_load, 1)
    sym_load[AttributeType.id] = [4]
    sym_load[AttributeType.node] = [2]
    sym_load[AttributeType.status] = [1]
    sym_load[AttributeType.type] = [LoadGenType.const_power]
    sym_load[AttributeType.p_specified] = [10e6]
    sym_load[AttributeType.q_specified] = [2e6]

    source = initialize_array(DatasetType.input, ComponentType.source, 1)
    source[AttributeType.id] = [10]
    source[AttributeType.node] = [1]
    source[AttributeType.status] = [1]
    source[AttributeType.u_ref] = [1.0]

    return {
        ComponentType.node: node,
        ComponentType.line: line,
        ComponentType.sym_load: sym_load,
        ComponentType.source: source,
    }


def _write_network(tmp_path: Path) -> Path:
    """Serialize the tiny grid to a JSON file inside tmp_path."""
    data = _build_tiny_input()
    serialized = json_serialize(data)
    p = tmp_path / "network.json"
    p.write_text(serialized)
    return p


def _write_profiles(
    tmp_path: Path,
    *,
    load_ids: list[int] | None = None,
    n_timestamps: int = 3,
    active_offset: float = 0.0,
    swap_reactive_columns: bool = False,
    different_timestamps: bool = False,
) -> tuple[Path, Path]:
    """Write a matching pair of (active, reactive) profile parquet files."""
    if load_ids is None:
        load_ids = [4]
    timestamps = pd.date_range("2024-01-01", periods=n_timestamps, freq="h")

    active = pd.DataFrame(
        np.ones((n_timestamps, len(load_ids))) * 8e6 + active_offset,
        index=timestamps,
        columns=load_ids,
    )
    reactive_timestamps = (
        pd.date_range("2025-06-01", periods=n_timestamps, freq="h") if different_timestamps else timestamps
    )
    reactive_cols = list(reversed(load_ids)) if swap_reactive_columns else load_ids
    reactive = pd.DataFrame(
        np.ones((n_timestamps, len(load_ids))) * 1.5e6,
        index=reactive_timestamps,
        columns=reactive_cols,
    )
    active_p = tmp_path / "active.parquet"
    reactive_p = tmp_path / "reactive.parquet"
    active.to_parquet(active_p)
    reactive.to_parquet(reactive_p)
    return active_p, reactive_p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_end_to_end_runs(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    output = runner.run()

    # Output should contain at least node and line results.
    assert ComponentType.node in output
    assert ComponentType.line in output
    # Batch dimension equals number of timestamps.
    assert output[ComponentType.node].shape[0] == 3
    # Voltages are positive numbers.
    u_pu = output[ComponentType.node][AttributeType.u_pu]
    assert np.all(u_pu > 0)


def test_load_network_returns_input_data(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    input_data = runner.load_network()
    assert ComponentType.node in input_data
    assert ComponentType.line in input_data


def test_build_batch_update_shapes(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path, n_timestamps=5)
    runner = PowerFlowRunner(network, active, reactive)
    runner.load_network()
    runner.load_profiles()
    update = runner.build_batch_update()
    arr = update[ComponentType.sym_load]
    # (n_timestamps, n_loads)
    assert arr.shape == (5, 1)


def test_profiles_with_different_shapes_raise(tmp_path: Path):
    # Active has 3 timestamps, reactive has 5.
    network = _write_network(tmp_path)
    timestamps_a = pd.date_range("2024-01-01", periods=3, freq="h")
    timestamps_r = pd.date_range("2024-01-01", periods=5, freq="h")
    active = pd.DataFrame(np.ones((3, 1)) * 1e6, index=timestamps_a, columns=[4])
    reactive = pd.DataFrame(np.ones((5, 1)) * 1e5, index=timestamps_r, columns=[4])
    active_p = tmp_path / "active.parquet"
    reactive_p = tmp_path / "reactive.parquet"
    active.to_parquet(active_p)
    reactive.to_parquet(reactive_p)

    runner = PowerFlowRunner(network, active_p, reactive_p)
    runner.load_network()
    with pytest.raises(ProfileMismatchError):
        runner.load_profiles()


def test_profiles_with_different_timestamps_raise(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path, different_timestamps=True)
    runner = PowerFlowRunner(network, active, reactive)
    runner.load_network()
    with pytest.raises(ProfileMismatchError):
        runner.load_profiles()


def test_profile_load_id_not_in_network_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    # Load id 999 is not in the network (only sym_load is id 4).
    active, reactive = _write_profiles(tmp_path, load_ids=[999])
    runner = PowerFlowRunner(network, active, reactive)
    runner.load_network()
    with pytest.raises(ProfileLoadIdError):
        runner.load_profiles()


def test_load_profiles_before_network_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    with pytest.raises(RuntimeError):
        runner.load_profiles()


def test_build_batch_before_profiles_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    runner.load_network()
    with pytest.raises(RuntimeError):
        runner.build_batch_update()


def test_invalid_network_raises(tmp_path: Path):
    # A network with a line whose to_node doesn't exist.
    data = _build_tiny_input()
    data[ComponentType.line][AttributeType.to_node] = [999]
    bad = tmp_path / "bad_network.json"
    bad.write_text(json_serialize(data))
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(bad, active, reactive)
    # PGM's own validator raises — we let it bubble up.
    with pytest.raises(Exception):  # noqa: B017, BLE001
        runner.load_network()


def test_malformed_network_json_raises(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a pgm dataset"}))
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(bad, active, reactive)
    with pytest.raises(Exception):  # noqa: B017, BLE001
        runner.load_network()

def test_aggregate_power_flow(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    runner.run()
    df = runner.aggregate_power_flow()

    # Should have one row per timestamp
    assert len(df) == 3
    # Should have the right columns
    assert "Maximum voltage (pu)" in df.columns
    assert "Maximum voltage node ID" in df.columns
    assert "Minimum voltage (pu)" in df.columns
    assert "Minimum voltage node ID" in df.columns
    # Max voltage should always be >= min voltage
    assert (df["Maximum voltage (pu)"] >= df["Minimum voltage (pu)"]).all()


def test_node_table(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    runner.run()
    df = runner.node_table()

    # Should have one row per line (only 1 line in tiny network)
    assert len(df) == 1
    # Should have the right columns
    assert "Energy loss (kWh)" in df.columns
    assert "Maximum loading (pu)" in df.columns
    assert "Minimum loading (pu)" in df.columns
    # Energy loss should be positive
    assert (df["Energy loss (kWh)"] >= 0).all()
    # Max loading should be >= min loading
    assert (df["Maximum loading (pu)"] >= df["Minimum loading (pu)"]).all()


def test_aggregate_power_flow_before_run_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    with pytest.raises(RuntimeError):
        runner.aggregate_power_flow()


def test_node_table_before_run_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive = _write_profiles(tmp_path)
    runner = PowerFlowRunner(network, active, reactive)
    with pytest.raises(RuntimeError):
        runner.node_table()
