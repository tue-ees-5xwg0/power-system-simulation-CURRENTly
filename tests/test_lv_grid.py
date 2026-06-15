"""Tests for power_system_simulation.lv_grid."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from power_grid_model import (
    AttributeType,
    Branch3Side,
    ComponentType,
    DatasetType,
    LoadGenType,
    WindingType,
    initialize_array,
)
from power_grid_model.utils import json_serialize

from power_system_simulation.graph_processing import GraphCycleError, GraphNotFullyConnectedError
from power_system_simulation.lv_grid import (
    FeederIdNotALineError,
    FeederNotAtTransformerError,
    InsufficientEvProfilesError,
    LVGrid,
    OptimizationCriteria,
    ProfileLoadIdMismatchError,
    ProfileLoadIdNotSymLoadError,
    ProfileTimestampMismatchError,
    SourceCountError,
    TransformerCountError,
)

# ---------------------------------------------------------------------------
# Tiny synthetic LV grid, matching the spec's example drawing in miniature.
#
#       source_100 (MV source)
#           |
#         node_0  (MV side)
#           |
#       transformer_50  (MV -> LV)
#           |
#         node_1  (LV busbar)
#         /     \
#      line_2    line_3                <-- feeders 2 and 3
#       /          \
#      node_4    node_5
#       |          |
#      load_6    load_7
# ---------------------------------------------------------------------------


def _build_input() -> dict:
    """Build a valid PGM input dataset matching the LV-grid shape above."""
    node = initialize_array(DatasetType.input, ComponentType.node, 4)
    node[AttributeType.id] = [0, 1, 4, 5]
    node[AttributeType.u_rated] = [10e3, 400.0, 400.0, 400.0]

    transformer = initialize_array(DatasetType.input, ComponentType.transformer, 1)
    transformer[AttributeType.id] = [50]
    transformer[AttributeType.from_node] = [0]  # MV side
    transformer[AttributeType.to_node] = [1]  # LV busbar
    transformer[AttributeType.from_status] = [1]
    transformer[AttributeType.to_status] = [1]
    transformer[AttributeType.u1] = [10e3]
    transformer[AttributeType.u2] = [400.0]
    transformer[AttributeType.sn] = [1e6]
    transformer[AttributeType.uk] = [0.06]
    transformer[AttributeType.pk] = [10e3]
    transformer[AttributeType.i0] = [0.001]
    transformer[AttributeType.p0] = [500.0]
    transformer[AttributeType.winding_from] = [WindingType.delta]
    transformer[AttributeType.winding_to] = [WindingType.wye_n]
    transformer[AttributeType.clock] = [5]
    transformer[AttributeType.tap_side] = [Branch3Side.side_1]
    transformer[AttributeType.tap_pos] = [0]
    transformer[AttributeType.tap_min] = [-3]
    transformer[AttributeType.tap_max] = [3]
    transformer[AttributeType.tap_nom] = [0]
    transformer[AttributeType.tap_size] = [100.0]

    # Two LV feeder lines (from node 1 -> node 4 / node 5).
    line = initialize_array(DatasetType.input, ComponentType.line, 2)
    line[AttributeType.id] = [2, 3]
    line[AttributeType.from_node] = [1, 1]
    line[AttributeType.to_node] = [4, 5]
    line[AttributeType.from_status] = [1, 1]
    line[AttributeType.to_status] = [1, 1]
    line[AttributeType.r1] = [0.1, 0.1]
    line[AttributeType.x1] = [0.05, 0.05]
    line[AttributeType.c1] = [1e-6, 1e-6]
    line[AttributeType.tan1] = [0.0, 0.0]
    line[AttributeType.i_n] = [500.0, 500.0]

    sym_load = initialize_array(DatasetType.input, ComponentType.sym_load, 2)
    sym_load[AttributeType.id] = [6, 7]
    sym_load[AttributeType.node] = [4, 5]
    sym_load[AttributeType.status] = [1, 1]
    sym_load[AttributeType.type] = [LoadGenType.const_power, LoadGenType.const_power]
    sym_load[AttributeType.p_specified] = [1000.0, 1000.0]
    sym_load[AttributeType.q_specified] = [100.0, 100.0]

    source = initialize_array(DatasetType.input, ComponentType.source, 1)
    source[AttributeType.id] = [100]
    source[AttributeType.node] = [0]
    source[AttributeType.status] = [1]
    source[AttributeType.u_ref] = [1.0]

    return {
        ComponentType.node: node,
        ComponentType.transformer: transformer,
        ComponentType.line: line,
        ComponentType.sym_load: sym_load,
        ComponentType.source: source,
    }


def _write_network(tmp_path: Path, data: dict | None = None) -> Path:
    if data is None:
        data = _build_input()
    p = tmp_path / "network.json"
    p.write_text(json_serialize(data))
    return p


def _write_profiles(
    tmp_path: Path,
    *,
    load_ids: list[int] | None = None,
    n_timestamps: int = 4,
    n_ev_profiles: int = 2,
    ev_timestamps_offset: bool = False,
    reactive_columns_swapped: bool = False,
    reactive_timestamps_offset: bool = False,
) -> tuple[Path, Path, Path]:
    if load_ids is None:
        load_ids = [6, 7]
    base_idx = pd.date_range("2025-01-01", periods=n_timestamps, freq="15min")
    active = pd.DataFrame(
        np.ones((n_timestamps, len(load_ids))) * 800.0,
        index=base_idx,
        columns=load_ids,
    )
    reactive_idx = (
        pd.date_range("2026-06-01", periods=n_timestamps, freq="15min") if reactive_timestamps_offset else base_idx
    )
    reactive_cols = list(reversed(load_ids)) if reactive_columns_swapped else load_ids
    reactive = pd.DataFrame(
        np.ones((n_timestamps, len(load_ids))) * 100.0,
        index=reactive_idx,
        columns=reactive_cols,
    )
    ev_idx = pd.date_range("2027-12-12", periods=n_timestamps, freq="15min") if ev_timestamps_offset else base_idx
    ev = pd.DataFrame(
        np.ones((n_timestamps, n_ev_profiles)) * 500.0,
        index=ev_idx,
        columns=list(range(n_ev_profiles)),
    )
    active_p = tmp_path / "active.parquet"
    reactive_p = tmp_path / "reactive.parquet"
    ev_p = tmp_path / "ev.parquet"
    active.to_parquet(active_p)
    reactive.to_parquet(reactive_p)
    ev.to_parquet(ev_p)
    return active_p, reactive_p, ev_p


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_loads(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])
    assert grid.transformer_id == 50
    assert grid.source_id == 100
    assert grid.transformer_lv_node == 1
    assert sorted(grid.sym_load_ids) == [6, 7]
    assert sorted(grid.line_ids) == [2, 3]


# ---------------------------------------------------------------------------
# Check 2: transformer / source count
# ---------------------------------------------------------------------------


def test_zero_transformers_raises(tmp_path: Path):
    data = _build_input()
    # Replace with an empty transformer array.
    data[ComponentType.transformer] = initialize_array(DatasetType.input, ComponentType.transformer, 0)
    network = _write_network(tmp_path, data)
    active, reactive, ev = _write_profiles(tmp_path)
    with pytest.raises(TransformerCountError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


def test_zero_sources_raises(tmp_path: Path):
    data = _build_input()
    data[ComponentType.source] = initialize_array(DatasetType.input, ComponentType.source, 0)
    network = _write_network(tmp_path, data)
    active, reactive, ev = _write_profiles(tmp_path)
    with pytest.raises(SourceCountError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


# ---------------------------------------------------------------------------
# Check 3: feeder IDs are valid lines
# ---------------------------------------------------------------------------


def test_unknown_feeder_id_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    with pytest.raises(FeederIdNotALineError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 999])


# ---------------------------------------------------------------------------
# Check 4: feeders start at transformer LV side
# ---------------------------------------------------------------------------


def test_feeder_not_at_transformer_raises(tmp_path: Path):
    # Tweak line 3 so it starts somewhere other than node 1.
    data = _build_input()
    data[ComponentType.line][AttributeType.from_node][1] = 4  # was 1
    # The line now goes 4 -> 5 (still valid PGM), but it doesn't start at
    # node 1 (the transformer's LV side).  This will produce a disconnected
    # initial state (node 5 unreachable from node 1), so the check we want
    # to verify (FeederNotAtTransformerError) must run *before* the graph
    # connectivity check.
    network = _write_network(tmp_path, data)
    active, reactive, ev = _write_profiles(tmp_path)
    with pytest.raises(FeederNotAtTransformerError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


# ---------------------------------------------------------------------------
# Checks 5 + 6: graph connectivity / acyclicity (via GraphProcessor)
# ---------------------------------------------------------------------------


def test_disconnected_initial_state_raises(tmp_path: Path):
    # Disable line 3 -> node 5 becomes unreachable.
    data = _build_input()
    data[ComponentType.line][AttributeType.from_status][1] = 0
    data[ComponentType.line][AttributeType.to_status][1] = 0
    network = _write_network(tmp_path, data)
    active, reactive, ev = _write_profiles(tmp_path)
    with pytest.raises(GraphNotFullyConnectedError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


def test_cycle_in_initial_state_raises(tmp_path: Path):
    # Add a third line that creates a cycle: node 4 -- node 5.
    data = _build_input()
    new_line = initialize_array(DatasetType.input, ComponentType.line, 3)
    new_line[AttributeType.id] = [2, 3, 99]
    new_line[AttributeType.from_node] = [1, 1, 4]
    new_line[AttributeType.to_node] = [4, 5, 5]
    new_line[AttributeType.from_status] = [1, 1, 1]
    new_line[AttributeType.to_status] = [1, 1, 1]
    new_line[AttributeType.r1] = [0.1, 0.1, 0.1]
    new_line[AttributeType.x1] = [0.05, 0.05, 0.05]
    new_line[AttributeType.c1] = [1e-6, 1e-6, 1e-6]
    new_line[AttributeType.tan1] = [0.0, 0.0, 0.0]
    new_line[AttributeType.i_n] = [500.0, 500.0, 500.0]
    data[ComponentType.line] = new_line
    network = _write_network(tmp_path, data)
    active, reactive, ev = _write_profiles(tmp_path)
    with pytest.raises(GraphCycleError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


# ---------------------------------------------------------------------------
# Check 7: timestamps match across all three profiles
# ---------------------------------------------------------------------------


def test_reactive_profile_timestamps_offset_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path, reactive_timestamps_offset=True)
    with pytest.raises(ProfileTimestampMismatchError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


def test_ev_profile_timestamps_offset_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path, ev_timestamps_offset=True)
    with pytest.raises(ProfileTimestampMismatchError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


# ---------------------------------------------------------------------------
# Check 8: active and reactive load columns match
# ---------------------------------------------------------------------------


def test_reactive_load_columns_swapped_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path, reactive_columns_swapped=True)
    with pytest.raises(ProfileLoadIdMismatchError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


# ---------------------------------------------------------------------------
# Check 9: profile IDs are valid sym_load IDs
# ---------------------------------------------------------------------------


def test_profile_ids_not_sym_loads_raises(tmp_path: Path):
    # Profiles use IDs 999, 998 -- not in the network (network has 6, 7).
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path, load_ids=[999, 998])
    with pytest.raises(ProfileLoadIdNotSymLoadError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


# ---------------------------------------------------------------------------
# Check 10: enough EV profiles
# ---------------------------------------------------------------------------


def test_too_few_ev_profiles_raises(tmp_path: Path):
    # Network has 2 sym_loads, give only 1 EV profile.
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path, n_ev_profiles=1)
    with pytest.raises(InsufficientEvProfilesError):
        LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])


def test_more_ev_profiles_than_loads_ok(tmp_path: Path):
    # Network has 2 sym_loads; 5 EV profiles is fine.
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path, n_ev_profiles=5)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])
    assert grid.ev_profile.shape[1] == 5


# ---------------------------------------------------------------------------
# Check: optimize_tap_position runs and returns a tap within the allowed range
# ---------------------------------------------------------------------------
def test_optimize_tap_position_energy_loss(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    result = grid.optimize_tap_position(criteria=OptimizationCriteria.MIN_ENERGY_LOSS)

    tap_min = int(grid._transformer[AttributeType.tap_min])
    tap_max = int(grid._transformer[AttributeType.tap_max])
    assert isinstance(result, int)
    assert min(tap_min, tap_max) <= result <= max(tap_min, tap_max)


def test_optimize_tap_position_voltage_deviation(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    result = grid.optimize_tap_position(criteria=OptimizationCriteria.MIN_VOLTAGE_DEVIATION)

    tap_min = int(grid._transformer[AttributeType.tap_min])
    tap_max = int(grid._transformer[AttributeType.tap_max])
    assert isinstance(result, int)
    assert min(tap_min, tap_max) <= result <= max(tap_min, tap_max)


def test_optimize_tap_position_default_criteria(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    # should work without passing criteria (uses default)
    result = grid.optimize_tap_position()

    tap_min = int(grid._transformer[AttributeType.tap_min])
    tap_max = int(grid._transformer[AttributeType.tap_max])
    assert isinstance(result, int)
    assert min(tap_min, tap_max) <= result <= max(tap_min, tap_max)
    voltage_table, line_table = grid.simulate_ev_penetration(penetration_level=1.0, seed=42)


def test_ev_penetration_runs(tmp_path: Path):
    """Happy path: at 100% penetration, 2 sym_loads on 2 feeders -> 1 EV each."""
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    voltage_table, line_table = grid.simulate_ev_penetration(penetration_level=1.0, seed=42)

    # 4 timestamps in the test profiles
    assert len(voltage_table) == 4
    assert "max_voltage_pu" in voltage_table.columns
    assert "min_voltage_pu" in voltage_table.columns
    assert (voltage_table["max_voltage_pu"] >= voltage_table["min_voltage_pu"]).all()

    # 2 lines in the test grid
    assert len(line_table) == 2
    assert "energy_loss_kwh" in line_table.columns
    assert "max_loading" in line_table.columns
    assert (line_table["energy_loss_kwh"] >= 0).all()


def test_ev_penetration_accepts_percentage(tmp_path: Path):
    """A value > 1 should be interpreted as a percentage."""
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    voltage_table_pct, _ = grid.simulate_ev_penetration(penetration_level=100, seed=7)
    voltage_table_frac, _ = grid.simulate_ev_penetration(penetration_level=1.0, seed=7)

    # 100% and 1.0 are equivalent, same seed -> identical results
    assert (voltage_table_pct["max_voltage_pu"] == voltage_table_frac["max_voltage_pu"]).all()


def test_ev_penetration_reproducible_with_seed(tmp_path: Path):
    """Same seed -> identical voltage tables."""
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    voltage_a, _ = grid.simulate_ev_penetration(penetration_level=1.0, seed=123)
    voltage_b, _ = grid.simulate_ev_penetration(penetration_level=1.0, seed=123)

    assert (voltage_a["max_voltage_pu"] == voltage_b["max_voltage_pu"]).all()
    assert (voltage_a["min_voltage_pu"] == voltage_b["min_voltage_pu"]).all()


def test_ev_penetration_return_assignment(tmp_path: Path):
    """When return_assignment=True, a third table is returned."""
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    result = grid.simulate_ev_penetration(penetration_level=1.0, seed=1, return_assignment=True)

    assert len(result) == 3
    _, _, assignment_table = result
    # 100% on 2 feeders with 2 houses -> 2 EVs assigned
    assert len(assignment_table) == 2
    assert "feeder_id" in assignment_table.columns
    assert "sym_load_id" in assignment_table.columns
    assert "ev_profile" in assignment_table.columns


def test_ev_penetration_zero_level_ok(tmp_path: Path):
    """Zero penetration: no EVs assigned, but should still run."""
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    voltage_table, line_table = grid.simulate_ev_penetration(penetration_level=0.0, seed=0)
    assert len(voltage_table) == 4
    assert len(line_table) == 2


def test_ev_penetration_negative_raises(tmp_path: Path):
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    with pytest.raises(ValueError, match="cannot be negative"):
        grid.simulate_ev_penetration(penetration_level=-0.1)


def test_ev_penetration_too_high_raises(tmp_path: Path):
    """A value above 100% (after fraction conversion) should raise."""
    network = _write_network(tmp_path)
    active, reactive, ev = _write_profiles(tmp_path)
    grid = LVGrid(network, active, reactive, ev, feeder_ids=[2, 3])

    # 200 -> 2.0 -> still > 1 -> raise
    with pytest.raises(ValueError, match="between 0 and 1"):
        grid.simulate_ev_penetration(penetration_level=200)
