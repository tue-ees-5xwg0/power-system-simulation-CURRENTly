"""
LV Grid for Assignment 3.

Loads a low-voltage grid (PGM JSON), an active and a reactive load profile,
an EV charging profile pool, and a list of LV feeder line IDs.  Performs
every input validity check required by the assignment in ``__init__``;
once construction succeeds, the resulting object holds validated data that
the other three A3 functionalities (EV penetration, optimal tap position,
N-1) can consume.

Validation rules (from the assignment text)
-------------------------------------------
1.  The LV grid is valid PGM input data.
2.  The grid has exactly one transformer and one source.
3.  All feeder IDs are valid line IDs.
4.  All feeder lines have ``from_node`` equal to the transformer's ``to_node``.
5.  The grid is fully connected in the initial state.
6.  The grid has no cycles in the initial state.
7.  Timestamps match across the active load profile, reactive load profile
    and EV charging profile.
8.  IDs in the active load profile and reactive load profile match.
9.  IDs in the active/reactive profiles are valid ``sym_load`` IDs.
10. The number of EV charging profiles is at least the number of ``sym_load``.

Rules 5 and 6 reuse ``GraphProcessor`` from Assignment 1 (its constructor
raises ``GraphNotFullyConnectedError`` / ``GraphCycleError`` if violated).
"""

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from power_grid_model import (
    AttributeType,
    CalculationType,
    ComponentType,
    DatasetType,
    PowerGridModel,
    initialize_array,
)
from power_grid_model.utils import json_deserialize
from power_grid_model.validation import assert_valid_input_data

from power_system_simulation.graph_processing import GraphProcessor

# ---------------------------------------------------------------------------
# Custom exceptions specific to A3
# ---------------------------------------------------------------------------


class TransformerCountError(Exception):
    """The LV grid does not contain exactly one transformer."""


class SourceCountError(Exception):
    """The LV grid does not contain exactly one source."""


class FeederIdNotALineError(Exception):
    """A feeder ID is not a valid line ID in the network."""


class FeederNotAtTransformerError(Exception):
    """A feeder line does not start at the transformer's LV side."""


class ProfileTimestampMismatchError(Exception):
    """Timestamps do not match across the three profiles."""


class ProfileLoadIdMismatchError(Exception):
    """Active and reactive profile columns do not match each other."""


class ProfileLoadIdNotSymLoadError(Exception):
    """A column in the load profiles is not a sym_load ID in the network."""


class InsufficientEvProfilesError(Exception):
    """There are fewer EV profiles than sym_loads in the network."""


class OptimizationCriteria(Enum):
    MIN_ENERGY_LOSS = "min_energy_loss"
    MIN_VOLTAGE_DEVIATION = "min_voltage_deviation"


# ---------------------------------------------------------------------------
# LVGrid
# ---------------------------------------------------------------------------


class LVGrid:
    """Validated bundle of the inputs the A3 functionalities need.

    Parameters
    ----------
    network_json_path
        Path to the PGM JSON network file (low-voltage grid).
    active_profile_path
        Parquet file with the active-power load profile per ``sym_load``.
    reactive_profile_path
        Parquet file with the reactive-power load profile per ``sym_load``.
    ev_profile_path
        Parquet file with the EV charging profile pool.  Columns are
        sequence numbers (0, 1, ...), not ``sym_load`` IDs.
    feeder_ids
        Line IDs that mark the beginning of each LV feeder.

    Attributes
    ----------
    input_data : dict
        PGM input dataset (numpy structured arrays per component type).
    active_profile, reactive_profile, ev_profile : pandas.DataFrame
        Loaded profiles.
    feeder_ids : list[int]
        Validated copy of the feeder IDs.
    graph : GraphProcessor
        Tree representation of the *initial-state* grid (only lines with
        both ``from_status`` and ``to_status`` set to 1 are enabled).  The
        source vertex is the transformer's LV-side node.

    Raises
    ------
    Anything from PGM's ``assert_valid_input_data`` (check 1).
    TransformerCountError, SourceCountError (check 2).
    FeederIdNotALineError (check 3).
    FeederNotAtTransformerError (check 4).
    GraphNotFullyConnectedError, GraphCycleError from
        ``GraphProcessor`` (checks 5 and 6).
    ProfileTimestampMismatchError (check 7).
    ProfileLoadIdMismatchError (check 8).
    ProfileLoadIdNotSymLoadError (check 9).
    InsufficientEvProfilesError (check 10).
    """

    def __init__(
        self,
        network_json_path: str | Path,
        active_profile_path: str | Path,
        reactive_profile_path: str | Path,
        ev_profile_path: str | Path,
        feeder_ids: list[int],
    ) -> None:
        self.network_json_path = Path(network_json_path)
        self.active_profile_path = Path(active_profile_path)
        self.reactive_profile_path = Path(reactive_profile_path)
        self.ev_profile_path = Path(ev_profile_path)
        self.feeder_ids: list[int] = list(feeder_ids)

        # ---- check 1: PGM-valid network ----------------------------------
        with self.network_json_path.open() as fp:
            raw = fp.read()
        self.input_data: dict = json_deserialize(raw)
        # Passthrough: ValidationException bubbles up.
        assert_valid_input_data(
            input_data=self.input_data,
            calculation_type=CalculationType.power_flow,
        )

        # ---- check 2: exactly one transformer and one source -------------
        transformers = self.input_data.get(ComponentType.transformer)
        if transformers is None or len(transformers) != 1:
            n = 0 if transformers is None else len(transformers)
            raise TransformerCountError(f"Expected exactly one transformer, found {n}.")
        sources = self.input_data.get(ComponentType.source)
        if sources is None or len(sources) != 1:
            n = 0 if sources is None else len(sources)
            raise SourceCountError(f"Expected exactly one source, found {n}.")

        self._transformer = transformers[0]
        self._source = sources[0]

        # ---- check 3: feeder IDs are valid line IDs ----------------------
        lines = self.input_data[ComponentType.line]
        line_ids = {int(x) for x in lines[AttributeType.id]}
        for fid in self.feeder_ids:
            if int(fid) not in line_ids:
                raise FeederIdNotALineError(f"Feeder ID {fid} is not a valid line ID.")

        # ---- check 4: feeder from_node == transformer.to_node ------------
        transformer_to_node = int(self._transformer[AttributeType.to_node])
        # Build a quick map: line_id -> from_node.
        line_from_node = {
            int(lines[AttributeType.id][i]): int(lines[AttributeType.from_node][i]) for i in range(len(lines))
        }
        for fid in self.feeder_ids:
            if line_from_node[int(fid)] != transformer_to_node:
                raise FeederNotAtTransformerError(
                    f"Feeder line {fid} does not start at the transformer's to_node ({transformer_to_node})."
                )

        # ---- checks 5 + 6: connectedness and acyclicity ------------------
        # The "LV grid" graph contains only the nodes that LV lines actually
        # touch.  The MV-side source node is upstream of the transformer and
        # has no line connection, so it is intentionally excluded; including
        # it would always flag the graph as disconnected because the
        # transformer is not modelled as a graph edge.  The LV busbar (the
        # transformer's to_node) is the source vertex.
        edge_ids = [int(x) for x in lines[AttributeType.id]]
        edge_vertex_id_pairs = [
            (
                int(lines[AttributeType.from_node][i]),
                int(lines[AttributeType.to_node][i]),
            )
            for i in range(len(lines))
        ]
        edge_enabled = [
            bool(lines[AttributeType.from_status][i]) and bool(lines[AttributeType.to_status][i])
            for i in range(len(lines))
        ]
        # Vertices = every node id that any line endpoint references.
        vertex_ids = sorted({v for pair in edge_vertex_id_pairs for v in pair})
        # GraphProcessor.__init__ raises GraphNotFullyConnectedError or
        # GraphCycleError if the initial state is bad.
        self.graph = GraphProcessor(
            vertex_ids=vertex_ids,
            edge_ids=edge_ids,
            vertex_edge_id_pairs=edge_vertex_id_pairs,
            edge_enabled=edge_enabled,
            source_vertex_id=transformer_to_node,
        )

        # ---- load profiles -----------------------------------------------
        self.active_profile = pd.read_parquet(self.active_profile_path)
        self.reactive_profile = pd.read_parquet(self.reactive_profile_path)
        self.ev_profile = pd.read_parquet(self.ev_profile_path)

        # ---- check 7: timestamps match across all three profiles ---------
        if not self.active_profile.index.equals(self.reactive_profile.index):
            raise ProfileTimestampMismatchError("Active and reactive load profiles do not share the same timestamps.")
        if not self.active_profile.index.equals(self.ev_profile.index):
            raise ProfileTimestampMismatchError("EV profile timestamps differ from the load profile timestamps.")

        # ---- check 8: active and reactive columns match ------------------
        active_cols = list(self.active_profile.columns)
        reactive_cols = list(self.reactive_profile.columns)
        if active_cols != reactive_cols:
            raise ProfileLoadIdMismatchError("Active and reactive load profiles do not share the same load IDs.")

        # ---- check 9: load profile IDs are valid sym_load IDs ------------
        sym_loads = self.input_data.get(ComponentType.sym_load)
        sym_load_ids = {int(x) for x in sym_loads[AttributeType.id]} if sym_loads is not None else set()
        profile_ids = {int(x) for x in active_cols}
        unknown = profile_ids - sym_load_ids
        if unknown:
            raise ProfileLoadIdNotSymLoadError(
                f"Load profile contains IDs that are not sym_loads in the network: {sorted(unknown)}."
            )

        # ---- check 10: enough EV profiles --------------------------------
        n_sym_loads = 0 if sym_loads is None else len(sym_loads)
        n_ev_profiles = self.ev_profile.shape[1]
        if n_ev_profiles < n_sym_loads:
            raise InsufficientEvProfilesError(
                f"Need at least {n_sym_loads} EV profiles (one per sym_load), but only {n_ev_profiles} were provided."
            )

    # ------------------------------------------------------------------ #
    # Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    @property
    def sym_load_ids(self) -> list[int]:
        sym_loads = self.input_data[ComponentType.sym_load]
        return [int(x) for x in sym_loads[AttributeType.id]]

    @property
    def line_ids(self) -> list[int]:
        lines = self.input_data[ComponentType.line]
        return [int(x) for x in lines[AttributeType.id]]

    @property
    def transformer_id(self) -> int:
        return int(self._transformer[AttributeType.id])

    @property
    def source_id(self) -> int:
        return int(self._source[AttributeType.id])

    @property
    def transformer_lv_node(self) -> int:
        """The LV-side node of the transformer (where the feeders start)."""
        return int(self._transformer[AttributeType.to_node])

    # ------------------------------------------------------------------ #
    # Finding the optimal tap position                                   #
    # ------------------------------------------------------------------ #
    def optimize_tap_position(self, criteria: OptimizationCriteria = OptimizationCriteria.MIN_VOLTAGE_DEVIATION) -> int:
        # Get the tap range from the transformer attributes
        tap_min = int(self._transformer[AttributeType.tap_min])
        tap_max = int(self._transformer[AttributeType.tap_max])
        # swap if wrong order
        tap_lower = min(tap_min, tap_max)  # 1
        tap_upper = max(tap_min, tap_max)  # 5
        # Build load update data for the time series
        n_timesteps = len(self.active_profile)
        n_loads = len(self.sym_load_ids)
        load_update = initialize_array(DatasetType.update, ComponentType.sym_load, (n_timesteps, n_loads))
        load_update[AttributeType.id] = np.tile(self.sym_load_ids, (n_timesteps, 1))
        load_update[AttributeType.p_specified] = self.active_profile.values
        load_update[AttributeType.q_specified] = self.reactive_profile.values
        batch_update = {ComponentType.sym_load: load_update}
        model = PowerGridModel(self.input_data)
        scores = {}

        for tap in range(tap_lower, tap_upper + 1):
            # Update the tap position in input_data directly
            tap_update = initialize_array(DatasetType.update, ComponentType.transformer, 1)
            tap_update[AttributeType.id] = [self.transformer_id]
            tap_update[AttributeType.tap_pos] = [tap]
            model.update(update_data={ComponentType.transformer: tap_update})
            # Run batch power flow over the whole time series
            time_interval_hours = (self.active_profile.index[1] - self.active_profile.index[0]).total_seconds() / 3600
            result = model.calculate_power_flow(update_data=batch_update)
            if criteria == OptimizationCriteria.MIN_ENERGY_LOSS:
                p_from = result[ComponentType.line][AttributeType.p_from]
                p_to = result[ComponentType.line][AttributeType.p_to]
                scores[tap] = float(np.sum(p_from + p_to) * time_interval_hours)

            elif criteria == OptimizationCriteria.MIN_VOLTAGE_DEVIATION:
                u_pu = result[ComponentType.node][AttributeType.u_pu]
                max_dev = np.abs(np.max(u_pu, axis=0) - 1.0)
                min_dev = np.abs(np.min(u_pu, axis=0) - 1.0)
                scores[tap] = float(np.mean((max_dev + min_dev) / 2))
        return min(scores, key=scores.get)
