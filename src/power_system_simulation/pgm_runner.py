"""
PGM Runner for Assignment 2.

Loads a power-grid-model network from a JSON file and a pair of (active,
reactive) load profiles from parquet files, builds a time-series batch
update, and runs the batch power-flow calculation.

Returns the raw PGM output so a downstream aggregation step (Assignment 2
chunk 4) can produce the summary tables.

Validation strategy
-------------------
* JSON deserialization, model construction and the batch calculation use
  PGM's built-in validators; any ``ValidationException`` they raise is
  allowed to propagate (this matches the spec's "passthrough" wording).
* Cross-checks between the two parquet files (matching timestamps and
  matching load IDs) and between the parquet files and the network
  (load IDs are valid sym_load IDs) are done here, since PGM cannot know
  what the parquet files are supposed to contain.
"""

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
from power_grid_model.validation import (
    assert_valid_batch_data,
    assert_valid_input_data,
)

# ---------------------------------------------------------------------------
# Custom exceptions for the cross-file checks
# ---------------------------------------------------------------------------


class ProfileMismatchError(Exception):
    """Raised when active and reactive profiles disagree on shape, timestamps
    or load IDs."""


class ProfileLoadIdError(Exception):
    """Raised when a load ID in the profiles is not a sym_load in the network."""


# ---------------------------------------------------------------------------
# PowerFlowRunner
# ---------------------------------------------------------------------------


class PowerFlowRunner:
    """End-to-end driver for the Assignment 2 time-series power flow.

    Parameters
    ----------
    network_json_path
        Path to the PGM JSON network file.
    active_profile_path
        Path to the parquet file with the active-power load profile.
    reactive_profile_path
        Path to the parquet file with the reactive-power load profile.
    """

    def __init__(
        self,
        network_json_path: str | Path,
        active_profile_path: str | Path,
        reactive_profile_path: str | Path,
    ) -> None:
        self.network_json_path = Path(network_json_path)
        self.active_profile_path = Path(active_profile_path)
        self.reactive_profile_path = Path(reactive_profile_path)

        # Filled in by the build_* methods.
        self.input_data: dict | None = None
        self.active_profile: pd.DataFrame | None = None
        self.reactive_profile: pd.DataFrame | None = None
        self.update_data: dict | None = None
        self.model: PowerGridModel | None = None
        self.output_data: dict | None = None

    # ------------------------------------------------------------------ #
    # Step 1 — load and validate the network                              #
    # ------------------------------------------------------------------ #

    def load_network(self) -> dict:
        """Deserialize the JSON network into PGM's input format and validate it.

        Raises whatever ``assert_valid_input_data`` raises if the file is bad.
        """
        with self.network_json_path.open() as fp:
            raw = fp.read()
        self.input_data = json_deserialize(raw)
        # Passthrough: ValidationException bubbles up.
        assert_valid_input_data(
            input_data=self.input_data,
            calculation_type=CalculationType.power_flow,
        )
        return self.input_data

    # ------------------------------------------------------------------ #
    # Step 2 — load the parquet profiles and cross-check them             #
    # ------------------------------------------------------------------ #

    def load_profiles(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Read both profiles and check they agree with each other and the network.

        Raises:
            ProfileMismatchError: shapes / timestamps / column IDs disagree
                between the two parquet files.
            ProfileLoadIdError: a column in the profiles is not a valid
                sym_load ID in the loaded network.
        """
        if self.input_data is None:
            raise RuntimeError("Call load_network() before load_profiles().")

        self.active_profile = pd.read_parquet(self.active_profile_path)
        self.reactive_profile = pd.read_parquet(self.reactive_profile_path)

        # Same shape?
        if self.active_profile.shape != self.reactive_profile.shape:
            raise ProfileMismatchError(
                f"Active profile shape {self.active_profile.shape} differs from "
                f"reactive profile shape {self.reactive_profile.shape}."
            )

        # Same timestamps (index)?
        if not self.active_profile.index.equals(self.reactive_profile.index):
            raise ProfileMismatchError("Active and reactive profiles do not share the same timestamps.")

        # Same load IDs (columns)?
        active_cols = list(self.active_profile.columns)
        reactive_cols = list(self.reactive_profile.columns)
        if active_cols != reactive_cols:
            raise ProfileMismatchError("Active and reactive profiles do not share the same load IDs.")

        # Profile load IDs must be sym_loads in the network.
        sym_load_ids = {int(x) for x in self.input_data[ComponentType.sym_load][AttributeType.id]}
        profile_ids = {int(x) for x in active_cols}
        unknown = profile_ids - sym_load_ids
        if unknown:
            raise ProfileLoadIdError(
                f"Profile contains load IDs that are not sym_loads in the network: {sorted(unknown)}."
            )

        return self.active_profile, self.reactive_profile

    # ------------------------------------------------------------------ #
    # Step 3 — build the time-series batch update                          #
    # ------------------------------------------------------------------ #

    def build_batch_update(self) -> dict:
        """Reshape the two profiles into a PGM batch update array.

        The resulting shape for sym_load is ``(n_timestamps, n_loads)``,
        with each row holding one timestamp's p_specified / q_specified
        values for every load.
        """
        if self.active_profile is None or self.reactive_profile is None:
            raise RuntimeError("Call load_profiles() before build_batch_update().")

        n_timestamps, n_loads = self.active_profile.shape
        load_ids = np.asarray(self.active_profile.columns, dtype=np.int32)

        load_profile = initialize_array(
            DatasetType.update,
            ComponentType.sym_load,
            (n_timestamps, n_loads),
        )
        # Same load IDs in every row — broadcasted by numpy.
        load_profile[AttributeType.id] = load_ids
        load_profile[AttributeType.p_specified] = self.active_profile.to_numpy()
        load_profile[AttributeType.q_specified] = self.reactive_profile.to_numpy()

        self.update_data = {ComponentType.sym_load: load_profile}

        # Passthrough: ValidationException bubbles up.
        assert_valid_batch_data(
            input_data=self.input_data,
            update_data=self.update_data,
            calculation_type=CalculationType.power_flow,
        )
        return self.update_data

    # ------------------------------------------------------------------ #
    # Step 4 — run the time-series power flow                              #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Convenience: load everything, build the batch, run the calculation.

        Returns the raw PGM batch output dictionary.
        """
        self.load_network()
        self.load_profiles()
        self.build_batch_update()
        self.model = PowerGridModel(self.input_data)
        # Default: fail the whole batch if any scenario raises.
        self.output_data = self.model.calculate_power_flow(
            update_data=self.update_data,
        )
        return self.output_data
