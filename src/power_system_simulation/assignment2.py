import time

import numpy as np
import pandas as pd
from networkx import display
from power_grid_model import (CalculationMethod, CalculationType,
                              ComponentType, DatasetType, PowerGridModel,
                              initialize_array)
from power_grid_model.validation import assert_valid_input_data

np.random.seed(0) # For reproducibility

class Timer:       #used to time calculations and print the execution time, not sure if it is needed/required
    def __init__(self, name: str):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        print(f'Execution time for {self.name} is {(time.perf_counter() - self.start):0.6f} s')

def load_input_data() -> dict[str, np.ndarray]:
    input_data = {}
    for component in [ComponentType.node, ComponentType.line, ComponentType.source, ComponentType.sym_load]:

        # Use pandas to read CSV data
        df = pd.read_csv(f'../data/{component.value}.csv')  ##This needs to be change to be able to handle the right data

        # Initialize array
        input_data[component] = initialize_array(DatasetType.input, component, len(df))

        # Fill the attributes
        for attr, values in df.items():
            input_data[component][attr] = values

        # Print some debug info
        print(f"{component:9s}: {len(input_data[component]):4d}")

    return input_data

def load_profile_data(active_profile: pd.DataFrame, reactive_profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not active_profile.index.equals(reactive_profile.index):
        raise ValueError("Timestamps of active and reactive profiles do not match!")
    # Check load ids match
    if not active_profile.columns.equals(reactive_profile.columns):
        raise ValueError("Load IDs of active and reactive profiles do not match!")
    return active_profile, reactive_profile


def validate_input_data(input_data: dict[str, np.ndarray]):
    assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow)

def construct_model(input_data: dict[str, np.ndarray]) -> PowerGridModel:
    model = PowerGridModel(input_data=input_data)
    return model

def power_flow_calculation(input_data: dict[str, np.ndarray], model: PowerGridModel):
    output_data = model.calculate_power_flow(update_data=input_data, method=CalculationMethod.newton_raphson)
    return output_data
def table_per_timestamp(output_data, timestamps) -> pd.DataFrame:
    # Create a table for the current timestamp
    table_data = []

    return pd.DataFrame(table_data)

def table_per_line(result, output_data, timestamps) -> pd.DataFrame:
    # Create a table for each line`
    max_voltage = []
    min_voltage = []
    for idx, _timestamp in enumerate(result[ComponentType.node]["id"]):
        max_voltage.append(max(result[ComponentType.node][idx]))
        min_voltage.append(min(result[ComponentType.node][idx]))

    table_data  = pd.DataFrame(
         {
         "Maximum voltage": max_voltage,
     ##   "ID of maximum voltage": max_voltage_id,
        "Minimum voltage": min_voltage,
      ##  "ID of minimum voltage": min_voltage_id,
    },
    index=_timestamp
    )
    display(table_data)
