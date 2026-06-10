import sys
from pathlib import Path

from power_grid_model import ComponentType

sys.path.append(str(Path(__file__).parent.parent / r"src\power_system_simulation"))  # tests/ → root → src/

from pgm_runner import PowerFlowRunner

DATA_DIR = Path(__file__).parent  # goes up from tests/ to project root

runner = PowerFlowRunner(
    network_json_path=DATA_DIR / "input_network_data.json",
    active_profile_path=DATA_DIR / "active_power_profile.parquet",
    reactive_profile_path=DATA_DIR / "reactive_power_profile.parquet",
)
runner.run()
print(runner.output_data[ComponentType.line].dtype.names)

print("=== Voltage Table ===")
print(runner.aggregate_power_flow())

print("=== Line Table ===")
print(runner.node_table())
