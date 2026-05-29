from collections import defaultdict
from pathlib import Path


# Error classes
class FileNotFoundError(Exception):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path
    pass

class FileEmptyError(Exception):
    pass

class DataMissingError(Exception):
    pass

# Object structures

class PowerNode:
    def __init__(self, u_rated: float) -> None:
        self.u_rated: float = u_rated
        pass

class PowerLine:
    def __init__(self, from_node: int, to_node: int, from_status: int, to_status: int, r1: float, x1: float, c1: float, tan1: float, i_n: float) -> None:
        self.from_node: int = from_node
        self.to_node: int = to_node
        self.from_status: int = from_status
        self.to_status: int = to_status
        self.r1: float = r1
        self.x1: float = x1
        self.c1: float = c1
        self.tan1: float = tan1
        self.i_n: float = i_n
        pass

class PowerLoad:
    def __init__(self, node: int, status: int, type: int, p: float, q: float) -> None:
        self.node: int = node
        self.status: int = status
        self.type: int = type
        self.p: float = p
        self.q: float = q
        pass

class PowerSource:
    def __init__(self, id: int, node:int, status: int, u_ref: float, sk: float) -> None:
        self.id: int = id
        self.node: int = node
        self.status: int = status
        self.u_ref: float = u_ref
        self.sk: float = sk
        pass

class PowerProfilePerTimestamp:
    def __init__(self, node_power: dict[int, float]) -> None:
        self.node_power: dict[int, float] = node_power
        pass

# Main class for parsing and storing power grid model data
class PowerGridModelData:
    def __init__(self, network_path: str, p_profile_path: str, q_profile_path: str) -> None:
        self.network_path: str = network_path
        self.p_profile_path: str = p_profile_path
        self.q_profile_path: str = q_profile_path
        self.is_populated: bool = False

        self.nodes: defaultdict[int, PowerNode] = defaultdict(list)
        self.lines: defaultdict[int, PowerLine] = defaultdict(list)
        self.load: defaultdict[int, PowerLoad] = defaultdict(list)
        self.source: PowerSource = None

        self.p_profile: defaultdict[str, PowerProfilePerTimestamp] = defaultdict(list)
        self.q_profile: defaultdict[str, PowerProfilePerTimestamp] = defaultdict(list)

        self.load_model()

    #instance method
    def load_model(self) -> None:
        # Check if files exists
        PowerGridModelData._does_file_exist(self.network_path)
        PowerGridModelData._does_file_exist(self.p_profile_path)

        # Parse file network
        with open(self.network_path, encoding='utf-8') as file:
            network_data = json.load(file).get("data", {})

        # Check if file contains data
        PowerGridModelData._CheckFileContainsData(network_data)

        # Parse data into structured objects
        self.nodes: defaultdict[int, PowerNode] = PowerGridModelData._ParseNodes(network_data) # Parse nodes
        self.lines: defaultdict[int, PowerLine] = PowerGridModelData._ParseLines(network_data) # Parse lines
        self.load: defaultdict[int, PowerLoad] = PowerGridModelData._ParseLoads(network_data) # Parse loads
        self.source: PowerSource = PowerGridModelData._ParseSource(network_data) # Parse source
        # Set is_populated to True if parsing is successful, otherwise False

    @staticmethod
    def _does_file_exist(path_str: str) -> bool:
        # Check if file exists at the given path
        path: Path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path=path_str, message=f"File not found at path: {path_str}")

        return True

    @staticmethod
    def _IsDictEmpty(data: dict) -> bool:
        return not bool(data)

    @staticmethod
    def  _CheckFileContainsData(data: dict) -> None:
        if PowerGridModelData._IsDictEmpty(data):
            raise FileEmptyError("File does not contain data.")
        return True

    @staticmethod
    def _CheckSectionContainsData(section_data: dict, section: str) -> None:
        if PowerGridModelData._IsDictEmpty(section_data):
            raise DataMissingError(f"Section \"{section}\" does not contain data.")
        return True

    @staticmethod
    def _ParseNodes(network_data: dict) -> defaultdict[int, PowerNode]:
        # Load and check section data
        entries = network_data.get("nodes",[])
        PowerGridModelData._CheckSectionContainsData(entries, "nodes")

        # Prepare structured data
        nodes: defaultdict[int, PowerNode] = defaultdict(list)

        # Parse entries into structured objects
        for entry in entries:
            
