from collections import defaultdict
from pathlib import Path
from json import load

# Error classes
class FileNotFoundError(Exception):
    def __init__(self, path: str, message: str) -> None:
        super().__init__(message)
        self.path = path
    pass

# Object structures

class PowerNode:
    def __init__(self, u_rated: int):
        self.u_rated: int = u_rated
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
        self.p_profile: defaultdict[str, PowerProfilePerTimestamp] = defaultdict(list)
        self.q_profile: defaultdict[str, PowerProfilePerTimestamp] = defaultdict(list)
        self.source: PowerSource = None

        self.load_model()

    #instance method
    def load_model(self) -> None:
        # Check if files exists
        PowerGridModelData._does_file_exist(self.network_path)
        PowerGridModelData._does_file_exist(self.p_profile_path)

        # Parse file network
        with open(self.network_path, 'r', encoding='utf-8') as f:
            network_data = json.load(f)
        # Set is_populated to True if parsing is successful, otherwise False

    @staticmethod
    def _does_file_exist(path_str: str) -> bool:
        # Check if file exists at the given path
        path: Path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path=path_str, message=f"File not found at path: {path_str}")

        return True
    
    def  
