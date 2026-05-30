from json import load
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

class MultipleSourcesError(Exception):
    pass

class IdNotUniqueError(Exception):
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
    def __init__(self, node: int, status: int, type: int, p_specified: float, q_specified: float) -> None:
        self.node: int = node
        self.status: int = status
        self.type: int = type
        self.p_specified: float = p_specified
        self.q_specified: float = q_specified
        pass

class PowerSource:
    def __init__(self, node: int, status: int, u_ref: float, sk: float) -> None:
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

        self.nodes: dict[int, PowerNode] = {}
        self.lines: dict[int, PowerLine] = {}
        self.load: dict[int, PowerLoad] = {}
        self.source: dict[int, PowerSource] = {}

        self.p_profile: dict[str, PowerProfilePerTimestamp] = {}
        self.q_profile: dict[str, PowerProfilePerTimestamp] = {}

        self.load_model()

    #instance method
    def load_model(self) -> None:
        # Check if files exists
        PowerGridModelData._does_file_exist(self.network_path)
        PowerGridModelData._does_file_exist(self.p_profile_path)
        PowerGridModelData._does_file_exist(self.q_profile_path)

        # Parse file network
        with open(self.network_path, encoding='utf-8') as file:
            network_data = load(file).get("data", {})

        # Check if file contains data
        PowerGridModelData._CheckFileContainsData(network_data)

        # Parse data into structured objects
        self.nodes: dict[int, PowerNode] = PowerGridModelData._ParseNodes(network_data) # Parse nodes
        self.lines: dict[int, PowerLine] = PowerGridModelData._ParseLines(network_data) # Parse lines
        self.load: dict[int, PowerLoad] = PowerGridModelData._ParseLoads(network_data) # Parse loads
        self.source: dict[int, PowerSource] = PowerGridModelData._ParseSource(network_data) # Parse source

        # Check for data consistency and integrity
        PowerGridModelData._CheckSingleSource(self.source) # Check for single source
        PowerGridModelData._CheckUniqueIds(self.nodes, self.lines, self.load, self.source) # Check for unique IDs

        # Set is_populated to True if parsing is successful, otherwise False

    @staticmethod
    def _does_file_exist(path_str: str) -> bool:
        # Check if file exists at the given path
        path: Path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(path=path_str, message=f"File not found at path: {path_str}")

        return True

    @staticmethod
    def _IsDictEmpty(data: any) -> bool:
        return not bool(data)

    @staticmethod
    def  _CheckFileContainsData(data: dict) -> None:
        if PowerGridModelData._IsDictEmpty(data):
            raise FileEmptyError("File does not contain data.")
        return True

    @staticmethod
    def _CheckSectionContainsData(section_data: list, section: str) -> None:
        if PowerGridModelData._IsDictEmpty(section_data):
            raise DataMissingError(f"Section \"{section}\" does not contain data.")
        return True

    @staticmethod
    def _ParseNodes(network_data: dict) -> dict[int, PowerNode]:
        # Load and check section data
        entries = network_data.get("node",[])
        PowerGridModelData._CheckSectionContainsData(entries, "node")

        # Prepare structured data
        nodes: dict[int, PowerNode] = {}

        # Parse entries into structured objects
        for entry in entries:
            # Extract fields and check type congruence
            id = int(entry["id"])
            u_rated = float(entry["u_rated"])

            # Create PowerNode object and add to nodes dictionary
            nodes[id] = PowerNode(u_rated=u_rated)
        return nodes

    @staticmethod
    def _ParseLines(network_data: dict) -> dict[int, PowerLine]:
        # Load and check section data
        entries = network_data.get("line",[])
        PowerGridModelData._CheckSectionContainsData(entries, "line")

        # Prepare structured data
        lines: dict[int, PowerLine] = {}

        # Parse entries into structured objects
        for entry in entries:
            # Extract fields and check type congruence
            id = int(entry["id"])
            from_node = int(entry["from_node"])
            to_node = int(entry["to_node"])
            from_status = int(entry["from_status"])
            to_status = int(entry["to_status"])
            r1 = float(entry["r1"])
            x1 = float(entry["x1"])
            c1 = float(entry["c1"])
            tan1 = float(entry["tan1"])
            i_n = float(entry["i_n"])

            # Create PowerLine object and add to lines dictionary
            lines[id] = PowerLine(from_node=from_node, to_node=to_node, from_status=from_status, to_status=to_status, r1=r1, x1=x1, c1=c1, tan1=tan1, i_n=i_n)
        return lines

    @staticmethod
    def _ParseLoads(network_data: dict) -> dict[int, PowerLoad]:
        # Load and check section data
        entries = network_data.get("load",[])
        PowerGridModelData._CheckSectionContainsData(entries, "sym_load")

        # Prepare structured data
        loads: dict[int, PowerLoad] = {}

        # Parse entries into structured objects
        for entry in entries:
            # Extract fields and check type congruence
            id = int(entry["id"])
            node = int(entry["node"])
            status = int(entry["status"])
            type = int(entry["type"])
            p_specified = float(entry["p_specified"])
            q_specified = float(entry["q_specified"])

            # Create PowerLoad object and add to loads dictionary
            loads[id] = PowerLoad(node=node, status=status, type=type, p_specified=p_specified, q_specified=q_specified)
        return loads

    @staticmethod
    def _ParseSource(network_data: dict) -> dict[int, PowerSource]:
        # Load and check section data
        entries = network_data.get("source",[])
        PowerGridModelData._CheckSectionContainsData(entries, "source")

        sources: dict[int, PowerSource] = {}

        for entry in entries:
            # Extract fields and check type congruence
            id = int(entry["id"])
            node = int(entry["node"])
            status = int(entry["status"])
            u_ref = float(entry["u_ref"])
            sk = float(entry["sk"])

            # Create PowerSource object and add to sources dictionary
            sources[id] = PowerSource(node=node, status=status, u_ref=u_ref, sk=sk)
        return sources

    @staticmethod
    def _CheckSingleSource(sources: dict[int, PowerSource]) -> None:
        if len(sources.keys()) > 1:
            raise MultipleSourcesError("Multiple sources found in the network data. Only one source is allowed.")
        return True

    @staticmethod
    def _CheckUniqueIds(nodes: dict[int, PowerNode], lines: dict[int, PowerLine], loads: dict[int, PowerLoad], sources: dict[int, PowerSource]) -> None:
        all_ids = []
        all_ids.extend(nodes.keys())
        all_ids.extend(lines.keys())
        all_ids.extend(loads.keys())
        all_ids.extend(sources.keys())
        if len(all_ids) != len(set(all_ids)):
            raise IdNotUniqueError("Duplicate IDs found across nodes, lines, loads, or sources. All IDs must be unique.")
        return



