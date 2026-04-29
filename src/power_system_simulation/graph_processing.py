import networkx as nx


# IDNotFoundError child class of Exception class
class IDNotFoundError(Exception):
    def __init__(
        self,
        id: int,
        message: str
    ) -> None:
        #Call base exception class to handle message
        super().__init__(message)

        self.id = id
    pass


class InputLengthDoesNotMatchError(Exception):
    def __init__(
        self,
        first_length: int,
        second_length: int,
        message: str
    ) -> None:
        #Call base exception class to handle message
        super().__init__(message)

        self.first_length = first_length
        self.second_length = second_length
    pass


class IDNotUniqueError(Exception):
    pass


class GraphNotFullyConnectedError(Exception):
    pass


class GraphCycleError(Exception):
    pass


class EdgeAlreadyDisabledError(Exception):
    pass


class GraphProcessor:
    """
    General documentation of this class.
    You need to describe the purpose of this class and the functions in it.
    We are using an undirected graph in the processor.
    """



    def __init__(
        self,
        vertex_ids: list[int],
        edge_ids: list[int],
        vertex_edge_id_pairs: list[tuple[int, int]],
        edge_enabled: list[bool],
        source_vertex_id: int,
    ) -> None:
        """
        Initialize a graph processor object with an undirected graph.
        Only the edges which are enabled are taken into account.
        Check if the input is valid and raise exceptions if not.
        The following conditions should be checked:
            1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
            2. edge_vertex_id_pairs should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            3. edge_vertex_id_pairs should contain valid vertex ids. (IDNotFoundError)
            4. edge_enabled should have the same length as edge_ids. (InputLengthDoesNotMatchError)
            5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
            6. The graph should be fully connected. (GraphNotFullyConnectedError)
            7. The graph should not contain cycles. (GraphCycleError)
        If one certain condition is not satisfied, the error in the parentheses should be raised.

        Args:
            vertex_ids: list of vertex ids
            edge_ids: liest of edge ids
            edge_vertex_id_pairs: list of tuples of two integer
                Each tuple is a vertex id pair of the edge.
            edge_enabled: list of bools indicating of an edge is enabled or not
            source_vertex_id: vertex id of the source in the graph
        """

        """
        Looking at the implementation use of the graph as given in exercise 2, we can observe that load components
        can be described using nodes. A graph can be described using graph matrix representation. Applying this format
        ensures compatibility with SciPy package
        """
        ## put your implementation here
        ## Check for conditions 1 - 5
        # 1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
        vertex_set = set(vertex_ids)
        edge_set = set(edge_ids)

        self._check_uniqueness(vertex_ids, edge_ids)

        # 2. edge_vertex_id_pairs should have the same length as edge_ids
        if len(vertex_ids) != len(edge_ids):
            raise InputLengthDoesNotMatchError(len(vertex_ids), len(edge_ids), "vertex_ids and edge_ids do not contain same number of elements.")

        # 3. vertex_edge_id_pairs should contain valid vertex ids. (IDNotFoundError)
        for vertex, edge in vertex_edge_id_pairs:
            if vertex not in vertex_set:
                raise IDNotFoundError(f"Vertex {vertex} not found in vertex set.")
            if edge not in edge_set:
                raise IDNotFoundError(f"Edge {edge} not found in edge set.")

        # 4. edge_enabled should have the same length as edge_ids.
        if len(edge_ids) is not len(edge_enabled):
            raise InputLengthDoesNotMatchError(len(edge_ids), len(edge_enabled), "edge_ids and edge_enabled do not contain same number of elements.")

        # 5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
        if source_vertex_id not in vertex_set:
            raise IDNotFoundError(source_vertex_id, "source_vertex_id not contained in vertex set.")

        ## Build graph matrix
        graph = nx.Graph()
        graph.add_nodes_from(vertex_ids)

        # Connect vertices and edges

        for i, (vertex, edge) in enumerate(vertex_edge_id_pairs):
            if edge_enabled[i] == True:
                graph.add()


        ## Check for conditions 6-7
        # Check if all components are connected using "connected_components(csgraph)"
        # Check for cycles using depth first search

        pass

    def _check_uniqueness(vertex_ids: list[int], edge_ids: list[int]):
        vertex_set = set(vertex_ids)
        edge_set = set(edge_ids)

        if len(vertex_set) != len(vertex_ids):
            raise IDNotUniqueError("vertex_ids are not unique.")
        if len(edge_set) != len(edge_ids):
            raise IDNotUniqueError("edge_ids are not unique.")

    def find_downstream_vertices(self, edge_id: int) -> list[int]:
        """
        Given an edge id, return all the vertices which are in the downstream of the edge,
            with respect to the source vertex.
            Including the downstream vertex of the edge itself!

        Only enabled edges should be taken into account in the analysis.
        If the given edge_id is a disabled edge, it should return empty list.
        If the given edge_id does not exist, it should raise IDNotFoundError.


        For example, given the following graph (all edges enabled):

            vertex_0 (source) --edge_1-- vertex_2 --edge_3-- vertex_4

        Call find_downstream_vertices with edge_id=1 will return [2, 4]
        Call find_downstream_vertices with edge_id=3 will return [4]

        Args:
            edge_id: edge id to be searched

        Returns:
            A list of all downstream vertices.
        """
        # put your implementation here
        pass

    def find_alternative_edges(self, disabled_edge_id: int) -> list[int]:
        """
        Given an enabled edge, do the following analysis:
            If the edge is going to be disabled,
                which (currently disabled) edge can be enabled to ensure
                that the graph is again fully connected and acyclic?
            Return a list of all alternative edges.
        If the disabled_edge_id is not a valid edge id, it should raise IDNotFoundError.
        If the disabled_edge_id is already disabled, it should raise EdgeAlreadyDisabledError.
        If there are no alternative to make the graph fully connected again, it should return empty list.


        For example, given the following graph:

        vertex_0 (source) --edge_1(enabled)-- vertex_2 --edge_9(enabled)-- vertex_10
                 |                               |
                 |                           edge_7(disabled)
                 |                               |
                 -----------edge_3(enabled)-- vertex_4
                 |                               |
                 |                           edge_8(disabled)
                 |                               |
                 -----------edge_5(enabled)-- vertex_6

        Call find_alternative_edges with disabled_edge_id=1 will return [7]
        Call find_alternative_edges with disabled_edge_id=3 will return [7, 8]
        Call find_alternative_edges with disabled_edge_id=5 will return [8]
        Call find_alternative_edges with disabled_edge_id=9 will return []

        Args:
            disabled_edge_id: edge id (which is currently enabled) to be disabled

        Returns:
            A list of alternative edge ids.
        """
        # put your implementation here
        pass
