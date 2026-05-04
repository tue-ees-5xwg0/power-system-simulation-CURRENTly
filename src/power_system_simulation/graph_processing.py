from collections import deque

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


# IDNotFoundError child class of Exception class
class IDNotFoundError(Exception):
    def __init__(self, id: int, message: str) -> None:
        super().__init__(message)
        self.id = id

    pass


# SourceVertexNotFoundError child class of IDNotFoundError
class SourceVertexNotFoundError(IDNotFoundError):
    pass


class InputLengthDoesNotMatchError(Exception):
    def __init__(self, first_length: int, second_length: int, message: str) -> None:
        super().__init__(message)
        self.first_length = first_length
        self.second_length = second_length

    pass


class IDNotUniqueError(Exception):
    pass


class VertexIDNotUnique(IDNotUniqueError):
    pass


class EdgeIDNotUnique(IDNotUniqueError):
    pass


class ImproperEdgeConnections(Exception):
    def __init__(self, edge_id: int | None, connections: int, message: str) -> None:
        super().__init__(message)
        self.edge_id = edge_id
        self.connections = connections

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

    # Class attributes
    # Example: Class: dog
    #           species = "Canine"

    # instance method
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
        ## Define instance variables
        self.vertex_ids: list[int] = vertex_ids
        self.edge_ids: list[int] = edge_ids
        self.vertex_edge_id_pairs: list[tuple[int, int]] = vertex_edge_id_pairs
        self.edge_enabled: list[bool] = edge_enabled
        self.source_vertex_id: int = source_vertex_id
        self.edge_dict = None
        self.graph: nx.classes.graph.Graph = None

        ## Basic checks
        # 1. vertex_ids and edge_ids should be unique. (IDNotUniqueError)
        GraphProcessor._check_uniqueness_edge_vertex(self.vertex_ids, self.edge_ids)

        # 2. vertex_edge_id_pairs should have the same length as edge_ids
        GraphProcessor._check_length_edge_pairs(self.vertex_edge_id_pairs, self.edge_ids)

        # 3. vertex_edge_id_pairs should contain valid vertex ids. (IDNotFoundError)
        GraphProcessor._check_valid_vertex_edge_id_pairs(self.vertex_edge_id_pairs, self.vertex_ids, self.edge_ids)

        # 4. edge_enabled should have the same length as edge_ids.
        GraphProcessor._check_length_edge_enabled(self.edge_enabled, self.edge_ids)

        # 5. source_vertex_id should be a valid vertex id. (IDNotFoundError)
        GraphProcessor._check_valid_source_vertex(self.source_vertex_id, self.vertex_ids)

        ## Create graph
        # Create dictionary with edge_id and whether edge is enabled or disabled
        self.edge_dict = GraphProcessor._organise_id_state(self.edge_ids, self.edge_enabled)
        # Pass instance variables to update routine
        self._update_graph()
        pass

    # instance method
    def add_disabled_edges_from(self, edge_enabled: list[int]) -> None:
        """
        Redefine disabled edges. New list should adhere to edge_enabled_new being same length as edge_ids.
        Further more, new active edges should make for a singleton and non-cicular graph.
        """
        ## Prechecks
        # Check length new array
        self._check_length_edge_enabled(edge_enabled, self.edge_ids)

        # Check new array is equivalent to old array
        if np.array_equal(self.edge_enabled, edge_enabled):
            raise EdgeAlreadyDisabledError("Edge_enabled equivalent to instance edge_enabled")

        # Overwrite instance variable
        self.edge_enabled = edge_enabled

        # Update and check graph
        self._update_graph()
        pass

    # instance method
    def toggle_edge(self, edge_id: int, bool: bool) -> None:
        """
        edge_id: id of the edge which should be switched.
        bool: value indicating whether the edge is enabled or disabled.

        Function toggles edge with edge_id to value of boolean. True: Edge enabled. False: Edge disabled.
        """
        # Check if id is valid
        self._check_valid_id(edge_id, self.edge_ids)

        # Check if edge already has current value
        old_val = self.edge_dict.get(edge_id)
        if old_val == bool:
            raise EdgeAlreadyDisabledError(f"Edge {edge_id} already has the value {bool}")

        # Update and check graph
        self._update_graph()
        pass

    # instance method
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
        self._check_valid_id(edge_id, self.edge_ids)

        # If the edge is disabled, we return an empty list.
        if not self.edge_dict.get(edge_id):
            return []

        # Find the (u, v) endpoints of this edge by scanning vertex_edge_id_pairs
        # in the same order as edge_ids.
        edge_index = self.edge_ids.index(edge_id)
        u, v = self.vertex_edge_id_pairs[edge_index]

        # BFS from the source over the enabled subgraph (self.graph) to learn
        # each vertex's parent. parent[source] is None.
        parent: dict[int, int | None] = {self.source_vertex_id: None}
        queue: deque = deque([self.source_vertex_id])
        while queue:
            node = queue.popleft()
            for neighbour in self.graph.neighbors(node):
                if neighbour not in parent:
                    parent[neighbour] = node
                    queue.append(neighbour)

        # Whichever endpoint has the OTHER endpoint as its parent is the
        # downstream root: the side of the edge facing away from the source.
        if parent.get(v) == u:
            downstream_root = v
        elif parent.get(u) == v:
            downstream_root = u
        else:
            return []

        # Collect the entire subtree rooted at downstream_root.
        subtree: list[int] = []
        seen = {downstream_root}
        queue = deque([downstream_root])
        while queue:
            node = queue.popleft()
            subtree.append(node)
            for neighbour in self.graph.neighbors(node):
                if neighbour not in seen and parent.get(neighbour) == node:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return subtree
        pass

    # instance method
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

    @staticmethod
    def _check_uniqueness(list: list[int]) -> None:
        list_set = set(list)
        if len(list_set) != len(list):
            raise IDNotUniqueError("Entries in list are not unique.")

    # @staticmethod makes sure you can use methods of its own class.
    # Be careful, declaring a static method
    # means you cannot mutate instance variables any longer.
    # Please see: GraphProcessor._check_uniqueness(vertex_ids)
    @staticmethod
    def _check_uniqueness_edge_vertex(vertex_ids: list[int], edge_ids: list[int]) -> None:
        # Test vertex ids
        try:
            GraphProcessor._check_uniqueness(vertex_ids)
        except IDNotUniqueError as IDError:
            raise VertexIDNotUnique("Vertex ids are not unique.") from IDError

        # Test edge ids
        try:
            GraphProcessor._check_uniqueness(edge_ids)
        except IDNotUniqueError as IDError:
            raise EdgeIDNotUnique("Vertex ids are not unique.") from IDError
        pass

    @staticmethod
    def _check_length_edge_pairs(vertex_edge_id_pairs: list[tuple[int, int]], edge_ids: list[int]) -> None:
        if len(vertex_edge_id_pairs) != len(edge_ids):
            raise InputLengthDoesNotMatchError(
                len(vertex_edge_id_pairs),
                len(edge_ids),
                "vertex_edge_id_pairs and edge_ids do not contain same number of elements.",
            )

    @staticmethod
    def _check_valid_vertex_edge_id_pairs(
        vertex_edge_id_pairs: list[tuple[int, int]], vertex_ids: list[int], edge_ids: list[int]
    ) -> None:
        vertex_set = set(vertex_ids)

        for u, v in vertex_edge_id_pairs:
            if u not in vertex_set:
                raise IDNotFoundError(u, f"Vertex {u} not found in vertex set.")
            if v not in vertex_set:
                raise IDNotFoundError(v, f"Vertex {v} not found in vertex set.")

    @staticmethod
    def _check_length_edge_enabled(edge_enabled, edge_ids) -> None:
        if len(edge_ids) != len(edge_enabled):
            raise InputLengthDoesNotMatchError(
                len(edge_ids), len(edge_enabled), "edge_ids and edge_enabled do not contain same number of elements."
            )
        pass

    @staticmethod
    def _check_valid_id(id: int, list: list[int]) -> None:
        list_set = set(list)
        if id not in list_set:
            raise IDNotFoundError(id, f"id {id} not found in list.")
        pass

    @staticmethod
    def _check_valid_source_vertex(source_vertex_id: int, vertex_ids: list[int]) -> None:
        try:
            GraphProcessor._check_valid_id(source_vertex_id, vertex_ids)
        except IDNotFoundError as IDError:
            raise SourceVertexNotFoundError(
                source_vertex_id, f"source_vertex_id: {id} not found in vertex_ids."
            ) from IDError
        pass

    @staticmethod
    def _organise_id_state(edge_ids: list[int], edge_enabled: list[bool]) -> dict[int, bool]:
        # 1. edge_ids should be unique.
        GraphProcessor._check_uniqueness(edge_ids)

        # 4. edge_enabled should have the same length as edge_ids.
        GraphProcessor._check_length_edge_enabled(edge_enabled, edge_ids)

        # Build dictionary
        edge_dict: dict[int, bool] = {}

        for i in range(len(edge_enabled)):
            edge_dict[edge_ids[i]] = edge_enabled[i]

        return edge_dict

    @staticmethod
    def _pair_vertices(
        vertex_edge_id_pairs: list[tuple[int, int]],
        edge_ids: list[int],
        edge_dict: dict[int, bool],
    ) -> list[tuple[int, int]]:
        # Pair nodes specified by edge_connected
        connectivity: list[tuple[int, int]] = []
        for edge_id, (u, v) in zip(edge_ids, vertex_edge_id_pairs):
            # Check for improper number of connections
            if edge_dict.get(edge_id):
                connectivity.append((u, v))
        return connectivity

    @staticmethod
    def _self_build_graph(connectivity: list[tuple[int, int]], vertex_ids: list[int]) -> nx.classes.graph.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(vertex_ids)
        graph.add_edges_from(connectivity)
        return graph

    @staticmethod
    def _nx_graph_to_scipy(graph: nx.classes.graph.Graph) -> sp._csr.csr_array:
        return nx.to_scipy_sparse_array(graph)

    @staticmethod
    def _check_graph_connected(graph: sp._csr.csr_array) -> None:
        n_components, _ = csgraph.connected_components(graph)
        if n_components != 1:
            raise GraphNotFullyConnectedError("Provided graph is not singelton.")
        pass

    @staticmethod
    def _check_contain_cycles(graph: nx.classes.graph.Graph) -> None:
        try:
            nx.find_cycle(graph)
        except nx.exception.NetworkXNoCycle:
            return
        raise GraphCycleError("Graph contains cycles")

    # instancemethod
    def _update_graph(self) -> None:
        # Determine connectivity matrix
        connectivity = GraphProcessor._pair_vertices(self.vertex_edge_id_pairs, self.edge_ids, self.edge_dict)

        # Connect vertices
        graph = GraphProcessor._self_build_graph(connectivity, self.vertex_ids)

        ## Graph checks
        # 6. The graph should be fully connected.
        GraphProcessor._check_graph_connected(GraphProcessor._nx_graph_to_scipy(graph))

        # 7. The graph should not contain cycles.
        GraphProcessor._check_contain_cycles(graph)

        # Assign created and checked graph to instance graph
        self.graph = graph
