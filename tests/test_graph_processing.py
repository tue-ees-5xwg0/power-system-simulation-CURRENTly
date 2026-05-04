import pytest

import power_system_simulation.graph_processing as gp


class TestError(Exception):
    pass


# check uniqueness subfunction
@pytest.mark.parametrize(
    "vertex_ids, edge_ids, expected_result, expected_error",
    [
        ([1, 2, 3], [4, 5, 6], True, None),  # Unique vertex and edge ids
        ([1, 2, 2], [4, 5, 6], False, "VertexIDNotUnique"),  # Duplicate vertex id
        ([1, 2, 3], [4, 5, 5], False, "EdgeIDNotUnique"),  # Duplicate edge id
    ],
)
def test_check_uniqueness_edge_vertex(
    vertex_ids: list[int], edge_ids: list[int], expected_result: bool, expected_error: str
) -> None:
    # If we expect a positive result, this branch should not raise errors
    if expected_result:
        gp.GraphProcessor._check_uniqueness_edge_vertex(vertex_ids=vertex_ids, edge_ids=edge_ids)
        return

    # If we expect an error, this branch matches error to whichever error we expect to raise FIRST
    match expected_error:
        case "VertexIDNotUnique":
            with pytest.raises(gp.VertexIDNotUnique):
                gp.GraphProcessor._check_uniqueness_edge_vertex(vertex_ids=vertex_ids, edge_ids=edge_ids)
                return
        case "EdgeIDNotUnique":
            with pytest.raises(gp.IDNotUniqueError):
                gp.GraphProcessor._check_uniqueness_edge_vertex(vertex_ids=vertex_ids, edge_ids=edge_ids)
                return
        case _:
            raise TestError(f"Could not match {expected_error} to error cases.")
    return


def _simple_chain():
    """Chain: 0 (source) --edge_1-- 2 --edge_3-- 4."""
    return gp.GraphProcessor(
        vertex_ids=[0, 2, 4],
        edge_ids=[1, 3],
        vertex_edge_id_pairs=[(0, 2), (2, 4)],
        edge_enabled=[True, True],
        source_vertex_id=0,
    )


def test_downstream_first_edge():
    g = _simple_chain()
    assert sorted(g.find_downstream_vertices(1)) == [2, 4]


def test_downstream_second_edge():
    g = _simple_chain()
    assert sorted(g.find_downstream_vertices(3)) == [4]


def test_downstream_unknown_edge_raises():
    g = _simple_chain()
    with pytest.raises(gp.IDNotFoundError):
        g.find_downstream_vertices(999)


def test_downstream_disabled_edge_returns_empty():
    g = gp.GraphProcessor(
        vertex_ids=[0, 2, 4],
        edge_ids=[1, 3, 5],
        vertex_edge_id_pairs=[(0, 2), (2, 4), (0, 4)],
        edge_enabled=[True, True, False],
        source_vertex_id=0,
    )
    assert g.find_downstream_vertices(5) == []


def test_invalid_source_raises():
    with pytest.raises(gp.IDNotFoundError):
        gp.GraphProcessor(
            vertex_ids=[0, 2, 4],
            edge_ids=[1, 3],
            vertex_edge_id_pairs=[(0, 2), (2, 4)],
            edge_enabled=[True, True],
            source_vertex_id=99,
        )


def test_unknown_vertex_in_edge_raises():
    with pytest.raises(gp.IDNotFoundError):
        gp.GraphProcessor(
            vertex_ids=[0, 2, 4],
            edge_ids=[1, 3],
            vertex_edge_id_pairs=[(0, 2), (2, 99)],
            edge_enabled=[True, True],
            source_vertex_id=0,
        )


def test_length_mismatch_raises():
    with pytest.raises(gp.InputLengthDoesNotMatchError):
        gp.GraphProcessor(
            vertex_ids=[0, 2, 4],
            edge_ids=[1, 3],
            vertex_edge_id_pairs=[(0, 2)],
            edge_enabled=[True, True],
            source_vertex_id=0,
        )


def test_not_connected_raises():
    with pytest.raises(gp.GraphNotFullyConnectedError):
        gp.GraphProcessor(
            vertex_ids=[0, 2, 4],
            edge_ids=[1, 3],
            vertex_edge_id_pairs=[(0, 2), (2, 4)],
            edge_enabled=[True, False],
            source_vertex_id=0,
        )


def test_cycle_raises():
    with pytest.raises(gp.GraphCycleError):
        gp.GraphProcessor(
            vertex_ids=[0, 2, 4],
            edge_ids=[1, 3, 5],
            vertex_edge_id_pairs=[(0, 2), (2, 4), (4, 0)],
            edge_enabled=[True, True, True],
            source_vertex_id=0,
        )
