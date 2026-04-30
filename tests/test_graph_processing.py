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
def test_check_uniqueness_edge_vertex(vertex_ids: list[int], edge_ids: list[int],
                                      expected_result: bool, expected_error: str) -> None:
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
