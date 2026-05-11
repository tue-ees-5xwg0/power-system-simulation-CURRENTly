import pytest

import power_system_simulation.graph_processing as gp


class TestError(Exception):
    pass


# check uniequeness subfunction
@pytest.mark.parametrize(
    "list, expected_result, expected_error",
    [
        ([1,2,3,4], True, None),
        ([1,2,2,3], False, "IDNotUniqueError"),
    ],
)
def test_check_uniqueness(list: list[int], expected_result: bool, expected_error: str|None) -> None:
    if expected_result:
        gp.GraphProcessor._check_uniqueness(list=list)

    match expected_error:
        case "IDNotUniqueError":
            with pytest.raises(gp.IDNotUniqueError):
                gp.GraphProcessor._check_uniqueness(list=list)
                return
        case _:
            raise TestError(f"Could not match {expected_error} to error cases.")
    return

# check uniqueness edge vertex subfunction
@pytest.mark.parametrize(
    "vertex_ids, edge_ids, expected_result, expected_error",
    [
        ([1, 2, 3], [4, 5, 6], True, None),  # Unique vertex and edge ids
        ([1, 2, 2], [4, 5, 6], False, "VertexIDNotUnique"),  # Duplicate vertex id
        ([1, 2, 3], [4, 5, 5], False, "EdgeIDNotUnique"),  # Duplicate edge id
    ],
)
def test_check_uniqueness_edge_vertex(vertex_ids: list[int], edge_ids: list[int],
                                      expected_result: bool, expected_error: str|None) -> None:
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

# check lenth edge pairs subfunction InputLengthDoesNotMatchError
@pytest.mark.parametrize(
    "vertex_edge_id_pairs, edge_ids, expected_result, expected_error",
    [
        ([(10,1), (11,1), (11,2), (12,2)], [1,2], True, None), # [10] -1- [11] -2- [12]
        ([(10,1), (10,2), (11,2), (11,3)], [1,2,3], False, "InputLengthDoesNotMatchError"), # -1- [10] -2- [11] -3-
        ([(10,1), (11,2)], [1,2], False, "InputLengthDoesNotMatchError"), # [10] -1- -2- [11]
    ],
)
def test_check_length_edge_pairs(vertex_edge_id_pairs: list[tuple[int,int]], edge_ids: list[int], expected_result: bool, expected_error: str|None) -> None:
    if expected_result:
        gp.GraphProcessor._check_length_edge_pairs(vertex_edge_id_pairs=vertex_edge_id_pairs, edge_ids=edge_ids)
        return

    match expected_error:
        case "InputLengthDoesNotMatchError":
            with pytest.raises(gp.InputLengthDoesNotMatchError):
                gp.GraphProcessor._check_length_edge_pairs(vertex_edge_id_pairs=vertex_edge_id_pairs, edge_ids=edge_ids)
                return
        case _:
            raise TestError(f"Could not match {expected_error} to error cases.")
    return

# test check valid vertex edge id pairs subfunction
@pytest.mark.parametrize(
    "vertex_edge_id_pairs, vertex_ids, edge_ids, expected_result, expected_error",
    [
        ([(10,1),(11,1),(11,2),(12,2)], [10,11,12], [1,2], True, None), # [10] -1- [11] -2- [12]
        ([(9,1),(11,1),(11,2),(12,2)], [10,11,12], [1,2], False, "IDNotFoundError"), #Undefined vertex
        ([(10,1),(11,1),(11,2),(12,3)], [10,11,12], [1,2], False, "IDNotFoundError"), #Undefined edge
    ]
)
def test_check_valid_vertex_edge_id_pairs(vertex_edge_id_pairs: list[tuple[int, int]], vertex_ids: list[int], edge_ids: list[int], expected_result: bool, expected_error: str|None) -> None:
    if expected_result:
        gp.GraphProcessor._check_valid_vertex_edge_id_pairs(vertex_edge_id_pairs=vertex_edge_id_pairs, vertex_ids=vertex_ids, edge_ids=edge_ids)
        return

    match expected_error:
        case "IDNotFoundError":
            with pytest.raises(gp.IDNotFoundError):
                gp.GraphProcessor._check_valid_vertex_edge_id_pairs(vertex_edge_id_pairs=vertex_edge_id_pairs, vertex_ids=vertex_ids, edge_ids=edge_ids)
                return
        case _:
            raise TestError(f"Could not match {expected_error} to error cases.")
    return

#test check length edge enabled
@pytest.mark.parametrize(
    "edge_enabled, edge_ids, expected_result, expected_error",
    [
        ([False, True, True],[1,2,3],True, None),
        ([False, False, False, True], [1,2,3], False, "InputLengthDoesNotMatchError"),
        ([False, False], [1,2,3], False, "InputLengthDoesNotMatchError"),
    ],
)
def test_check_length_edge_enabled(edge_enabled: int[bool], edge_ids: int[int], expected_result: bool, expected_error: str|None) -> None:
    if expected_result:
        gp.GraphProcessor._check_length_edge_enabled(edge_enabled=edge_enabled, edge_ids=edge_ids)
        return

    match expected_error:
        case "InputLengthDoesNotMatchError":
            with pytest.raises(gp.InputLengthDoesNotMatchError):
                gp.GraphProcessor._check_length_edge_enabled(edge_enabled=edge_enabled, edge_ids=edge_ids)
                return
        case _:
            raise TestError(f"Could not match {expected_error} to error cases.")
    return

#test check valid id
@pytest.mark.parametrize(
    "id, list, expected_result, expected_error",
    [
        (1, [1,2,3,4], True, None),
        (5, [1,2,3,4], False, "IDNotFoundError"),
    ],
)
def test_check_valid_id(id: int, list: list[int], expected_result: bool, expected_error: str|None) -> None:
    if expected_result:
        gp.GraphProcessor._check_valid_id(id=id, list=list)
        return

    match expected_error:
        case "IDNotFoundError":
            with pytest.raises(gp.IDNotFoundError):
                gp.GraphProcessor._check_valid_id(id=id, list=list)
                return
        case _:
            raise TestError(f"Could not match {expected_error} to error cases.")
    return
