import pytest

from power_system_simulation.graph_processing import GraphProcessor, IDNotUniqueError


#check uniqueness subfunction
@pytest.mark.parametrize(
    "vertex_ids, edge_ids, expected_result",
    [
        ([1, 2, 3], [4, 5, 6], True),  # Unique vertex and edge ids
        ([1, 2, 2], [4, 5, 6], False),  # Duplicate vertex id
        ([1, 2, 3], [4, 5, 5], False),  # Duplicate edge id
    ]
)
def test_check_uniqueness(vertex_ids, edge_ids, expected_result):
    if expected_result:
        GraphProcessor._check_uniqueness(vertex_ids=vertex_ids, edge_ids=edge_ids)
    else:
        with pytest.raises(IDNotUniqueError):
            GraphProcessor._check_uniqueness(vertex_ids=vertex_ids, edge_ids=edge_ids)

