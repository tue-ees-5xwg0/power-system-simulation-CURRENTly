import json
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

import power_system_simulation.parser as parser

# --- Global Test File Paths ---
# Simply update these strings whenever your file locations change
NETWORK_PATH = "tests/test_files/network_config.json"
ACTIVE_POWER_PATH = "tests/test_files/active_power.parquet"
REACTIVE_POWER_PATH = "tests/test_files/reactive_power.parquet"


# --- Mock Data Fixtures ---


@pytest.fixture
def valid_network_data() -> dict:
    return {
        "node": [{"id": 1, "u_rated": 110.0}, {"id": 2, "u_rated": 110.0}],
        "line": [
            {
                "id": 10,
                "from_node": 1,
                "to_node": 2,
                "from_status": 1,
                "to_status": 1,
                "r1": 0.1,
                "x1": 0.2,
                "c1": 0.01,
                "tan1": 0.0,
                "i_n": 100.0,
            }
        ],
        "load": [{"id": 20, "node": 2, "status": 1, "type": 1, "p_specified": 50.0, "q_specified": 10.0}],
        "source": [{"id": 30, "node": 1, "status": 1, "u_ref": 1.0, "sk": 1000.0}],
    }


@pytest.fixture
def mock_profile_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 2 loads (IDs 20), 2 timestamps format matches %d-%b-%Y %H:%M:%S
    timestamps = pd.to_datetime(["2026-06-10 10:00:00", "2026-06-10 11:00:00"])
    df_p = pd.DataFrame({20: [10.5, 12.0]}, index=timestamps)
    df_q = pd.DataFrame({20: [2.1, 3.0]}, index=timestamps)
    return df_p, df_q


# --- Testing Data Classes Directly ---


def test_power_node_instantiation() -> None:
    node = parser.PowerNode(u_rated=220.0)
    assert node.u_rated == 220.0


def test_power_line_instantiation() -> None:
    line = parser.PowerLine(
        from_node=1, to_node=2, from_status=1, to_status=1, r1=0.5, x1=0.6, c1=0.02, tan1=0.1, i_n=150.0
    )
    assert line.from_node == 1
    assert line.x1 == 0.6


def test_power_load_instantiation() -> None:
    load = parser.PowerLoad(node=3, status=1, type=2, p_specified=100.0, q_specified=25.0)
    assert load.node == 3
    assert load.q_specified == 25.0


def test_power_source_instantiation() -> None:
    source = parser.PowerSource(node=1, status=1, u_ref=1.05, sk=500.0)
    assert source.node == 1
    assert source.sk == 500.0


def test_power_profile_per_timestamp_instantiation() -> None:
    profile = parser.PowerProfilePerTimestamp(node_power={20: 10.5})
    assert profile.node_power[20] == 10.5


# --- Testing PowerGridModelData Internal Functions ---


def test_does_file_exist() -> None:
    # Pass this current test file's path—it is guaranteed to exist!
    assert parser.PowerGridModelData._does_file_exist(__file__) is True

    # Test an invalid path that definitely does not exist
    with pytest.raises(parser.FileNotFoundError):
        parser.PowerGridModelData._does_file_exist("this_file_definitely_does_not_exist.json")


def test_is_dict_empty() -> None:
    assert parser.PowerGridModelData._IsDictEmpty({}) is True
    assert parser.PowerGridModelData._IsDictEmpty({"has_data": True}) is False


def test_check_file_contains_data() -> None:
    assert parser.PowerGridModelData._CheckFileContainsData({"key": "val"}) is True
    with pytest.raises(parser.FileEmptyError):
        parser.PowerGridModelData._CheckFileContainsData({})


def test_check_section_contains_data() -> None:
    assert parser.PowerGridModelData._CheckSectionContainsData([1], "node") is True
    with pytest.raises(parser.DataMissingError):
        parser.PowerGridModelData._CheckSectionContainsData([], "node")


# --- Testing Parsing Functions ---


def test_parse_nodes(valid_network_data: dict) -> None:
    nodes = parser.PowerGridModelData._ParseNodes(valid_network_data)
    assert len(nodes) == 2
    assert isinstance(nodes[1], parser.PowerNode)


def test_parse_lines(valid_network_data: dict) -> None:
    lines = parser.PowerGridModelData._ParseLines(valid_network_data)
    assert len(lines) == 1
    assert isinstance(lines[10], parser.PowerLine)


def test_parse_loads(valid_network_data: dict) -> None:
    loads = parser.PowerGridModelData._ParseLoads(valid_network_data)
    assert len(loads) == 1
    assert isinstance(loads[20], parser.PowerLoad)


def test_parse_source(valid_network_data: dict) -> None:
    sources = parser.PowerGridModelData._ParseSource(valid_network_data)
    assert len(sources) == 1
    assert isinstance(sources[30], parser.PowerSource)


# --- Testing Validation Functions ---


def test_check_single_source() -> None:
    valid_sources = {1: parser.PowerSource(1, 1, 1.0, 100.0)}
    invalid_sources = {1: parser.PowerSource(1, 1, 1.0, 100.0), 2: parser.PowerSource(2, 1, 1.0, 100.0)}
    assert parser.PowerGridModelData._CheckSingleSource(valid_sources) is True

    with pytest.raises(parser.MultipleSourcesError):
        parser.PowerGridModelData._CheckSingleSource(invalid_sources)


def test_check_unique_ids() -> None:
    nodes = {1: MagicMock()}
    lines = {2: MagicMock()}
    loads = {3: MagicMock()}
    sources = {4: MagicMock()}

    parser.PowerGridModelData._CheckUniqueIds(nodes, lines, loads, sources)

    sources_dup = {1: MagicMock()}
    with pytest.raises(parser.IdNotUniqueError):
        parser.PowerGridModelData._CheckUniqueIds(nodes, lines, loads, sources_dup)


@pytest.mark.parametrize(
    "node_ids, source_ids, load_ids, expected_error",
    [
        ([1, 2], [1], [2], None),
        ([1, 2], [1], [2, 3], "TooFewNodesError"),
        ([1, 2, 3], [1], [2], "UnusedNodesError"),
    ],
)
def test_check_node_usage(node_ids: list, source_ids: list, load_ids: list, expected_error: str) -> None:
    nodes = {i: MagicMock() for i in node_ids}
    sources = {i: MagicMock() for i in source_ids}
    loads = {i: MagicMock() for i in load_ids}

    if expected_error == "TooFewNodesError":
        with pytest.raises(parser.TooFewNodesError):
            parser.PowerGridModelData._CheckNodeUsage(nodes, sources, loads)
    elif expected_error == "UnusedNodesError":
        with pytest.raises(parser.UnusedNodesError):
            parser.PowerGridModelData._CheckNodeUsage(nodes, sources, loads)
    else:
        parser.PowerGridModelData._CheckNodeUsage(nodes, sources, loads)


def test_list_load_ids() -> None:
    loads = {100: MagicMock(), 200: MagicMock()}
    assert parser.PowerGridModelData._ListLoadIds(loads) == [100, 200]


# --- Profile Validation Tests ---


def test_get_dataframe_headers() -> None:
    df = pd.DataFrame(columns=[10, 20, 30])
    assert parser.PowerGridModelData._GetDataframeHeaders(df) == [10, 20, 30]


def test_check_header_match_load_ids() -> None:
    parser.PowerGridModelData._CheckHeaderMatchLoadIds([20, 21], [20, 21])

    with pytest.raises(parser.DataInconformationError, match="missing from the power profile data"):
        parser.PowerGridModelData._CheckHeaderMatchLoadIds([20], [20, 21])

    with pytest.raises(parser.DataInconformationError, match="do not match any load IDs"):
        parser.PowerGridModelData._CheckHeaderMatchLoadIds([20, 21, 22], [20, 21])


def test_check_dataframe_row_entries() -> None:
    df = pd.DataFrame({"val": [1, 2, 3]})
    assert parser.PowerGridModelData._CheckDataframeRowentries(df) == 3

    with pytest.raises(parser.FileEmptyError):
        parser.PowerGridModelData._CheckDataframeRowentries(pd.DataFrame())


def test_check_row_entries_match() -> None:
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [3, 4]})
    df3 = pd.DataFrame({"c": [5]})

    parser.PowerGridModelData._CheckRowEntriesMatch(df1, df2)

    with pytest.raises(parser.DataInconformationError):
        parser.PowerGridModelData._CheckRowEntriesMatch(df1, df3)


def test_get_dataframe_timestamps_list(mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_p, _ = mock_profile_dfs
    # 1. Define what we expect (matching the "%d-%b-%Y %H:%M:%S" format)
    expected = ["10-Jun-2026 10:00:00", "10-Jun-2026 11:00:00"]
    # 2. Get the actual result
    result = parser.PowerGridModelData._GetDataframeTimestampslist(df_p)
    # 3. Assert length AND content
    assert len(result) == 2
    assert result == expected


def test_check_unique_timestamps(mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_p, _ = mock_profile_dfs
    parser.PowerGridModelData._CheckUniqueTimestamps(df_p)

    df_dup = pd.concat([df_p, df_p.iloc[[0]]])
    with pytest.raises(parser.DataInconformationError):
        parser.PowerGridModelData._CheckUniqueTimestamps(df_dup)


def test_check_same_timestamps(mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_p, df_q = mock_profile_dfs
    parser.PowerGridModelData._CheckSameTimestamps(df_p, df_q)

    df_different = df_q.copy()
    df_different.index = pd.to_datetime(["2026-06-10 12:00:00", "2026-06-10 13:00:00"])
    with pytest.raises(parser.DataInconformationError):
        parser.PowerGridModelData._CheckSameTimestamps(df_p, df_different)


def test_check_timestamp_exists(mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_p, _ = mock_profile_dfs
    timestamps = parser.PowerGridModelData._GetDataframeTimestampslist(df_p)
    valid_ts = timestamps[0]

    parser.PowerGridModelData._CheckTimestampExists(df_p, valid_ts)

    with pytest.raises(parser.TimeStampNotFoundError):
        parser.PowerGridModelData._CheckTimestampExists(df_p, "01-Jan-2026 00:00:00")


# --- Testing Profile Fetching Methods ---


def test_get_power_profile_for_timestamp(mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_p, _ = mock_profile_dfs
    timestamps = parser.PowerGridModelData._GetDataframeTimestampslist(df_p)

    # Executing the internal method
    profile = parser.PowerGridModelData._GetPowerProfileForTimestamp(df_p, timestamps[0])
    assert isinstance(profile, parser.PowerProfilePerTimestamp)


def test_get_power_from_timestamp_static(mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    df_p, _ = mock_profile_dfs
    timestamps = parser.PowerGridModelData._GetDataframeTimestampslist(df_p)

    profile = parser.PowerGridModelData.GetPowerFromTimestamp(df_p, timestamps[0])
    assert isinstance(profile, parser.PowerProfilePerTimestamp)


# --- Final File Loading Test (Runs Last) ---
@pytest.mark.parametrize(
    "network_path, p_profile_path, q_profile_path",
    [(NETWORK_PATH, ACTIVE_POWER_PATH, REACTIVE_POWER_PATH)],
)
def test_load_set_of_test_files(network_path: str, p_profile_path: str, q_profile_path: str) -> None:
    """
    Attempts to load a full set of input test files using the PowerGridModelData class.
    """
    # Simple existence check
    from pathlib import Path

    if not Path(network_path).exists():
        pytest.skip(f"Integration files missing at {network_path}. Skipping test.")

    try:
        model = parser.PowerGridModelData(
            network_path=network_path, p_profile_path=p_profile_path, q_profile_path=q_profile_path
        )

        assert model.is_populated is True
        assert len(model.GetNodeIds()) > 0
        assert len(model.GetLineIds()) > 0
        assert len(model.GetLoadIds()) > 0
        assert len(model.GetSourceIds()) > 0
        assert len(model.GetTimestamps()) > 0
        assert len(model.GetVertexEdgeIdPairs()) > 0

    except Exception as e:
        pytest.fail(f"Failed to load the set of test files. Error raised: {e}")


def test_custom_file_not_found_error_attributes() -> None:
    """Exercises the custom constructor of the overridden FileNotFoundError."""
    fake_path = "fake/directory/file.json"
    fake_message = "Custom missing file alert."

    err = parser.FileNotFoundError(path=fake_path, message=fake_message)
    assert err.path == fake_path
    assert str(err) == fake_message


# --- Mocked End-to-End Initialization & Getter Tests ---


def test_power_grid_model_data_mocked_load_and_getters(
    mock_profile_dfs: tuple[pd.DataFrame, pd.DataFrame], valid_network_data: dict
) -> None:
    """
    Simulates successful file discovery, JSON reading, and Parquet parsing
    to fully test the class constructor, load_model, and all instance getters.
    """
    df_p, df_q = mock_profile_dfs

    # Pack the fixture dictionary into the structure load_model expects: load(file).get("data", {})
    mocked_json_content = json.dumps({"data": valid_network_data})

    # Intercept all real disk/file requirements
    with (
        patch("power_system_simulation.parser.PowerGridModelData._does_file_exist", return_value=True),
        patch("builtins.open", mock_open(read_data=mocked_json_content)),
        patch("pandas.read_parquet") as mock_read_parquet,
    ):
        # First call gets active power profile, second call gets reactive power profile
        mock_read_parquet.side_effect = [df_p, df_q]

        # Instantiate the main class safely using placeholder paths
        model = parser.PowerGridModelData(
            network_path="mock_net.json", p_profile_path="mock_p.parquet", q_profile_path="mock_q.parquet"
        )

        # 1. Verify successful lifecycle execution
        assert model.is_populated is True

        # 2. Fully cover and verify all instance getter methods
        assert model.GetNodeIds() == [1, 2]
        assert model.GetLineIds() == [10]
        assert model.GetLoadIds() == [20]
        assert model.GetSourceIds() == [30]

        timestamps = model.GetTimestamps()
        assert len(timestamps) == 2
        assert "10-Jun-2026 10:00:00" in timestamps

        # 3. Verify graph node-edge relationship pairings
        pairs = model.GetVertexEdgeIdPairs()
        assert len(pairs) == 2
        assert (1, 10) in pairs  # from_node to line_id
        assert (2, 10) in pairs  # to_node to line_ids
