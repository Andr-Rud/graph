import copy
import dataclasses
import typing as tp

import pytest
from pytest import approx

from compgraph import operations as ops


class _Key:
    def __init__(self, *args: str) -> None:
        self._items = args

    def __call__(self, d: tp.Mapping[str, tp.Any]) -> tuple[str, ...]:
        return tuple(str(d.get(key)) for key in self._items)


@dataclasses.dataclass
class MapCase:
    mapper: ops.Mapper
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    mapper_item: int = 0
    mapper_ground_truth_items: tuple[int, ...] = (0,)


MAP_CASES = [
    MapCase(
        mapper=ops.Function(column="func", function=lambda x: 1 / x),
        data=[
            {"test_id": 1, "func": 1},
            {"test_id": 2, "func": 2},
            {"test_id": 3, "func": 3}
        ],
        ground_truth=[
            {"test_id": 1, "func": 1},
            {"test_id": 2, "func": 1 / 2},
            {"test_id": 3, "func": 1 / 3}
        ],
        cmp_keys=("test_id", "text")
    ),
    MapCase(
        mapper=ops.Function(column="func", function=lambda x: x ** 2),
        data=[
            {"test_id": 1, "func": 1.1},
            {"test_id": 2, "func": 2.2},
            {"test_id": 3, "func": 3.3}
        ],
        ground_truth=[
            {"test_id": 1, "func": approx(1.21, 0.001)},
            {"test_id": 2, "func": approx(4.84, 0.001)},
            {"test_id": 3, "func": approx(10.89, 0.001)}
        ],
        cmp_keys=("test_id", "text")
    ),
    MapCase(
        mapper=ops.Function(column="func", function=lambda x: x.lower()),
        data=[
            {"test_id": 1, "func": "ASD"},
            {"test_id": 2, "func": "Qwe"},
            {"test_id": 3, "func": "vvv"}
        ],
        ground_truth=[
            {"test_id": 1, "func": "asd"},
            {"test_id": 2, "func": "qwe"},
            {"test_id": 3, "func": "vvv"}
        ],
        cmp_keys=("test_id", "text")
    ),
    MapCase(
        mapper=ops.Date("enter_time", "weekday", "hour"),
        data=[
            {"leave_time": "20171020T112238.723000", "enter_time": "20171020T112237.427000",
             "edge_id": 8414926848168493057},
            {"leave_time": "20171011T145553.040000", "enter_time": "20171011T145551.957000",
             "edge_id": 8414926848168493057},
            {"leave_time": "20171020T090548.939000", "enter_time": "20171020T090547.463000",
             "edge_id": 8414926848168493057},
            {"leave_time": "20171024T144101.879000", "enter_time": "20171024T144059.102000",
             "edge_id": 8414926848168493057},
            {"leave_time": "20171022T131828.330000", "enter_time": "20171022T131820.842000",
             "edge_id": 5342768494149337085},
            {"leave_time": "20171014T134826.836000", "enter_time": "20171014T134825.215000",
             "edge_id": 5342768494149337085},
            {"leave_time": "20171010T060609.897000", "enter_time": "20171010T060608.344000",
             "edge_id": 5342768494149337085},
            {"leave_time": "20171027T082600.201000", "enter_time": "20171027T082557.571000",
             "edge_id": 5342768494149337085}
        ],
        ground_truth=[
            {"leave_time": "20171020T112238.723000", "enter_time": "20171020T112237.427000",
             "edge_id": 8414926848168493057,
             "weekday": "Fri", "hour": 11},
            {"leave_time": "20171011T145553.040000", "enter_time": "20171011T145551.957000",
             "edge_id": 8414926848168493057,
             "weekday": "Wed", "hour": 14},
            {"leave_time": "20171020T090548.939000", "enter_time": "20171020T090547.463000",
             "edge_id": 8414926848168493057,
             "weekday": "Fri", "hour": 9},
            {"leave_time": "20171024T144101.879000", "enter_time": "20171024T144059.102000",
             "edge_id": 8414926848168493057,
             "weekday": "Tue", "hour": 14},
            {"leave_time": "20171022T131828.330000", "enter_time": "20171022T131820.842000",
             "edge_id": 5342768494149337085,
             "weekday": "Sun", "hour": 13},
            {"leave_time": "20171014T134826.836000", "enter_time": "20171014T134825.215000",
             "edge_id": 5342768494149337085,
             "weekday": "Sat", "hour": 13},
            {"leave_time": "20171010T060609.897000", "enter_time": "20171010T060608.344000",
             "edge_id": 5342768494149337085,
             "weekday": "Tue", "hour": 6},
            {"leave_time": "20171027T082600.201000", "enter_time": "20171027T082557.571000",
             "edge_id": 5342768494149337085,
             "weekday": "Fri", "hour": 8}
        ],
        cmp_keys=("leave_time", "enter_time", "edge_id", "weekday", "hour")
    ),
    MapCase(
        mapper=ops.HaversineDistance("start", "end", "haversine"),
        data=[
            {"start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
             "edge_id": 8414926848168493057},
            {"start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
             "edge_id": 5342768494149337085},
            {"start": [37.56963176652789, 55.846845586784184], "end": [37.57018438540399, 55.8469259692356],
             "edge_id": 5123042926973124604},
            {"start": [37.41463478654623, 55.654487907886505], "end": [37.41442892700434, 55.654839486815035],
             "edge_id": 5726148664276615162},
            {"start": [37.584684155881405, 55.78285809606314], "end": [37.58415022864938, 55.78177368734032],
             "edge_id": 451916977441439743},
            {"start": [37.736429711803794, 55.62696328852326], "end": [37.736344216391444, 55.626937723718584],
             "edge_id": 7639557040160407543},
            {"start": [37.83196756616235, 55.76662947423756], "end": [37.83191015012562, 55.766647034324706],
             "edge_id": 1293255682152955894},
        ],
        ground_truth=[{"start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
                       "edge_id": 8414926848168493057, "haversine": 0.03202394407224201},
                      {"start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
                       "edge_id": 5342768494149337085, "haversine": 0.045464188432109455},
                      {"start": [37.56963176652789, 55.846845586784184], "end": [37.57018438540399, 55.8469259692356],
                       "edge_id": 5123042926973124604, "haversine": 0.035647728095922},
                      {"start": [37.41463478654623, 55.654487907886505], "end": [37.41442892700434, 55.654839486815035],
                       "edge_id": 5726148664276615162, "haversine": 0.041184536692075085},
                      {"start": [37.584684155881405, 55.78285809606314], "end": [37.58415022864938, 55.78177368734032],
                       "edge_id": 451916977441439743, "haversine": 0.1251565805619792},
                      {"start": [37.736429711803794, 55.62696328852326],
                       "end": [37.736344216391444, 55.626937723718584], "edge_id": 7639557040160407543,
                       "haversine": 0.0060755402662239395},
                      {"start": [37.83196756616235, 55.76662947423756], "end": [37.83191015012562, 55.766647034324706],
                       "edge_id": 1293255682152955894, "haversine": 0.004089016811623846}],
        cmp_keys=("start", "end", "edge_id", "haversine")
    ),
]


@pytest.mark.parametrize("case", MAP_CASES)
def test_custom_mapper(case: MapCase) -> None:
    mapper_data_row = copy.deepcopy(case.data[case.mapper_item])
    mapper_ground_truth_rows = [copy.deepcopy(case.ground_truth[i]) for i in case.mapper_ground_truth_items]

    key_func = _Key(*case.cmp_keys)

    mapper_result = case.mapper(mapper_data_row)
    assert isinstance(mapper_result, tp.Iterator)
    assert sorted(mapper_ground_truth_rows, key=key_func) == sorted(mapper_result, key=key_func)

    result = ops.Map(case.mapper)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.ground_truth, key=key_func) == sorted(result, key=key_func)


@dataclasses.dataclass
class ReduceCase:
    reducer: ops.Reducer
    reducer_keys: tuple[str, ...]
    data: list[ops.TRow]
    ground_truth: list[ops.TRow]
    cmp_keys: tuple[str, ...]
    reduce_data_items: tuple[int, ...] = (0,)
    reduce_ground_truth_items: tuple[int, ...] = (0,)


REDUCE_CASES = [
    ReduceCase(
        reducer=ops.AverageSpeed("haversine", "enter_time", "leave_time", "speed"),
        reducer_keys=("weekday", "hour"),
        data=[{"leave_time": "20171027T082600.201000", "enter_time": "20171027T082557.571000",
               "edge_id": 5342768494149337085, "weekday": "Fri", "hour": 8,
               "start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
               "haversine": 0.045464188432109455},
              {"leave_time": "20171020T090548.939000", "enter_time": "20171020T090547.463000",
               "edge_id": 8414926848168493057, "weekday": "Fri", "hour": 9,
               "start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
               "haversine": 0.03202394407224201},
              {"leave_time": "20171020T112238.723000", "enter_time": "20171020T112237.427000",
               "edge_id": 8414926848168493057, "weekday": "Fri", "hour": 11,
               "start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
               "haversine": 0.03202394407224201},
              {"leave_time": "20171014T134826.836000", "enter_time": "20171014T134825.215000",
               "edge_id": 5342768494149337085, "weekday": "Sat", "hour": 13,
               "start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
               "haversine": 0.045464188432109455},
              {"leave_time": "20171022T131828.330000", "enter_time": "20171022T131820.842000",
               "edge_id": 5342768494149337085, "weekday": "Sun", "hour": 13,
               "start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
               "haversine": 0.045464188432109455},
              {"leave_time": "20171010T060609.897000", "enter_time": "20171010T060608.344000",
               "edge_id": 5342768494149337085, "weekday": "Tue", "hour": 6,
               "start": [37.524768467992544, 55.88785375468433], "end": [37.52415172755718, 55.88807155843824],
               "haversine": 0.045464188432109455},
              {"leave_time": "20171024T144101.879000", "enter_time": "20171024T144059.102000",
               "edge_id": 8414926848168493057, "weekday": "Tue", "hour": 14,
               "start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
               "haversine": 0.03202394407224201},
              {"leave_time": "20171011T145553.040000", "enter_time": "20171011T145551.957000",
               "edge_id": 8414926848168493057, "weekday": "Wed", "hour": 14,
               "start": [37.84870228730142, 55.73853974696249], "end": [37.8490418381989, 55.73832445777953],
               "haversine": 0.03202394407224201}],
        ground_truth=[
            {"weekday": "Fri", "hour": 8, "speed": approx(62.2322, 0.001)},
            {"weekday": "Fri", "hour": 9, "speed": approx(78.1070, 0.001)},
            {"weekday": "Fri", "hour": 11, "speed": approx(88.9552, 0.001)},
            {"weekday": "Sat", "hour": 13, "speed": approx(100.9690, 0.001)},
            {"weekday": "Sun", "hour": 13, "speed": approx(21.8577, 0.001)},
            {"weekday": "Tue", "hour": 6, "speed": approx(105.3901, 0.001)},
            {"weekday": "Tue", "hour": 14, "speed": approx(41.5145, 0.001)},
            {"weekday": "Wed", "hour": 14, "speed": approx(106.4505, 0.001)}
        ],
        cmp_keys=("weekday", "hour", "speed")
    ),
]


@pytest.mark.parametrize("case", REDUCE_CASES)
def test_reducer(case: ReduceCase) -> None:
    key_func = _Key(*case.cmp_keys)
    result = ops.Reduce(case.reducer, case.reducer_keys)(iter(case.data))
    assert isinstance(result, tp.Iterator)
    assert sorted(case.ground_truth, key=key_func) == sorted(result, key=key_func)
