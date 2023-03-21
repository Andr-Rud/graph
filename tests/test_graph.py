import json

from compgraph.graph import Graph
from compgraph import operations as ops


def test_graph_from_iter() -> None:
    simple = [
        {"word": "a", "num": 1, "flag": True}
    ]
    graph = Graph.graph_from_iter("simple")
    assert list(graph.run(simple=lambda: iter(simple))) == simple


def test_graph_from_file() -> None:
    with open("temp_in", "w") as f:
        json.dump({"word": "a", "num": 1, "flag": True}, f)

    simple = [
        {"word": "a", "num": 1, "flag": True}
    ]
    graph = Graph.graph_from_file("temp_in", json.loads)
    result = list(graph.run())
    assert result == simple


def test_graph_map() -> None:
    mapping = [
        {"word": "a", "num": 1, "flag": True},
        {"word": "b", "num": 2, "flag": False}
    ]

    expected = [
        {"word": "a", "num": 2, "flag": True},
        {"word": "b", "num": 4, "flag": False}
    ]
    graph = Graph.graph_from_iter("mapping").map(ops.Function("num", lambda x: x * 2))
    assert list(graph.run(mapping=lambda: iter(mapping))) == expected


def test_graph_reduce() -> None:
    mapping = [
        {"word": "a", "num": 1, "flag": True},
        {"word": "a", "num": 3, "flag": True},
        {"word": "b", "num": 2, "flag": False}
    ]

    expected = [
        {"word": "a", "count": 2},
        {"word": "b", "count": 1}
    ]
    graph = Graph.graph_from_iter("mapping").reduce(ops.Count("count"), ["word"])
    result = sorted(list(graph.run(mapping=lambda: iter(mapping))), key=lambda x: x["count"], reverse=True)

    assert result == expected


def test_graph_join() -> None:
    tab_a = [
        {"word": "a", "num": 1, "flag": True},
        {"word": "a", "num": 3, "flag": True},
        {"word": "b", "num": 2, "flag": False}
    ]
    tab_b = [
        {"word": "a", "param": 0.11},
        {"word": "a", "param": 2.},
        {"word": "b", "param": 3.}
    ]

    expected = [
        {"flag": True, "num": 1, "param": 0.11, "word": "a"},
        {"flag": True, "num": 1, "param": 2.0, "word": "a"},
        {"flag": True, "num": 3, "param": 0.11, "word": "a"},
        {"flag": True, "num": 3, "param": 2.0, "word": "a"},
        {"flag": False, "num": 2, "param": 3.0, "word": "b"}
    ]

    graph_a = Graph.graph_from_iter("tab_a")
    graph_b = Graph.graph_from_iter("tab_b")
    graph_join = graph_a.join(ops.InnerJoiner(), graph_b, ["word"])

    assert list(graph_join.run(tab_a=lambda: iter(tab_a), tab_b=lambda: iter(tab_b))) == expected
