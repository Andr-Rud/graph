from copy import deepcopy
import typing as tp

from . import operations as ops
from .external_sort import ExternalSort


class Graph:
    """Computational graph implementation"""

    def __init__(self) -> None:
        self.__operations: list["ops.Operation"] = []
        self.__graphs_for_join: list["Graph"] = []

    @staticmethod
    def graph_from_iter(name: str) -> "Graph":
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from "kwargs" passed to "run" method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        graph = Graph()
        graph.__operations = [deepcopy(ops.ReadIterFactory(name))]
        return graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> "Graph":
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        graph = Graph()
        graph.__operations = [deepcopy(ops.Read(filename, parser))]
        return graph

    def map(self, mapper: ops.Mapper) -> "Graph":
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        self.__operations.append(ops.Map(mapper=mapper))
        return self

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> "Graph":
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        self.__operations.append(ops.Reduce(keys=keys, reducer=reducer))
        return self

    def sort(self, keys: tp.Sequence[str], reverse: bool = False) -> "Graph":
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        :param reverse: reversed sort
        """
        self.__operations.append(ExternalSort(keys=keys, reverse=reverse))
        return self

    def join(self, joiner: ops.Joiner, join_graph: "Graph", keys: tp.Sequence[str]) -> "Graph":
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        self.__operations.append(ops.Join(joiner=joiner, keys=keys))
        self.__graphs_for_join.append(join_graph)
        return self

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        operations = iter(self.__operations)
        graph = next(operations)(**kwargs)
        i = 0
        for operation in operations:
            if type(operation) == ops.Join:
                if len(self.__graphs_for_join):
                    table = self.__graphs_for_join[i].run(**kwargs)
                    graph = operation(graph, table)
                    i += 1
            else:
                graph = operation(graph)

        return graph
