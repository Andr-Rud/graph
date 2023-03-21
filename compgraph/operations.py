import math
import heapq
import string
import calendar
from itertools import groupby
from datetime import datetime
from math import acos, sin, cos
from abc import abstractmethod, ABC

import re
import typing as tp

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    """
    Operations (mappers) that can be applied to column
    """

    def __init__(self, mapper: Mapper) -> None:
        """
        :param: mapper: choose mapper to apply
        """
        self.mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        for row in rows:
            yield from self.mapper(row)  # or just yield ?...


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        :param group_key: saved keys
        """
        pass


class Reduce(Operation):
    """
    Operations (reducers) that can change rows
    """

    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        """
        :param reducer: operation that change columns by keys
        :param keys: keys for make reduce operation
        """
        self.reducer = reducer
        self.keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        for _, v in groupby(rows, key=lambda x: [x[i] for i in self.keys]):
            yield from self.reducer(tuple(self.keys), v)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = "_1", suffix_b: str = "_2") -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod  # tp.Iterator[tp.List[None]] doesn't work instead of Any :(
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable | tp.Any,
                 rows_b: TRowsIterable | tp.Any) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    @staticmethod
    def _next_iter(it: tp.Any) -> tp.Any:
        try:
            return next(it)
        except StopIteration:
            return None

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        """
        :param rows: table1 rows
        :param args: table2 rows
        """
        rows1 = groupby(rows, key=lambda x: [x[k] for k in self.keys])
        rows2 = groupby(args[0], key=lambda x: [x[k] for k in self.keys])

        key1, value1 = self._next_iter(rows1)
        key2, value2 = self._next_iter(rows2)

        end = False
        flag = False

        while not end:
            if key1 == key2:
                yield from self.joiner(tuple(self.keys), value1, value2)

                if (next_it_1 := self._next_iter(rows1)) is None:
                    end = True
                else:
                    key1, value1 = next_it_1

                if (next_it_2 := self._next_iter(rows2)) is None:
                    end, flag = True, True
                else:
                    key2, value2 = next_it_2

            elif key1 < key2:
                yield from self.joiner(tuple(self.keys), value1, iter([None]))
                if (next_it_1 := self._next_iter(rows1)) is None:
                    end = True
                else:
                    key1, value1 = next_it_1
            elif key1 > key2:
                yield from self.joiner(tuple(self.keys), iter([None]), value2)
                if (next_it_2 := self._next_iter(rows2)) is None:
                    end = True
                else:
                    key2, value2 = next_it_2
            else:
                raise ValueError("Keys error")

        if flag:
            yield from self.joiner(tuple(self.keys), value1, iter([None]))
        else:
            yield from self.joiner(tuple(self.keys), iter([None]), value2)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = row[self.column].translate(str.maketrans("", "", string.punctuation))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self._lower_case(row[self.column])
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator

    @staticmethod
    def split_iter(line: str) -> tp.Generator[str, None, None]:
        return (x.group(0) for x in re.finditer(r"[A-Za-z']+", line))

    def __call__(self, row: TRow) -> TRowsGenerator:
        for line in self.split_iter(row[self.column]):
            row_new = row.copy()
            row_new[self.column] = line
            yield row_new


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(self, columns: tp.Sequence[str], result_column: str = "product") -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        result = 1
        for column in self.columns:
            result *= row[column]
        row[self.result_column] = result
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        new_row = {}
        for column in self.columns:
            new_row[column] = row[column]
        yield new_row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        h: list[tp.Any] = []
        result: list[tp.Any] = []
        for row in rows:
            if len(h) <= self.n or (len(h) and h[0] <= row[self.column_max]):
                heapq.heappush(h, row[self.column_max])
                result.append(row)
            while len(h) > self.n:
                for i in range(len(result)):
                    if h[0] == result[i][self.column_max]:
                        result = result[:i] + result[i + 1:]
                        break
                heapq.heappop(h)
        for row in result:
            yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = "tf", count_column: str | None = None) -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column
        self.count_column = count_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        r_rows = {}
        n = 0
        for row in rows:
            if row[self.words_column] not in r_rows:
                new_row = {}
                for t in group_key:
                    new_row[t] = row[t]
                new_row[self.words_column] = row[self.words_column]
                new_row[self.result_column] = 0
                r_rows[row[self.words_column]] = new_row

            if self.count_column:
                r_rows[row[self.words_column]][self.result_column] += row[self.count_column]
                n += row[self.count_column]
            else:
                r_rows[row[self.words_column]][self.result_column] += 1
                n += 1

        for w, row in r_rows.items():
            row[self.result_column] /= n
            yield row


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        new_row = {self.column: 1}
        row = next(iter(rows))
        for _ in rows:
            new_row[self.column] += 1

        for t in group_key:
            new_row[t] = row[t]

        yield new_row


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        r_rows = {}
        for row in rows:
            tpl: tp.Any = []
            for t in group_key:
                tpl.append(row[t])
            tpl = tuple(tpl)

            if tpl not in r_rows:
                new_row = {self.column: row[self.column]}
                for t in group_key:
                    new_row[t] = row[t]
                r_rows[tpl] = new_row
            else:
                r_rows[tpl][self.column] += row[self.column]

        for k, row in r_rows.items():
            yield row


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        copy_b = list(rows_b)
        for row_a in rows_a:
            for row_b in copy_b:
                if row_a is not None and row_b is not None:
                    for k in keys:
                        if k not in row_a:
                            break
                        if k not in row_b:
                            break

                    new_row = {}
                    for k in row_a:
                        if k in row_b and k not in keys:
                            new_row[k + self._a_suffix] = row_a[k]
                        else:
                            new_row[k] = row_a[k]
                    for k in row_b:
                        if k in row_a and k not in keys:
                            new_row[k + self._b_suffix] = row_b[k]
                        else:
                            new_row[k] = row_b[k]
                    yield new_row


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        new_row = {}

        for row_a in rows_a:
            if row_a is not None:
                for k in keys:
                    if k not in row_a:
                        break
                for k in row_a:
                    new_row[k] = row_a[k]

        for row_b in rows_b:
            if row_b is not None:
                for k in keys:
                    if k not in row_b:
                        break
                for k in row_b:
                    new_row[k] = row_b[k]

        yield new_row


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        for row_b in rows_b:
            for row_a in rows_a:
                if row_a is not None:
                    for k in keys:
                        if k not in row_a:
                            raise ValueError("That key not in rows_a")

                    new_row = {}
                    for k in row_a:
                        new_row[k] = row_a[k]
                    if row_b is not None:
                        for k in keys:
                            if k not in row_b:
                                raise ValueError("That key not in rows_b")
                        for k in row_b:
                            new_row[k] = row_b[k]
                    yield new_row


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        for row_b in rows_b:
            for row_a in rows_a:
                if row_b is not None:
                    for k in keys:
                        if k not in row_b:
                            raise ValueError("That key not in rows_b")

                    new_row = {}
                    for k in row_b:
                        new_row[k] = row_b[k]
                    if row_a is not None:
                        for k in keys:
                            if k not in row_a:
                                raise ValueError("That key not in rows_a")
                        for k in row_a:
                            new_row[k] = row_a[k]
                    yield new_row


class Function(Mapper):
    def __init__(self, column: str, function: tp.Callable[[tp.Any], tp.Any]) -> None:
        """
        :param column: name of the column to apply the function
        :param function: function that you can apply to column
        """
        self.column = column
        self.function = function

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self.function(row[self.column])
        yield row


class HaversineDistance(Mapper):
    """
    Class for calculation haversine distance on th Earth
    """

    def __init__(self, start_column: str, end_column: str, result_column: str):
        self.start_column = start_column
        self.end_column = end_column
        self.result_column = result_column
        self.EARTH_RADIUS_KM = 6373.

    def calculate(self, row: TRow) -> float:
        lon_1, lat_1, lon_2, lat_2 = map(math.radians, [*row[self.start_column], *row[self.end_column]])
        return self.EARTH_RADIUS_KM * acos(sin(lat_1) * sin(lat_2) + cos(lat_1) * cos(lat_2) * cos(lon_2 - lon_1))

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.result_column] = self.calculate(row)
        yield row


class Date(Mapper):
    """
    Transform date to human-readable format
    """

    def __init__(self, enter_time_column: str, weekday_result_column: str, hour_result_column: str) -> None:
        self.enter_time_column = enter_time_column
        self.weekday_result_column = weekday_result_column
        self.hour_result_column = hour_result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        try:
            date = datetime.strptime(row[self.enter_time_column], "%Y%m%dT%H%M%S.%f")
        except ValueError:
            date = datetime.strptime(row[self.enter_time_column], "%Y%m%dT%H%M%S")
        row[self.weekday_result_column] = calendar.day_name[date.weekday()][:3]
        row[self.hour_result_column] = date.hour
        yield row


class AverageSpeed(Reducer):
    """
    Calculate average speed by the distances and times
    """

    def __init__(self, distance_column: str, enter_time_column: str, leave_time_column: str,
                 speed_result_column: str) -> None:
        self.distance_column = distance_column
        self.enter_time_column = enter_time_column
        self.leave_time_column = leave_time_column
        self.speed_result_column = speed_result_column

    def __call__(self, group_key: tp.Tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        time_: float = 0
        dist_: float = 0
        new_row: TRow = {}

        for row in rows:
            if not len(new_row):
                for t in group_key:
                    new_row[t] = row[t]

            dist_ += row[self.distance_column]
            try:
                time_ += (datetime.strptime(row[self.leave_time_column], "%Y%m%dT%H%M%S.%f") -
                          datetime.strptime(row[self.enter_time_column], "%Y%m%dT%H%M%S.%f")).total_seconds() / 3600
            except ValueError:
                time_ += (datetime.strptime(row[self.leave_time_column], "%Y%m%dT%H%M%S") -
                          datetime.strptime(row[self.enter_time_column], "%Y%m%dT%H%M%S")).total_seconds() / 3600

        new_row[self.speed_result_column] = dist_ / time_
        yield new_row
