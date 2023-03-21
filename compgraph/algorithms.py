import math
from copy import deepcopy
import typing as tp
from . import Graph, operations


def word_count_graph(input_stream_name: str, text_column: str = "text", count_column: str = "count",
                     *args: tp.Any) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    if args:
        graph = Graph.graph_from_file(input_stream_name, args[0])
    else:
        graph = Graph.graph_from_iter(input_stream_name)
    return graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = "doc_id", text_column: str = "text",
                         result_column: str = "tf_idf", *args: tp.Any) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    if args:
        graph = Graph.graph_from_file(input_stream_name, args[0])
    else:
        graph = Graph.graph_from_iter(input_stream_name)

    split_word = deepcopy(graph) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    count_docs = deepcopy(graph) \
        .reduce(operations.Count("count_docs"), [])

    count_idf = deepcopy(split_word) \
        .sort([text_column, doc_column]) \
        .reduce(operations.FirstReducer(), [text_column, doc_column]) \
        .sort([text_column]) \
        .reduce(operations.Count("words_count"), [text_column]) \
        .join(operations.InnerJoiner(), count_docs, []) \
        .map(operations.Function("words_count", lambda x: 1 / x)) \
        .map(operations.Product(["words_count", "count_docs"], "idf")) \
        .map(operations.Function("idf", math.log))

    tf = deepcopy(split_word) \
        .sort([doc_column]) \
        .reduce(operations.TermFrequency(text_column, "tf"), [doc_column]) \
        .sort([text_column])

    tf_idf = tf \
        .join(operations.InnerJoiner(), count_idf, [text_column]) \
        .map(operations.Product(["idf", "tf"], result_column)) \
        .map(operations.Project([doc_column, text_column, result_column])) \
        .reduce(operations.TopN(result_column, 3), [text_column])

    return tf_idf


def pmi_graph(input_stream_name: str, doc_column: str = "doc_id", text_column: str = "text",
              result_column: str = "pmi", *args: tp.Any) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    if args:
        graph = Graph.graph_from_file(input_stream_name, args[0])
    else:
        graph = Graph.graph_from_iter(input_stream_name)

    split_word = graph \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    filtered = split_word \
        .sort([doc_column, text_column]) \
        .reduce(operations.Count("word_count"), [doc_column, text_column]) \
        .map(operations.Filter(lambda row: len(row[text_column]) > 4)) \
        .map(operations.Filter(lambda row: row["word_count"] >= 2))

    tf_in_doc = deepcopy(filtered) \
        .reduce(operations.TermFrequency(text_column, "tf_in_doc", "word_count"), [doc_column])

    tf_in_all_docs = filtered \
        .reduce(operations.TermFrequency(text_column, "tf_in_all_docs", "word_count"), [])

    pmi = tf_in_doc.join(operations.InnerJoiner(), tf_in_all_docs, [text_column]) \
        .map(operations.Function("tf_in_all_docs", lambda x: 1 / x)) \
        .map(operations.Product(["tf_in_doc", "tf_in_all_docs"], result_column)) \
        .map(operations.Function(result_column, lambda x: math.log(x))) \
        .map(operations.Project([result_column, doc_column, text_column])) \
        .sort([text_column]) \
        .sort([result_column], reverse=True) \
        .sort([doc_column]) \
        .reduce(operations.TopN(result_column, 10), [doc_column])

    return pmi


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = "enter_time", leave_time_column: str = "leave_time",
                      edge_id_column: str = "edge_id", start_coord_column: str = "start", end_coord_column: str = "end",
                      weekday_result_column: str = "weekday", hour_result_column: str = "hour",
                      speed_result_column: str = "speed", *args: tp.Any) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""

    if args:
        graph_date = Graph.graph_from_file(input_stream_name_time, args[0])
        graph_dist = Graph.graph_from_file(input_stream_name_length, args[0])
    else:
        graph_date = Graph.graph_from_iter(input_stream_name_time)
        graph_dist = Graph.graph_from_iter(input_stream_name_length)

    date = graph_date \
        .map(operations.Date(enter_time_column, weekday_result_column, hour_result_column))

    dist = graph_dist \
        .map(operations.HaversineDistance(start_coord_column, end_coord_column, "haversine"))

    average_speed = date \
        .join(operations.InnerJoiner(), dist, [edge_id_column]) \
        .sort([weekday_result_column, hour_result_column]) \
        .reduce(operations.AverageSpeed("haversine", enter_time_column, leave_time_column, speed_result_column),
                [weekday_result_column, hour_result_column]) \
        .sort([weekday_result_column, hour_result_column])

    return average_speed
