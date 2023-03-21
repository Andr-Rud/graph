# Compgraph lib.

Библиотека представляет собой графовое вычисление над таблицами, которые храняться как словари.
Ядро библиотеки -- модуль graph.py, который собирает граф по частям. Части в стою очередь состоят из операций.
Базовые операции (map/reduce/join), и более конкретные, такие как Innerjoin, Count и тд.
Для наглядности применения библиотеки, реализовано 4 алгоритма, которые позволяют найти топ 10 слов используя
[PMI](https://www.wikiwand.com/en/Pointwise_mutual_information) индекс для каждого слова, а так же топ 3 по [частоте](https://www.wikiwand.com/ru/TF-IDF)
использования слова в тексте. Все примеры находятся в examples.

Библиотека не ограничивается лишь этими алгоритмами, 