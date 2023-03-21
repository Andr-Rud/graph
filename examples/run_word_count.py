import json

import click
from compgraph.algorithms import word_count_graph


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.argument("text_column", type=str, default="text")
@click.argument("count_column", type=str, default="count")
def main(input_file: str, output_file: str, text_column: str = "text", count_column: str = "count") -> None:
    result = word_count_graph(input_file, text_column, count_column, json.loads).run()
    with open(output_file, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
