import json

import click
from compgraph.algorithms import pmi_graph


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.argument("document_column", type=str, default="doc_id")
@click.argument("text_column", type=str, default="text")
@click.argument("pmi_column", type=str, default="pmi")
def main(input_file: str, output_file: str, document_column: str = "doc_id",
         text_column: str = "text", pmi_column: str = "pmi") -> None:
    result = pmi_graph(input_file, document_column, text_column, pmi_column, json.loads).run()
    with open(output_file, "w") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
