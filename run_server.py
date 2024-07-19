import os

import click
from blade_bench.llms.local.server import app, MAPPING
import uvicorn


@click.command()
@click.option(
    "--port",
    default=8000,
    help="Port to run the server on",
)
@click.option(
    "--model",
    default="l7b",
    type=click.Choice(list(MAPPING.keys())),
    help="Model to use",
)
def run_server(port: int, model: str):
    os.environ["HF_MODEL"] = model
    uvicorn.run(app, host="localhost", port=port)


if __name__ == "__main__":
    run_server()
