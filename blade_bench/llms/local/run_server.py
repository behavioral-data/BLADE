import os
from blade_bench.llms.local.server import app, MAPPING
import uvicorn
import requests
from blade_bench.logger import logger


def run_server(port: int = 8000):
    model = os.environ.get("HF_MODEL")
    if model is None:
        logger.info(
            f"No model specified in the environment variable 'HF_MODEL', defaulting to 'l7b'. The options are {list(MAPPING.keys())}"
        )
        os.environ["HF_MODEL"] = "l7b"

    uvicorn.run(app, host="localhost", port=port)


def get_model_name(port: int = 8000):
    model_name = requests.get(f"http://localhost:{port}/model_name")
    return model_name.json().get("model_name")


if __name__ == "__main__":
    run_server()
