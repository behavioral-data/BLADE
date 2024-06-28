from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from blade_bench.llms.local.hf_interence import LocalLLM
from dotenv import load_dotenv
import os
from blade_bench.logger import API_LEVEL_NAME, logger

load_dotenv()

MAPPING = {
    "l7b": "meta-llama/CodeLlama-7b-Instruct-hf",
    "l13b": "meta-llama/CodeLlama-13b-Instruct-hf",
    "d7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
}


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.5
    stop_sequences: Optional[List[str]] = None


models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name = MAPPING[os.environ["HF_MODEL"]]
    logger.info(f"Loading Local LLM model: {model_name}")
    models["llm"] = LocalLLM(model_name)
    yield
    models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/generate_chat")
async def generate_chat(request: ChatRequest):
    model: LocalLLM = models["llm"]
    try:
        logger.log(API_LEVEL_NAME, f"Received request:")
        response = model.generate_chat(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop_sequences=request.stop_sequences,
        )
        return response
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_name")
async def get_model_name():
    model = models["llm"]
    try:
        model_name = model.get_model_name()
        return {"model_name": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
