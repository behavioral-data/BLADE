import requests
from typing import List, Dict, Optional

from blade_bench.llms.datamodel.local import LocalResponse


class LocalLLMClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_model_name(self) -> str:
        response = requests.get(f"{self.base_url}/model_name")
        if response.status_code == 200:
            return response.json().get("model_name")
        else:
            response.raise_for_status()

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 2048,
        temperature: Optional[float] = 0.5,
        stop_sequences: Optional[List[str]] = None,
    ) -> LocalResponse:
        data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop_sequences:
            data["stop_sequences"] = stop_sequences
        response = requests.post(f"{self.base_url}/generate_chat", json=data)
        if response.status_code == 200:
            return LocalResponse(**response.json())
        else:
            response.raise_for_status()


if __name__ == "__main__":
    # Initialize the wrapper with the base URL of the FastAPI server
    api_wrapper = LocalLLMClient("http://localhost:8000")

    # Get the model name
    model_name = api_wrapper.get_model_name()
    print(f"Model Name: {model_name}")

    # Generate a chat response
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": "Please write me some code to sort a list of integers in Python using the bubble sort algorithm.",
        },
    ]

    response = api_wrapper.generate_chat(
        messages, max_tokens=700, temperature=0.7, stop_sequences=["please"]
    )
    print(f"Chat Response: {response}")
