from copy import deepcopy
import pytest
import requests_mock
from requests_mock.exceptions import NoMockAddress
from blade_bench.llms.datamodel.gen_config import HuggingFaceGenConfig, TextGenConfig
from blade_bench.llms.local.local_client import LocalLLMClient
from blade_bench.llms.textgen_huggingface import HuggingFaceTextGenerator


HF_CONFIG = HuggingFaceGenConfig(
    model="some_hf_model",  # This is not used in the test
    api_base="http://localhost:8000",
    llm_client="default",
    textgen_config=TextGenConfig(
        temperature=0.7,
    ),
    use_cache=False,
)


@pytest.fixture
def client():
    return LocalLLMClient(base_url="http://localhost:8000")


def generate_chat_mock(url, request_mock):
    request_mock.post(
        f"{url}/generate_chat",
        json={
            "response": "Mocked response",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
        status_code=200,
    )


def get_model_name_mock(url, request_mock):
    request_mock.get(
        f"{url}/model_name",
        json={"model_name": "test_model"},
        status_code=200,
    )


@pytest.fixture
def mock_generate_chat(client):
    with requests_mock.Mocker() as m:
        generate_chat_mock(client.base_url, m)
        yield m


@pytest.fixture
def mock_get_model_name(client):
    with requests_mock.Mocker() as m:
        get_model_name_mock(client.base_url, m)
        yield m


def test_client_get_model_name(client, mock_get_model_name):
    assert client.get_model_name() == "test_model"


def test_client_generate_chat(client, mock_generate_chat):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Test message"},
    ]
    response = client.generate_chat(
        messages, max_tokens=700, temperature=0.7, stop_sequences=["please"]
    )
    assert response.response == "Mocked response"
    assert response.usage.prompt_tokens == 1
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 3


def test_hf_generate(client, requests_mock):
    generate_chat_mock(client.base_url, requests_mock)
    get_model_name_mock(client.base_url, requests_mock)
    with pytest.raises(AssertionError):
        llm = HuggingFaceTextGenerator(config=HF_CONFIG, cache_dir=None)
    conf = deepcopy(HF_CONFIG)
    conf.model = "test_model"
    llm = HuggingFaceTextGenerator(config=conf, cache_dir=None)

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Test message"},
    ]

    response = llm.generate_core(messages)

    assert response.text[0].content == "Mocked response"
    assert response.usage.prompt_tokens == 1
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 3


def test_hf_generate_bad_url(client, requests_mock):
    generate_chat_mock(client.base_url, requests_mock)
    get_model_name_mock(client.base_url, requests_mock)

    conf = deepcopy(HF_CONFIG)
    conf.api_base = "http://localhost:8001"
    # check that an error is raised
    with pytest.raises(NoMockAddress):
        llm = HuggingFaceTextGenerator(config=conf, cache_dir=None)


if __name__ == "__main__":
    pytest.main([__file__])
