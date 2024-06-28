import pytest

from blade_bench.eval.convert import Convert
from blade_bench.eval.exceptions import (
    LMSubmissionExecutionError,
    LMSubmissionEmptyError,
)
from blade_bench.llms import OpenAITextGenerator, OpenAIGenConfig
from blade_bench.llms.datamodel.gen_config import TextGenConfig


@pytest.fixture
def textgen():
    textgen_config = TextGenConfig(n=1, temperature=0, max_tokens=1000, use_cache=True)
    config = OpenAIGenConfig(
        provider="azureopenai",
        api_key_env_name="OPENAI_AZURE_AGENTBENCH_EVAL_KEY",
        api_base="https://aagenteval.openai.azure.com/",
        api_version="2024-05-01-preview",
        deployment="gpt-4o-eval",
        model="gpt-4o",
        textgen_config=textgen_config,
    )
    return OpenAITextGenerator(config=config)


@pytest.fixture
def converter(textgen):
    return Convert(
        run_dataset="hurricane",
        text_gen=textgen,
        use_code_cache=False,
    )


@pytest.mark.asyncio
async def test_raise_code_error_when_empty(converter):
    code = ""
    with pytest.raises(LMSubmissionEmptyError):
        await converter.get_state_data_from_code(code)


@pytest.mark.asyncio
async def test_raise_code_error_when_syntax(converter):
    code = "print('hello)"
    with pytest.raises(LMSubmissionExecutionError):
        await converter.get_state_data_from_code(code)


@pytest.mark.asyncio
async def test_raise_code_error_when_execution(converter):
    code = "a = 3/0"
    with pytest.raises(LMSubmissionExecutionError):
        await converter.get_state_data_from_code(code)


if __name__ == "__main__":
    pytest.main([__file__])
