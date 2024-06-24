from typing import Dict
from absl.testing import absltest
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from blade_bench.llms import (
    AnthropicGenConfig,
    GeminiGenConfig,
    OpenAIGenConfig,
    LLMHistory,
    OpenAITextGenerator,
    AnthropicTextGenerator,
    GeminiTextGenerator,
    LLMBase,
)

from blade_bench.logger import logger


load_dotenv()


OAI_CONFIG = OpenAIGenConfig(
    n=2,
    temperature=0.4,
    max_tokens=100,
    top_p=1.0,
    top_k=50,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    use_cache=True,
    api_key_env_name="OPENAI_AZURE_AGENTBENCH_EVAL_KEY",
    api_base="https://aagenteval.openai.azure.com/",
    api_version="2024-05-01-preview",
    deployment="gpt-4o-eval",
    model="gpt-4o",
)

ANTHROPIC_CONFIG = AnthropicGenConfig(
    temperature=0.4,
    max_tokens=100,
    use_cache=True,
    api_key_env_name="ANTHROPIC_API_KEY",
)

GEMINI_CONFIG = GeminiGenConfig(
    temperature=0.4,
    max_tokens=100,
    use_cache=True,
    api_key_env_name="GEMINI_API_KEY",
    model="gemini-1.5-pro-latest",
)

MESSAGES = [
    {
        "role": "user",
        "content": "What is the capital of France? Only respond with the exact answer",
    }
]


class MockPydanticObj(BaseModel):
    field1: str
    field2: Dict[str, int]


BAD_OBJ_STR = """ 
{
    "field1": "test",
    "field2": {"key": "54"},
}

some other text
"""


class TestGenerators(absltest.TestCase):

    def test_openai(self):
        oai_gen = OpenAITextGenerator(OAI_CONFIG)
        oai_response = oai_gen.generate(messages=MESSAGES)
        answer = oai_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(oai_response.text, 2)

    def test_anthropic(self):
        anthropic_gen = AnthropicTextGenerator(ANTHROPIC_CONFIG)
        anthropic_response = anthropic_gen.generate(messages=MESSAGES)
        answer = anthropic_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(anthropic_response.text, 1)

    def test_gemini(self):
        gemini_gen = GeminiTextGenerator(GEMINI_CONFIG)
        gemini_response = gemini_gen.generate(messages=MESSAGES)
        answer = gemini_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(gemini_response.text, 1)

    def test_llm_generate(self):
        history = LLMHistory()
        llm = LLMBase.init_from_llm_config(OAI_CONFIG, history=history)
        response = llm.generate(prompt_template=MESSAGES)
        logger.debug(f"Answer: {response}")
        self.assertIn("paris", response.lower())
        self.assertLen(history.prompt_history, 1)
        self.assertLen(history.token_usage_history, 1)

    def test_llm_pydantic_parser(self):
        llm = LLMBase.init_from_llm_config(OAI_CONFIG)
        parser = PydanticOutputParser(pydantic_object=MockPydanticObj)
        res: MockPydanticObj = llm.get_pydantic_obj_w_retires(
            parser, BAD_OBJ_STR, retries=3
        )
        self.assertEqual(res.field1, "test")
        self.assertEqual(res.field2, {"key": 54})


if __name__ == "__main__":
    absltest.main()
