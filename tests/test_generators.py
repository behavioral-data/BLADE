from typing import Dict
from absl.testing import absltest
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from blade_bench.llms.config import get_llm_config
from blade_bench.llms import (
    LLMHistory,
    TextGenConfig,
    OpenAITextGenerator,
    LLMBase,
    llm,
)

from blade_bench.logger import logger


load_dotenv()


textgen_config = TextGenConfig(n=1, temperature=0.4, max_tokens=100, use_cache=True)


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

    def test_load_config(self):
        textgen_config = TextGenConfig(
            n=2, temperature=0.4, max_tokens=100, use_cache=True
        )
        llm_config = get_llm_config(
            provider="openai", model="gpt-3.5-turbo", textgen_config=textgen_config
        )
        oai_gen = OpenAITextGenerator(llm_config)
        oai_response = oai_gen.generate(messages=MESSAGES)
        answer = oai_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(oai_response.text, 2)

    def test_openai(self):
        textgen_config = TextGenConfig(
            n=2, temperature=0.4, max_tokens=100, use_cache=True
        )
        oai_gen = llm("openai", "gpt-3.5-turbo", textgen_config=textgen_config)
        oai_response = oai_gen.generate(messages=MESSAGES)
        answer = oai_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(oai_response.text, 2)

    def test_anthropic(self):
        anthropic_gen = llm(
            "anthropic", "claude-3.5-sonnet", textgen_config=textgen_config
        )
        anthropic_response = anthropic_gen.generate(messages=MESSAGES)
        answer = anthropic_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(anthropic_response.text, 1)

    def test_gemini(self):
        gemini_gen = llm("gemini", "gemini-1.5-pro", textgen_config=textgen_config)
        gemini_response = gemini_gen.generate(messages=MESSAGES)
        answer = gemini_response.text[0].content
        logger.debug(f"Answer: {answer}")
        self.assertIn("paris", answer.lower())
        self.assertLen(gemini_response.text, 1)

    def test_groq(self):
        groq_gen = llm("groq", "mixtral-8x7b")
        response = groq_gen.generate(messages=MESSAGES)
        answer = response.text[0].content
        self.assertIn("paris", answer.lower())

    def test_mistral(self):
        mistral_gen = llm("mistral", "mixtral-8x7b")
        response = mistral_gen.generate(messages=MESSAGES)
        answer = response.text[0].content
        self.assertIn("paris", answer.lower())

    def test_llm_generate(self):
        history = LLMHistory()
        oai_gen = llm("openai", "gpt-3.5-turbo")
        llm_gen = LLMBase(oai_gen, history=history)
        response = llm_gen.generate(prompt_template=MESSAGES)
        logger.debug(f"Answer: {response}")
        self.assertIn("paris", response.lower())
        self.assertLen(history.prompt_history, 1)
        self.assertLen(history.token_usage_history, 1)

    def test_llm_pydantic_parser(self):
        oai_gen = llm("openai", "gpt-3.5-turbo")
        llm_gen = LLMBase(oai_gen)
        parser = PydanticOutputParser(pydantic_object=MockPydanticObj)
        res: MockPydanticObj = llm_gen.get_pydantic_obj_w_retires(
            parser, BAD_OBJ_STR, retries=3
        )
        self.assertEqual(res.field1, "test")
        self.assertEqual(res.field2, {"key": 54})


if __name__ == "__main__":
    absltest.main()
