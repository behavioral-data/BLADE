import json
import re
from typing import Any, Callable, Dict, List, Union
from langchain.schema import BaseOutputParser
from langchain.schema.output_parser import OutputParserException


from .config import get_llm_config, get_text_gen

from .datamodel import (
    AnthropicGenConfig,
    LLMHistory,
    OpenAIGenConfig,
    GeminiGenConfig,
    PromptHistoryEntry,
    TextGenResponse,
)
from .base import TextGenerator


from blade_bench.logger import LLM_LEVEL_NAME, PROMPT_LEVEL_NAME, logger

FORMAT_SYSTEM_PROMPT = """You are an AI Data Analysis assistant who is an expert at \
parsing JSON from text given the JSON schema."""

FORMAT_INSTRUCTION_PROMPT = """<Instruction> Given a pydantic JSON schema, and the pydantic parsing error, \
please fix the JSON such that it conforms to the schema. 
**IMPORTANT** if the original input does not contain the information to even be placed in the JSON, \
return an empty JSON (i.e. `{{}}`).
</Instruction>

<JSON Schema>
{format_instructions}
</JSON Schema>

Original JSON: ```
{orig_str}
```
Parsing error: "{error}"
New JSON: """


class LLMBase:
    """
    Base class for all classes with calling LLM functionality
    """

    @classmethod
    def init_from_base_llm_config(cls):
        llm_config = get_llm_config("azureopenai", "gpt-4o-azure")
        return cls.init_from_llm_config(llm_config)

    @classmethod
    def init_from_llm_config(
        cls,
        llm_config: Union[OpenAIGenConfig, GeminiGenConfig, AnthropicGenConfig],
        cache_dir: str = None,
        **kwargs,
    ):
        """
        Initialize the class from the LLM config
        """
        text_gen = get_text_gen(llm_config, cache_dir=cache_dir)
        return cls(text_gen, **kwargs)

    def __init__(self, text_gen: TextGenerator, history: LLMHistory = None):
        self.text_gen = text_gen
        self.history = history

    def generate_chat_prompt(
        self,
        prompt_template: Union[str, List[Dict[str, str]]],
        prompt_variables: Dict[str, str] = None,
    ) -> List[Dict[str, str]]:
        if type(prompt_template) == str:
            if prompt_variables:
                prompt = prompt_template.format(**prompt_variables)
            else:
                prompt = prompt_template
            return [{"role": "user", "content": prompt}]
        else:
            chat_prompt = []
            for entry in prompt_template:
                role = entry["role"]
                if prompt_variables:
                    try:
                        content = entry["content"].format(**prompt_variables)
                    except KeyError as e:
                        formatted_str = (
                            entry["content"].replace("{", "{{").replace("}", "}}")
                        )
                        content = formatted_str.format(**prompt_variables)
                else:
                    content = entry["content"]
                chat_prompt.append({"role": role, "content": content})
            return chat_prompt

    def generate(
        self,
        prompt_template: Union[str, List[Dict[str, str]]],
        prompt_variables: Dict[str, str] = None,
        stop_sequences: List[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, str] = None,
    ) -> str:
        self.text_gen.config.textgen_config.stop_sequences = stop_sequences
        chat_prompt = self.generate_chat_prompt(prompt_template, prompt_variables)

        logger.bind(config=self.text_gen.config.textgen_config.model_dump()).log(
            PROMPT_LEVEL_NAME,
            f"Sending prompt from {self.__class__}",
            messages=chat_prompt,
        )
        response = self.text_gen.generate(chat_prompt)

        logger.bind(
            config=self.text_gen.config.textgen_config.model_dump(),
            message=response.text[0].content,
            from_cache=response.from_cache,
            usage=response.usage.model_dump(),
            api_elapsed_time=(
                "{:.3f}s".format(response.api_elapsed_time)
                if response.api_elapsed_time
                else ""
            ),
            cache_elapsed_time=(
                "{:.3f}ms".format(response.cache_elapsed_time * 1000)
                if response.cache_elapsed_time
                else ""
            ),
        ).log(LLM_LEVEL_NAME, f"Received response from {self.__class__}")

        metadata = metadata or {}
        tags = tags or []

        if self.history is not None:
            self.history.token_usage_history.append(response.usage)
            prompt_entry = PromptHistoryEntry(
                messages=chat_prompt,
                response=response,
                tags=tags,
            )
            self.history.prompt_history.append(prompt_entry)

        if self.text_gen.config.log_file:
            self.log_to_file(response, chat_prompt)

        return response.text[0].content

    def log_to_file(self, response: TextGenResponse, chat_prompt: List[Dict[str, str]]):
        with open(self.text_gen.config.log_file, "a") as f:
            f.write(
                f"\n===================[PROMPT (API Time: {response.api_elapsed_time}, Cache Time: {response.cache_elapsed_time})]=====================\n"
            )
            for propmt in chat_prompt:
                f.write(f"=====[{propmt['role'].upper()}]=====:\n{propmt['content']}\n")
            usage = response.usage.model_dump() if response.usage else {}
            f.write(
                f"\n===================[{self.text_gen.config.model} RESPONSE ({usage.get('completion_tokens')})]=====================\n"
            )
            f.write(response.text[0].content)
            f.write("\n===================[TOKENS]=====================\n")
            f.write(f"Number of prompt tokens: {usage.get('prompt_tokens')}\n")
            f.write(f"Number of total tokens: {usage.get('total_tokens')}\n")

    def generate_with_pydantic_parser(
        self,
        parser: BaseOutputParser,
        prompt_template: Union[str, List[Dict[str, str]]],
        prompt_variables: Dict[str, str] = None,
        transform: Callable[[str], str] = None,
        tags: List[str] = None,
        metadata: Dict[str, str] = None,
        retries: int = 3,
    ):
        response = self.generate(
            prompt_template=prompt_template,
            prompt_variables=prompt_variables,
            tags=tags,
            metadata=metadata,
        )

        if transform:
            response = transform(response)

        return self.get_pydantic_obj_w_retires(parser, response, retries=retries)

    def match_json_catch_error(self, text: str):
        try:
            return self.match_json(text)
        except json.JSONDecodeError:
            return {}

    def match_json(self, text: str):
        match = re.search(
            r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        json_str = ""
        if match:
            json_str = match.group()
        json_object = json.loads(json_str, strict=False)
        return json_object

    def match_jsons(self, text: str, parser: BaseOutputParser):
        matches = re.findall(
            r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        ret = None
        for match in matches:
            json_object = None
            try:
                json_object = json.loads(match, strict=False)
                obj = parser.parse(match)
                return obj
            except (OutputParserException, json.JSONDecodeError):
                if isinstance(json_object, dict) and len(json_object) == 0:
                    ret = {}
                continue
        return ret

    def get_pydantic_obj_w_retires(
        self,
        parser: BaseOutputParser,
        response: str,
        retries: int = 3,
    ) -> Any:
        try:
            res_json = self.match_json(response)
            if len(res_json) == 0:
                return {}
            return parser.parse(response)
        except (OutputParserException, json.JSONDecodeError) as e:
            ret = self.match_jsons(response, parser)
            if ret is not None:
                return ret
            if retries == 0:
                raise ValueError(f"Failed to parse the response after retries")

            prompt_template = [
                {
                    "role": "system",
                    "content": FORMAT_SYSTEM_PROMPT,
                },
                {"role": "user", "content": FORMAT_INSTRUCTION_PROMPT},
            ]

            prompt_variables = {
                "format_instructions": parser.get_format_instructions(),
                "orig_str": response,
                "error": str(e),
            }

            response = self.generate(
                prompt_template=prompt_template,
                prompt_variables=prompt_variables,
                tags=["pydantic_parse_error"],
            )
            return self.get_pydantic_obj_w_retires(
                parser,
                response,
                retries=retries - 1,
            )

    def _parse_reflection_and_code(self, raw_response: str, parse_str: str = "Result:"):
        try:
            reflection, code = raw_response.split(parse_str)
            reflection = reflection.strip()
            code = code.split("```python")[-1].split("```")[0].strip()
        except ValueError:
            reflection = ""
            code = ""
        return reflection, code

    def _parse_code(self, raw_response: str):
        code = raw_response.split("```python")[-1].split("```")[0].strip()
        return code
