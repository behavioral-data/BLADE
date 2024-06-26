from abc import ABC, abstractmethod
import os.path as osp
import time
from typing import List, Union
from diskcache import Cache
from .datamodel import ProviderModelConfig, TextGenConfig, TextGenResponse
from .utils import cache_request, get_user_cache_dir
from blade_bench.logger import API_LEVEL_NAME, logger


class TextGenerator(ABC):
    def __init__(
        self,
        config: ProviderModelConfig,
        cache_dir: str = None,
        **kwargs,
    ):
        self.config = config
        cache_dir_default = get_user_cache_dir("blade_bench")
        cache_dir_based_on_model = osp.join(
            cache_dir_default, self.config.provider, self.config.model
        )
        self.cache_dir = kwargs.get("cache_dir", cache_dir_based_on_model)

        self.cache_dir = (
            osp.join(osp.dirname(osp.abspath(__file__)), "cache")
            if cache_dir is None
            else cache_dir
        )
        self.cache = Cache(self.cache_dir, size_limit=2**30)

    def cache_request(
        self,
        params: dict,
    ):
        start_time = time.time()
        response = cache_request(cache=self.cache, params=params)
        if response:
            response = TextGenResponse(**response)
            elapsed_time = time.time() - start_time
            response.cache_elapsed_time = elapsed_time
            response.from_cache = True
            return response

    @abstractmethod
    def generate_core(
        self, messages: Union[List[dict], str], **kwargs
    ) -> TextGenResponse:
        pass

    def generate(self, messages: Union[List[dict], str], **kwargs):
        params = self.config.textgen_config.model_dump() | {"messages": messages}
        use_cache = True if self.config.use_cache is None else self.config.use_cache
        if use_cache:
            response: TextGenResponse = self.cache_request(params)
            if response is not None:
                return response

        start_time = time.time()
        logger.info(
            f"Sending {self.config.provider} API Request ({self.config.model})",
            config=self.config.model_dump() | {"messages": messages},
        )
        response: TextGenResponse = self.generate_core(messages, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(
            f"{self.config.provider} response took {elapsed_time: 0.2f} seconds",
        )
        response.api_elapsed_time = elapsed_time
        logger.bind(response=response.model_dump(), cached=False).log(
            API_LEVEL_NAME,
            f"Called {self.config.provider} API ({self.config.model})",
        )

        cache_request(self.cache, params=params, values=response.model_dump())
        response.api_elapsed_time = elapsed_time
        return response
