import base64
import json
import logging
import os
from typing import Any, List, Union
from diskcache import Cache
import hashlib
import platform

import tiktoken
from .datamodel import Message

from blade_bench.logger import logger


def backoff_hdlr(details):
    logger.warning(
        str(type(details["exception"]))
        + "\n"
        + str(details["exception"])
        + "\n\n"
        + "Backing off {wait:0.1f} seconds after {tries} tries calling function {target}".format(
            **details
        )
    )


def num_tokens_from_messages(
    messages: Union[List[Message], dict], model="gpt-3.5-turbo-0301"
):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if (
        model == "gpt-3.5-turbo-0301" or True
    ):  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            if not isinstance(message, dict):
                message = message.model_dump()
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )

            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


def get_hash(params: Any):
    return hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()


def cache_request(cache: Cache, params: Any, values: Any = None) -> Any:
    # Generate a unique key for the request

    key = get_hash(params)
    # Check if the request is cached
    if key in cache and values is None:
        return cache[key]

    # Cache the provided values and return them
    if values:
        cache[key] = values
    # else:
    # print("Didn't cache")
    return values


def get_user_cache_dir(app_name: str) -> str:
    system = platform.system()
    if system == "Windows":
        cache_path = os.path.join(os.getenv("LOCALAPPDATA"), app_name, "Cache")
    elif system == "Darwin":
        cache_path = os.path.join(os.path.expanduser("~/Library/Caches"), app_name)
    else:  # Linux and other UNIX-like systems
        cache_path = os.path.join(
            os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), app_name
        )
    os.makedirs(cache_path, exist_ok=True)
    return cache_path
