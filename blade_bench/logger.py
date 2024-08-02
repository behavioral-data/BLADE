from collections import OrderedDict
from loguru import logger
import json  # <!- add this line
import sys


LLM_LEVEL_NAME = "LLM"
PROMPT_LEVEL_NAME = "PROMPT"
API_LEVEL_NAME = "API"
CODE_ENV_QUERY = "CODE_ENV_QUERY"
CODE_ENV_RESP = "CODE_ENV_RESP"
TS_STATE_QUERY = "TS_STATE_QUERY"
TS_STATE_RESP = "TS_STATE_RESP"


class Formatter:
    def __init__(self):
        self.padding = 0
        self.fmt = "[<green><b>{time:YYYY-MM-DD hh:mm:ss.SS}</b></green>][<cyan><b>{file}:{line}</b></cyan> - <cyan>{name:}:{function}</cyan>][<level>{level}</level>] {message}\n"

    def format(self, record):
        length = len("{file}:{line} - {name:}:{function}".format(**record))
        self.padding = max(self.padding, length)
        record["extra"]["padding"] = " " * (self.padding - length)
        fmt = ""
        if record["level"].name == LLM_LEVEL_NAME and "message" in record["extra"]:
            if record["extra"]["from_cache"]:
                fmt = "<LG>===================[[<b>Response (cache time={extra[cache_elapsed_time]}  completion tokens={extra[usage][completion_tokens]}  total_tokens={extra[usage][total_tokens]})</b>]]===================</LG>\n{extra[message]}\n"
            else:
                fmt = "<LY>===================[[<b>Response (API time={extra[api_elapsed_time]}  completion tokens={extra[usage][completion_tokens]}  total_tokens={extra[usage][total_tokens]})</b>]]===================</LY>\n{extra[message]}\n"
        elif record["level"].name == CODE_ENV_QUERY:
            fmt = "<LC>===================[[<b>CODE ENV QUERY</b>]]===================</LC>\n{extra[message]}\n"
        elif record["level"].name == CODE_ENV_RESP:
            if record["extra"]["from_cache"]:
                fmt = "<LG>===================[[<b>CODE RESPONSE (cache time={extra[cache_elapsed_time]})</b>]]===================</LG>\n{extra[message]}\n"
            else:
                fmt = "<LY>===================[[<b>CODE RESPONSE (run time={extra[api_elapsed_time]})</b>]]===================</LY>\n{extra[message]}\n"
        elif (
            record["level"].name == PROMPT_LEVEL_NAME and "messages" in record["extra"]
        ):
            for i, message in enumerate(record["extra"]["messages"]):
                fmt += (
                    f"<LC>===================[[<b>{message['role']:}</b>]]===================</LC>"
                    + f"\n{{extra[messages][{i}][content]}}\n"
                )
        elif record["level"].name == TS_STATE_QUERY:
            fmt = "<LC>===================[[<b>TS STATE QUERY</b>]]===================</LC>\n{extra[message]}\n"

        elif record["level"].name == TS_STATE_RESP:
            if record["extra"]["from_cache"]:
                fmt = "<LG>===================[[<b>TS STATE RESPONSE (cache time={extra[cache_elapsed_time]})</b>]]===================</LG>\n{extra[message]}\n"
            else:
                fmt = "<LY>===================[[<b>TS STATE RESPONSE (run time={extra[api_elapsed_time]})</b>]]===================</LY>\n{extra[message]}\n"

        ret_fmt = self.fmt

        ret_fmt = ret_fmt.replace("{serialized_short}", "")
        return ret_fmt + fmt


def serialize(record):
    subset = OrderedDict()
    subset["level"] = record["level"].name
    subset["message"] = record["message"]
    subset["time"] = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    subset["file"] = {
        "name": record["file"].name,
        "path": record["file"].path,
        "function": record["function"],
        "line": record["line"],
    }
    subset["extra"] = record["extra"]
    return json.dumps(subset)


def serialize_extras(record):
    return json.dumps(record["extra"])


def patching(record):
    extras = serialize_extras(record)
    record["serialized_short"] = extras[:50] + "..." if len(extras) > 50 else extras
    record["extra"]["serialized"] = serialize(record)

    def add_tabs_to_string(s: str):
        return "\n".join(["\t" + l for l in s.split("\n")])

    if record["level"].name == LLM_LEVEL_NAME and "message" in record["extra"]:
        record["extra"]["message"] = record["extra"]["message"]
    elif record["level"].name == PROMPT_LEVEL_NAME and "messages" in record["extra"]:
        for i, message in enumerate(record["extra"]["messages"]):
            record["extra"]["messages"][i]["content"] = message["content"]


logger.remove(0)
logger = logger.patch(patching)
logger.level(PROMPT_LEVEL_NAME, no=10, color="<yellow><bold>", icon="📋")
logger.level(LLM_LEVEL_NAME, no=10, color="<lm><bold>", icon="🤖")
logger.level(API_LEVEL_NAME, no=10, color="<red><bold>", icon="🛜")
logger.level(CODE_ENV_QUERY, no=10, color="<cyan><bold>", icon="🔍")
logger.level(CODE_ENV_RESP, no=10, color="<yellow><bold>", icon="🔍")
logger.level(TS_STATE_QUERY, no=10, color="<cyan><bold>", icon="🔍")
logger.level(TS_STATE_RESP, no=10, color="<yellow><bold>", icon="🔍")


formatter = Formatter()
logger.add(sys.stdout, format=formatter.format)
