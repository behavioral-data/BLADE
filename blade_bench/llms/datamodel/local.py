from pydantic import BaseModel
from blade_bench.llms.datamodel.usage import UsageData


class LocalResponse(BaseModel):
    response: str
    usage: UsageData
