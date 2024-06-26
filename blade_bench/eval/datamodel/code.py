import re
from typing import List
from pydantic import BaseModel


class CodeAndReflection(BaseModel):
    code: str
    reflection: str

    @property
    def transform_funcs(self) -> List[str]:
        function_regex = re.compile(r"def (\w+)\(")
        transform_funcs = function_regex.findall(self.code)
        return transform_funcs
