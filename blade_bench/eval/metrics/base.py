from pydantic import BaseModel


class BaseMatchMetrics(BaseModel):
    num_match1: int
    num_match2: int
    num_total1: int
    num_total2: int

    @property
    def match_rate1(self):
        return self.num_match1 / self.num_total1

    @property
    def match_rate2(self):
        return self.num_match2 / self.num_total2
