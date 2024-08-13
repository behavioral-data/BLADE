import datetime
import re

import numpy as np
import pytz


SAVE_CODE_TEMPLATE = """from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('{data_path}')

# ======== TRANSFORM CODE ========
{transform_code}

# ======== MODEL CODE ========
{model_code}

"""

SAVE_CONVERTED_CODE_TEMPLATE = """
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pickle

class TransformDataReturn(BaseModel):
    df: pd.DataFrame
    column_mapping: Dict[FrozenSet[str], str]
    transform_verb: Literal[
        "derive", "filter", "groupby", "deduplicate", "impute", "rollup", "orderby"
    ]
    groupby_cols: Optional[Set[str]] = set()  # only for groupby verb
    code: str = ""


df = pd.read_csv('{data_path}')

# ======== ORIGINAL CODE ========
{transform_code}

# ======== CONVERTED CODE ========
{converted_code}
"""


def escape_newlines_in_quotes(code_str):
    # Regular expression pattern to match \n inside quotes
    pattern = r'(["\'])(.*?)\1'

    def replacer(match):
        # Get the matched string
        matched_str = match.group(0)
        # Replace \n with \\n inside the matched string
        return matched_str.replace("\n", "\\n")

    # Replace \n inside quotes with \\n
    escaped_str = re.sub(pattern, replacer, code_str, flags=re.DOTALL)

    return escaped_str


def normalize_code_string(code_str):
    # Replace common escaped sequences
    code_str = code_str.split("```python")[-1].split("```")[0].strip()

    code_str = code_str.replace("\\n", "\n")
    code_str = code_str.replace("\\t", "\t")
    code_str = code_str.replace("\\'", "'")
    code_str = code_str.replace('\\"', '"')
    code_str = escape_newlines_in_quotes(code_str)
    return code_str


def get_curr_pst_time_str() -> str:
    current_time = datetime.datetime.now()
    # Get Pacific timezone
    pst_timezone = pytz.timezone("US/Pacific")
    current_time_with_pst = current_time.astimezone(pst_timezone)
    formatted_time = current_time_with_pst.strftime("%m-%d-%Y at %I:%M:%S %p")
    return formatted_time


def bootstrap_fn(
    series, fn, n_samples=1000, cis=[0.025, 0.975], to_str=False, precision=3
):
    results = []
    for i in range(n_samples):
        x = series.sample(frac=1.0, replace=True)
        results.append(fn(x))

    mean = np.mean(results)
    cis = np.quantile(results, cis)
    if to_str:
        mean = "{0:.{1}f}".format(mean, precision)
        ci_0 = "{0:.{1}f}".format(cis[0], precision)
        ci_1 = "{0:.{1}f}".format(cis[1], precision)
        return f"{mean} ({ci_0}, {ci_1})"

    return mean, cis


def bootstrapped_mean(x, **kwargs):
    return bootstrap_fn(x, np.mean, **kwargs)
