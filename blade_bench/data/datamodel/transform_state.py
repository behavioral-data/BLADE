from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd


class SingleColState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    col: str
    spec_id: str
    expanded_spec_id: str = ""
    value_hash: str = ""
    categorical_value_hash: str = ""
    graph_hash: str = ""
    df_col: Optional[pd.Series] = Field(default=None, exclude=True)
    cols_graph_id: int = None
    cols_nid: str = ""  # nid in the cols graph
    code: Optional[str] = None


class TransformState(BaseModel):
    """
    All permutations of the path that lead to an equivalent state in terms of the df vale
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    spec_id: str
    expanded_spec_id: str = ""
    df_value_hash: Dict[str, str] = {}
    df_categorical_value_hash: Dict[str, str] = {}
    df: Optional[Union[pd.DataFrame, pd.Series, None]] = Field(
        default=None, exclude=True
    )
    columns: List[str] = []
    spec_name: Optional[str] = None

    def get_transform_state_from_cols(
        self,
        expanded_spec_id: str,
        cols: List[str],
    ):
        """
        Create a new transform state from the current state but only keep the cols
        """
        new_state = TransformState(
            spec_id=self.spec_id,
            expanded_spec_id=expanded_spec_id,
            df_value_hash={col: self.df_value_hash[col] for col in cols},
            df_categorical_value_hash={
                col: self.df_categorical_value_hash[col] for col in cols
            },
            df=self.df[cols] if isinstance(self.df, pd.DataFrame) else self.df,
            columns=cols,
            spec_name=self.spec_name,
        )
        return new_state

    def get_single_state_from_col(
        self, col: str, col_nid: str, code: str = None
    ) -> SingleColState:
        return SingleColState(
            col=col,
            spec_id=self.spec_id,
            expanded_spec_id=self.expanded_spec_id,
            value_hash=self.df_value_hash[col],
            categorical_value_hash=self.df_categorical_value_hash[col],
            df_col=self.df[col],
            cols_nid=col_nid,
            code=code,
        )

    def get_single_col_states(self) -> List[SingleColState]:
        """
        Create a new single col state from the current state but only keep the col
        """
        return [
            SingleColState(
                col=col,
                spec_id=self.spec_id,
                expanded_spec_id=self.expanded_spec_id,
                value_hash=self.df_value_hash[col],
                categorical_value_hash=self.df_categorical_value_hash[col],
                df_col=self.df[col],
            )
            for col in self.columns
        ]


if __name__ == "__main__":
    state_instance = SingleColState(
        col="example_column",
        spec_id="spec_001",
        expanded_spec_id="exp_spec_001",
        value_hash="hash_value",
        categorical_value_hash="cat_hash_value",
        graph_hash="graph_hash_value",
        df_col=pd.Series([1, 2, 3]),
        cols_graph_id=123,
        cols_nid="nid_001",
        code="example_code",
    )
    print(state_instance)
