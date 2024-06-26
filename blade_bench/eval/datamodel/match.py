import pickle
from typing import Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel

from blade_bench.data.datamodel import (
    ConceptualVarSpec,
    ModelSpec,
    TransformSpec,
    TransformDatasetState,
)


class MatchCvar(BaseModel):
    var1: Union[ConceptualVarSpec, str]
    var2: Union[ConceptualVarSpec, str]
    rationale: str
    similarity: Optional[int]


class MatchedCvars(BaseModel):
    input_vars1: Union[List[ConceptualVarSpec], List[str]]
    input_vars2: Union[List[ConceptualVarSpec], List[str]]
    matched: Dict[Tuple[int, int], MatchCvar]

    def is_cvar_all_matched(self, score_threshold: int = 8):
        s1 = set(range(len(self.input_vars1)))
        s1m = set(
            [
                i - 1
                for i, j in self.matched
                if self.matched[i, j].similarity >= score_threshold
            ]
        )
        s2 = set(range(len(self.input_vars2)))
        s2m = set(
            [
                j - 1
                for i, j in self.matched
                if self.matched[i, j].similarity >= score_threshold
            ]
        )
        if s1 == s1m and s2 == s2m:
            return True
        else:
            return False


class MatchModel(BaseModel):
    model1: Union[ModelSpec, str]
    model2: Union[ModelSpec, str]
    rationale: str
    matched_cvars: Optional[Dict[str, MatchedCvars]] = None

    def is_cvar_all_matched(self, score_threshold: int = 8):
        for cvar in self.matched_cvars.values():
            if not cvar.is_cvar_all_matched(score_threshold):
                return False
        return True


class MatchedModels(BaseModel):
    input_models1: Union[List[ModelSpec], List[str]]
    input_models2: Union[List[ModelSpec], List[str]]
    mspec_id_to_cvars1: Dict[str, Union[List[ConceptualVarSpec], List[str]]]
    mspec_id_to_cvars2: Dict[str, Union[List[ConceptualVarSpec], List[str]]]
    matched: Dict[Tuple[int, int], MatchModel]
    input_models1_unique: Optional[Union[List[ModelSpec], List[str]]] = []
    matched_unique: Optional[Dict[Tuple[int, int], MatchModel]] = {}


class MatchedTSpecs(BaseModel):
    vspecs1: List[TransformSpec]
    vspecs2: List[TransformSpec]
    gspecs1: List[TransformSpec]
    gspecs2: List[TransformSpec]
    cat_specs1: List[TransformSpec]
    cat_specs2: List[TransformSpec]


class MatchTransforms(BaseModel):
    transform_state1: Union[TransformDatasetState, None]
    transform_state2: Union[TransformDatasetState, None]
    matched_tspecs: MatchedTSpecs


class MatchedAnnotations(BaseModel):
    matched_models: MatchedModels
    matched_transforms: MatchTransforms
    matched_cvars: Dict[Literal["Control", "IV", "DV", "Moderator"], MatchedCvars]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def tspecs1(self):
        return len(self.matched_transforms.transform_state1.expanded_id_to_spec) - 1

    @property
    def mspecs1(self):
        return len(self.matched_models.input_models1)

    @property
    def mspecs_unique(self):
        return len(self.matched_models.input_models1_unique)

    @property
    def cvars1(self):
        ret = []
        for cvar in self.matched_cvars.values():
            ret.extend(cvar.input_vars1)
        return ret

    @property
    def cvars2(self):
        ret = []
        for cvar in self.matched_cvars.values():
            ret.extend(cvar.input_vars2)
        return ret
