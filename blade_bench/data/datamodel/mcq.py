from typing import Dict, List, Literal, Optional, Set
import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from blade_bench.data.annotation import AnnotationDBData
from blade_bench.data.datamodel.graph import SerialGraphCodeRunInfo
from blade_bench.data.datamodel.specs import (
    Branch,
    ConceptualVarSpec,
    TransformSpec,
    ModelSpec,
)
from blade_bench.data.process.transforms.graph_paths import GraphPaths


CHOICES = ["A", "B", "C", "D"]


class ColCvarTransform(BaseModel):
    col: str
    cvar: ConceptualVarSpec
    tspec: Optional[TransformSpec] = None

    @property
    def cvar_concept(self):
        return self.cvar.concept


class ModelAssociatedCol(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_grp_ind: int
    model_spec: ModelSpec
    dv: List[ColCvarTransform]
    ivs: List[ColCvarTransform]
    controls: List[ColCvarTransform]

    def hash_str(self):
        dv_cols = list(sorted(dv.col for dv in self.dv))
        iv_cols = list(sorted(iv.col for iv in self.ivs))
        control_cols = list(sorted(control.col for control in self.controls))
        return f"{self.model_grp_ind}-dv-{dv_cols}_iv-{iv_cols}_control-{control_cols}"

    def model_associated_tspecs(self) -> List[TransformSpec]:
        return [
            col.tspec
            for col in self.dv + self.ivs + self.controls
            if col.tspec is not None
        ]


class MCQResponse(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(
        ..., title="The answer to the multiple choice question"
    )
    rationale: str = Field(..., title="The rationale for the answer")

    @property
    def answer_index(self):
        return CHOICES.index(self.answer)


class MCQTransformChoice(BaseModel):
    code: str
    rationale: str
    is_llm_generated: bool

    def __hash__(self) -> int:
        return hash(self.code + self.rationale)


class MCQCvarChoice(BaseModel):
    cvar_str: str
    rationale: str
    is_llm_generated: bool

    def __hash__(self) -> int:
        return hash(self.cvar_str + self.rationale)


class MCQSimpleTransform(BaseModel):
    coneptual_var_str: str
    mc_type: Literal["select_pos", "select_neg"]
    options: List[MCQTransformChoice]
    correct_answer: MCQTransformChoice

    @property
    def task_instruction(self):
        s = "MOST" if self.mc_type == "select_pos" else "LEAST"
        instr = f"""Given the research question and dataset, we want to perform an analysis to answer the question. 
Specifically we want to operationalize the conceptual variable *{self.coneptual_var_str.lower()}* which we will use for statistical modeling. 
Of the choices given, select transformation code that is {s} justifiable to operationalize *{self.coneptual_var_str.lower()}*."
"""
        return instr

    @property
    def valid_values(self):
        return CHOICES[: len(self.options)]

    @property
    def choices(self):
        s = ""
        for i, opt in enumerate(self.options):
            s += f"{CHOICES[i]}.\n```python\n{opt.code}\n```\n"
        return s

    def __str__(self):
        s = f"MCQ Transform: {self.mc_type} for operationalizing {self.coneptual_var_str}\n"
        for i, option in enumerate(self.options):
            s += f"Option {i+1}: {option.code}\n"
            s += f"Rationale: {option.rationale}\n"
            s += "----\n"
        s += f"Correct Answer: {self.options.index(self.correct_answer) + 1}\n"
        return s


class MCQSimpleCvar(BaseModel):
    mc_type: Literal["select_pos", "select_neg"]
    options: List[MCQCvarChoice]
    correct_answer: MCQCvarChoice

    def __str__(self):
        s = f"MCQ: {self.mc_type}\n"
        for i, option in enumerate(self.options):
            s += f"Option {i+1}: {option.cvar_str}\n"
            s += f"Rationale: {option.rationale}\n"
            s += "----\n"
        s += f"Correct Answer: {self.correct_answer.cvar_str}\n"
        return s

    @property
    def task_instruction(self):
        s = "MOST" if self.mc_type == "select_pos" else "LEAST"
        instr = f"""Given the research question and dataset, we want to perform an analysis to answer the question.
Of the choices below, select the conceptual variable that is {s} justifiable for the analysis.
"""
        return instr

    @property
    def valid_values(self):
        return CHOICES[: len(self.options)]

    @property
    def choices(self):

        s = ""
        for i, opt in enumerate(self.options):
            s += f"{CHOICES[i]}. {opt.cvar_str}\n"
        return s


class MCQDatasetSimple(BaseModel):
    mcqs_cvar: List[MCQSimpleCvar]
    mcqs_transform: Dict[str, List[MCQSimpleTransform]]

    @property
    def num_cvars(self):
        return len(self.mcqs_cvar)

    @property
    def num_transforms(self):
        return sum([len(v) for v in self.mcqs_transform.values()])

    @property
    def num_mcqs(self):
        return self.num_cvars + self.num_transforms
    
    @property
    def expected_correct(self):
        count = 0
        for mcq in self.mcqs_cvar:
            count += 1 / len(mcq.options)
        for k, mcqs in self.mcqs_transform.items():
            for mcq in mcqs:
                count += 1 / len(mcq.options)
        return count

class MultipleChoiceAnswer(BaseModel):
    answer: str  # Literal?
    rationale: str


class TransformCodeChoice(BaseModel):
    tspec: TransformSpec
    orig_code: str
    anon_code: str
    col_mapping: Dict[str, str]
    orig_col_name: str
    anon_col_name: str


class MultipleChoiceTransformCode(BaseModel):
    cvar: ConceptualVarSpec
    pos_choice: TransformCodeChoice
    neg_choices: List[TransformCodeChoice]


class TransformColsChoice(BaseModel):
    associated_og_cols: Set[str]


class MultipleChoiceTransformCols(BaseModel):
    cvar: ConceptualVarSpec
    pos_choice: TransformColsChoice
    neg_choices: List[TransformColsChoice]


class MultipleChoiceStatsModel(BaseModel):
    pos_choice: ModelAssociatedCol
    neg_choices: List[ModelAssociatedCol]

    def model_summary(self):
        return {
            "pos_choice": self.pos_choice.hash_str(),
            "neg_choices": [neg_choice.hash_str() for neg_choice in self.neg_choices],
        }

    def question_code(self, annotatorData: AnnotationDBData):
        t_specs = []
        for neg_choice in self.neg_choices:
            t_specs.extend(neg_choice.model_associated_tspecs())
        for tspec in self.pos_choice.model_associated_tspecs():
            t_specs.append(tspec)

        tspec_ids = set()
        for tspec in t_specs:
            if tspec.spec_id in tspec_ids:
                continue
            tspec_ids.add(tspec.spec_id)

        def get_subgraph(nx_g, nodes):
            ancestors = set()
            for n in nodes:
                ancestors.update(nx.ancestors(nx_g, n))
            return nx_g.subgraph(ancestors.union(nodes))

        subg = get_subgraph(annotatorData.nx_g, tspec_ids)
        leaf_nodes = [n for n in subg.nodes if subg.out_degree(n) == 0]
        branches = [Branch(dependencies=[n]) for n in leaf_nodes]
        gp = GraphPaths(annotatorData.transform_specs)
        s: List[SerialGraphCodeRunInfo] = gp.get_graphs_from_branches(branches)
        code, output_cols = s[0].build_code_str(annotatorData.transform_specs)
        return code
