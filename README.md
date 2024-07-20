# BLADE

Dataset and code for ["BLADE: Benchmarking Language Model Agents for Data-Driven Science"]()

---
## Introduction
BLADE is a comprehensive benchmark designed to evaluate Language Model Agents on writing justifiable analyses on real-world scientific research questions from data (e.g., *Are soccer players with a dark skin tone more likely than those with a light skin tone to receive red cards from referees?* from [Silberzahn et al.](https://journals.sagepub.com/doi/10.1177/2515245917747646)). In particular, BLADE evaluates Agents' ability to iteratively integrate scientific domain knowledge, statistical expertise, and data understanding to make nuanced analytical decisions

BLADE consists of 12 dataset and research question pairs with high-quality ground truth analysis decisions (i.e., choice of conceptual construct, transformations, statistical model) made by expert data scientists and researchers who independently conducted the analyses. In addition, BLADE contains Y multiple choice questions for discerning justifiable analysis decisions. 


![Main](assets/main_white.png)
<p align="left">
  <em><b>Overvie of BLADE Construction and Evaluation.</b> We gathered research questions and datasets from existing research papers, crowd-sourced analysis studies and statistic textbooks as well as analyses from expert annotators (boxes 1-3). Given a research question and dataset, LM agents generate a full analysis containing the relevant conceptual variables, a data transform function, and a statistical modeling function (boxes 1,4, and 5). BLADE then performs automatic evaluation against the ground truth (box 6).</em>
</p>

## Getting Started
To get started with BLADE, follow the steps below:
### 1. Installation
TODO

### 2. LM Setup
BLADE evaluates Language Models and uses them for evaluation. Set the api keys for different LM services 
```bash
# for openai
export OPENAI_API_KEY=<your key>

# for google gemini
export GEMINI_API_KEY=<your key>

# for anthropic
export ANTHROPIC_API_KEY=<your key>
```
Some default model configurations (e.g., environment variable for the api key) are specified in [llm_config.yml](blade_bench/conf/config.default.yml). You can also add set your own configurations by creating your own yaml file using the yaml format and setting the environment variable `LLM_CONFIG_PATH` to the file.

### 3. Running LMs and Agent
We provide a starter script to run a basic LM or ReACT agent for our benchmark.

```bash
Usage: run_gen_analyses.py [OPTIONS]

Options:
  --run_dataset [fish|boxes|conversation|reading|crofoot|panda_nuts|fertility|hurricane|teachingratings|mortgage|soccer|affairs|amtl|caschools]
                                  Dataset to run  [required]
  -n, --num_runs INTEGER          Number of runs to perform  [default: 10]
  --use_agent                     Whether to use agent or just the base LM
  --no_cache_code_reuslts         [ONLY used when use_agent=True] Whether to
                                  cache code results when running code.
  --no_use_data_desc              Whether to use data description in the
                                  prompts for the LM  [default: True]
  --llm_config_path FILE          Path to the LLM config file, used to specify
                                  the provider, model, and text generation
                                  config such as the temperature.  [default:
                                  ./conf/llm.yaml]
  --llm_provider [openai|azureopenai|gemini|anthropic|huggingface]
                                  Provider for the LLM to override the config
                                  file at llm_config_path
  --llm_model TEXT                Model for the LLM to override the config
                                  file at llm_config_path. Default options are
                                  ['gpt-4', 'gpt-4-32k', 'gpt-3.5-turbo',
                                  'gpt-3.5-turbo-0301', 'gpt-3.5-turbo-16k',
                                  'gpt-4o-azure', 'gpt-4o-eval',
                                  'gpt-3.5-turbo', 'gemini-1.5-pro-latest',
                                  'claude-3-opus', 'claude-3.5-sonnet',
                                  'codellama-7b-instruct', 'deepseek-
                                  coder-6.7b-instruct']
  --llm_eval_config_path FILE     Path to the LLM eval config file, used to
                                  specify the provider, model, and text
                                  generation config such as the temperature.
                                  [default: ./conf/llm_eval.yaml]
  --output_dir DIRECTORY          output directory to store saved analyses
  --help                          Show this message and exit.
```
This will write the results to the folder specified by `output_dir`. After running the script, in the output folder, there will be a `multirun_analyses.json` file (which can be loaded into the [MultiRunResults](blade_bench/eval/datamodel/multirun.py) pydantic object) which can be used for evaluation.

An example is provided in `example/multirun_analyses.json`.

### 4. Evaluating Agent Generated Analyses
We provide a starter script to evaluate the outputs of running`run_gen_analyses.py`. Run `run_get_eval.py` as follows:

```bash
Usage: run_get_eval.py [OPTIONS]

Options:
  --multirun_load_path FILE      [EITHER multirun_load_path or
                                 submission_load_path is REQUIRED] Path to
                                 load the multirun analyses.
  --submission_load_path FILE    [EITHER multirun_load_path or
                                 submission_load_path is REQUIRED]
  --llm_eval_config_path FILE    Path to the LLM eval config file
  --no_cache_code_results        Whether to not cache code results when
                                 running code for the evaluation
  --output_dir DIRECTORY         output directory to store saved eval results
  --ks TEXT                      List of k values for diversity metrics.
                                 Default is [1, 5, 10]
  --diversity_n_samples INTEGER  Number of samples to use for diversity
                                 metrics
  --help                         Show this message and exit.
```
Here is an example:
```bash
python run_get_eval.py --multirun_load_path ./examples/multirun_analyses.json
```

## Data Exploration Functions
To access the dataset and research question we can
```python
from blade_bench.data import load_dataset_info, list_datasets, DatasetInfo

all_datasets = list_datasets()
dinfo: DatasetInfo = load_dataset_info("soccer", load_df=True)
rq = dinfo.research_question
df = dinfo.df
```

To explore the ground truth annotations, we can
```python
from blade_bench.data import load_ground_truth, AnnotationDBData

gnd_truth: AnnotationDBData = load_ground_truth('soccer')
# see AnnotationDBData object for details 
print(len(gnd_truth.transform_specs))
print(len(gnd_truth.cv_specs))
```
More details about the structure of the ground truth is available in the paper.

## Evaluating a Submission on BLADE
To evalute your own submission to BLADE, the LM agent must generate a `json` file that conforms to the folloing schema. An example shown in [example/submission_analyses.json](example/submission_analyses.json). Then just specify --submission_load_path when running `run_get-eval.py`.

```json
{
  "$defs": {
    "AgentCVarsWithCol": {
      "properties": {
        "ivs": {
          "items": {
            "$ref": "#/$defs/IVarWithCol"
          },
          "title": "Independent variables",
          "type": "array"
        },
        "dv": {
          "allOf": [
            {
              "$ref": "#/$defs/DVarWithCol"
            }
          ],
          "title": "Dependent variable"
        },
        "controls": {
          "items": {
            "$ref": "#/$defs/ControlVarWithCol"
          },
          "title": "Control variables",
          "type": "array"
        }
      },
      "required": [
        "ivs",
        "dv",
        "controls"
      ],
      "title": "AgentCVarsWithCol",
      "type": "object"
    },
    "ControlVarWithCol": {
      "properties": {
        "description": {
          "title": "Description of the control variable variable",
          "type": "string"
        },
        "is_moderator": {
          "title": "Whether the variable is a moderator.",
          "type": "boolean"
        },
        "moderator_on": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": "",
          "title": "The variable that the control variable is moderating. Only applicable for control variables that are moderators."
        },
        "columns": {
          "items": {
            "type": "string"
          },
          "title": "The column(s) in the FINAL dataframe used in the STATISTICAL MODEL that corresponds to the control variable",
          "type": "array"
        }
      },
      "required": [
        "description",
        "is_moderator",
        "columns"
      ],
      "title": "ControlVarWithCol",
      "type": "object"
    },
    "DVarWithCol": {
      "properties": {
        "description": {
          "title": "Description of the dependent variable variable",
          "type": "string"
        },
        "columns": {
          "items": {
            "type": "string"
          },
          "title": "The column(s) in the FINAL dataframe used in the STATISTICAL MODEL that corresponds to the dependent variable",
          "type": "array"
        }
      },
      "required": [
        "description",
        "columns"
      ],
      "title": "DVarWithCol",
      "type": "object"
    },
    "EntireAnalysis": {
      "properties": {
        "cvars": {
          "allOf": [
            {
              "$ref": "#/$defs/AgentCVarsWithCol"
            }
          ],
          "title": "Conceptual variables"
        },
        "transform_code": {
          "title": "The code that transforms the data",
          "type": "string"
        },
        "m_code": {
          "title": "The code for the statistical modeling",
          "type": "string"
        }
      },
      "required": [
        "cvars",
        "transform_code",
        "m_code"
      ],
      "title": "EntireAnalysis",
      "type": "object"
    },
    "IVarWithCol": {
      "properties": {
        "description": {
          "title": "Description of the independent variable variable",
          "type": "string"
        },
        "columns": {
          "items": {
            "type": "string"
          },
          "title": "The column(s) in the FINAL dataframe used in the STATISTICAL MODEL that corresponds to the independent variable",
          "type": "array"
        }
      },
      "required": [
        "description",
        "columns"
      ],
      "title": "IVarWithCol",
      "type": "object"
    }
  },
  "properties": {
    "dataset_name": {
      "title": "Dataset Name",
      "type": "string"
    },
    "analyses": {
      "items": {
        "$ref": "#/$defs/EntireAnalysis"
      },
      "title": "Analyses",
      "type": "array"
    }
  },
  "required": [
    "dataset_name",
    "analyses"
  ],
  "title": "DatasetSubmission",
  "type": "object"
}
```
## Contributing

TODO

## Citation

If you use our dataset or models in your research, please cite us as follows:

```bibtex

```
