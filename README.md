
<h1 align="center">
<img src="assets/logo.png" width="100" alt="logo" />
<br>
BLADE: Benchmarking Language Model Agents for Data-Driven Science
</h1>

Dataset and code for ["BLADE: Benchmarking Language Model Agents for Data-Driven Science"](https://arxiv.org/abs/2408.09667)

We are working on a hold-out test set. Details soon!

## 📝 Introduction
BLADE is a comprehensive benchmark designed to evaluate Language Model (LM) Agents on writing justifiable analyses on real-world scientific research questions from data (e.g., *Are soccer players with a dark skin tone more likely than those with a light skin tone to receive red cards from referees?* from [Silberzahn et al.](https://journals.sagepub.com/doi/10.1177/2515245917747646)). In particular, BLADE evaluates Agents' ability to iteratively integrate scientific domain knowledge, statistical expertise, and data understanding to make nuanced analytical decisions

BLADE consists of X dataset and research question pairs with high-quality ground truth analysis decisions (i.e., choice of conceptual construct, transformations, statistical model) made by expert data scientists and researchers who independently conducted the analyses. In addition, BLADE contains Y multiple choice questions for discerning justifiable analysis decisions. 


![Main](assets/main_white.png)
<p align="left">
  <em><b>Overview of BLADE Construction and Evaluation.</b> We gathered research questions and datasets from existing research papers, crowd-sourced analysis studies and statistic textbooks as well as analyses from expert annotators (boxes 1-3). Given a research question and dataset, LM agents generate a full analysis containing the relevant conceptual variables, a data transform function, and a statistical modeling function (boxes 1, 4, and 5). BLADE then performs automatic evaluation against the ground truth (box 6).</em>
</p>

## 🚀 Getting Started
To get started with BLADE, follow the steps below:
### 1. Installation
Clone the repository:
```bash
git clone https://github.com/behavioral-data/BLADE.git
cd BLADE
```
Install locally (developed in python=3.10.14)
```bash
# recommended to do this inside another environment
conda create --name blade python=3.10 -y    
conda activate blade
pip install -e .
```

### 2. LM Setup
Next, set the API keys for different LM services. BLADE both not only evalutes Language Models but needs one for evaluation.
```bash
# for openai
export OPENAI_API_KEY=<your key>

# for google gemini
export GEMINI_API_KEY=<your key>

# for anthropic
export ANTHROPIC_API_KEY=<your key>
```
Some default model configurations (e.g., environment variable for the api key) are specified in [llm_config.yml](blade_bench/conf/llm_config.yml). You can also set your own configurations by creating your own yaml file folloing the format in `llm_config.yml` and setting the environment variable `LLM_CONFIG_PATH` to the file.

Here's a minimal example to test that the llm is working.
```python
from blade_bench.llms import llm
gen = llm(provider="anthropic", model="claude-3.5-sonnet")
response = gen.generate([{"role": "user", "content": "Hello world"}])
```

### 3. Running LMs and Agent
We provide a starter script to run a basic one shot LM or ReACT agent for our benchmark.

```
Usage: run_gen_analyses.py [OPTIONS]

  For a given dataset and research question, generate analyses for the dataset
  using a language model or a basic ReAct agent that interacts with a notebook
  environment.

  Running this generates the following files in output_dir:

  - command.sh: A bash script that contains the command used to run this script
  - config.json: The configuration used to run this experiment
  - run.log: The log file for the multirun experiment
  - llm.log: The log file for LM prompts and responses for the experiment
  - multirun_analyses.json: The analyses generated. **Note**: This file is used in run_get_eval.py to get the evaluation results.
  - llm_analysis_*.py: The code generated for each run (if it was generated properly) for quick reference

Options:
  --run_dataset [fish|boxes|conversation|reading|crofoot|panda_nuts|fertility|hurricane|teachingratings|mortgage|soccer|affairs|amtl|caschools]
                                  Dataset to run  [required]
  -n, --num_runs INTEGER          Number of runs to perform  [default: 10]
  --use_agent                     Whether to use agent or just the base LM
  --no_cache_code_results         [ONLY used when use_agent=True] Whether to
  --no_cache_code_results         [ONLY used when use_agent=True] Whether to
                                  cache code results when running code.
  --no_use_data_desc              Whether to use data description in the
                                  prompts for the LM  [default: True]
  --llm_config_path FILE          Path to the LLM config file, used to specify
                                  the provider, model, and text generation
                                  config such as the temperature.  [default:
                                  ./conf/llm.yaml]
  --llm_provider [openai|azureopenai|groq|mistral|together|gemini|anthropic|huggingface]
                                  Provider for the LLM to override the config
                                  file at llm_config_path
  --llm_model TEXT                Model for the LLM to override the config
                                  file at llm_config_path. 
  --llm_eval_config_path FILE     Path to the LLM eval config file, used to
                                  specify the provider, model, and text
                                  generation config such as the temperature.
                                  [default: ./conf/llm_eval.yaml]
  --output_dir DIRECTORY          output directory to store saved analyses
  --help                          Show this message and exit.
```
This will write the results to the folder specified by `output_dir`. After running the script, in the output folder, there will be a `multirun_analyses.json` file which is used for evaluation.

An example is provided in [example/multirun_analyses.json](example/multirun_analyses.json).

### 4. Evaluating Agent Generated Analyses
We provide a starter script to evaluate the outputs of `run_gen_analyses.py`. Run `run_get_eval.py` as follows:

```
Usage: run_get_eval.py [OPTIONS]

  Runs evaluation and saves the results to the output_dir directory. Running
  this saves the following key files:

  - command.sh: A bash script that contains the command used to run this script
  - eval_results.json of the EvalResults class
  - eval_metrics.json of the MetricsAcrossRuns class containing the metrics
  - llm_history.json of the LLM history class containing the prompt history
  
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
                                 Default is []
  --diversity_n_samples INTEGER  Number of samples to use for diversity
                                 metrics
  --help                         Show this message and exit.
```
Here is an example:
```bash
python run_get_eval.py --multirun_load_path ./examples/multirun_analyses.json
```

## 🔍 Data Exploration Functions
To access the dataset and research question we can:
```python
from blade_bench.data import load_dataset_info, list_datasets, DatasetInfo

all_datasets = list_datasets()
dinfo: DatasetInfo = load_dataset_info("soccer", load_df=True)
rq = dinfo.research_question
df = dinfo.df
```

To explore the ground truth annotations, we can:
```python
from blade_bench.data import load_ground_truth, AnnotationDBData

# each dataset annotations will be prepared when it is run the first time
gnd_truth: AnnotationDBData = load_ground_truth('soccer') 
print(len(gnd_truth.transform_specs))
print(len(gnd_truth.cv_specs))
```
More details about the structure of the ground truth is available in the paper.

## 🎯 Evaluating a Submission on BLADE
To evalute your own agent analysis for a dataset in BLADE, the LM agent must generate a `json` file that conforms to the schema in [example/submission_schema.json](example/submission_schema.json). An example is shown in [example/submission_analyses.json](example/submission_analyses.json). Then, we just need to specify --submission_load_path when running `run_get_eval.py`.

```bash
python run_get_eval.py --submission_load_path ./example/submission_analyses.json
```

## Citation

If you use our dataset or models in your research, please cite us as follows:

```bibtex
@article{gu2024bladebenchmarkinglanguagemodel,
      title={BLADE: Benchmarking Language Model Agents for Data-Driven Science}, 
      author={Ken Gu and Ruoxi Shang and Ruien Jiang and Keying Kuang and Richard-John Lin and Donghe Lyu and Yue Mao and Youran Pan and Teng Wu and Jiaqian Yu and Yikun Zhang and Tianmai M. Zhang and Lanyi Zhu and Mike A. Merrill and Jeffrey Heer and Tim Althoff},
      year={2024},
      journal = {arXiv},
      url={https://arxiv.org/abs/2408.09667}, 
}
```
