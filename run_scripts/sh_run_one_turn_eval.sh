#!/bin/bash
# Array of llm values
# after running sh_run.sh do we run this script
llm_values=("anthropic:claude-3.5-sonnet")

run_datasets=("affairs" "hurricane" "amtl" "boxes" "caschools" "crofoot" "fish" "mortgage" "panda_nuts" "reading" "soccer" "teachingratings")

set -e  # Exit immediately if a command exits with a non-zero status
trap 'echo "Script failed at: $(eval echo $BASH_COMMAND)"' ERR

for model_provider in "${llm_values[@]}"; do
    IFS=':' read -r provider llm  <<< "$model_provider"
    for run_dataset in "${run_datasets[@]}"; do
        save_path="./outputs/final/${llm}/${run_dataset}/multirun_analyses.json"
        output_folder="./outputs/final/${llm}/${run_dataset}/eval"
        echo "save_path=$save_path"
        echo "output_folder=$output_folder"

        # Execute the python script with the current values
        python run_get_eval.py --multirun_load_path $save_path \
            --ks '[]' \
            --output_dir $output_folder
    done
done