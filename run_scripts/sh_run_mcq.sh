#!/bin/bash

# Array of llm values
llm_values=("anthropic:claude-3.5-sonnet")

for model_provider in "${llm_values[@]}"; do
    IFS=':' read -r provider llm  <<< "$model_provider"
    echo "running with llm=$llm, provider=$provider"

    # Execute the python script with the current values
    python run_mcq.py --llm_provider $provider \
        --llm_model $llm \

done