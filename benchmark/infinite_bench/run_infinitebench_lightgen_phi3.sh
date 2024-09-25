# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

TASKS=("passkey" "longbook_choice_eng" "math_find" "longbook_qa_chn" "longbook_qa_eng" "longdialogue_qa_eng" "code_debug" "longbook_sum_eng" "number_string")

export TOKENIZERS_PARALLELISM=false
SCRIPT_DIR=$(dirname "$0")

for task in ${TASKS[@]}; do
echo $task
python "$SCRIPT_DIR/run_infinitebench_pred.py" \
    --task $task \
    --model_name_or_path ${1} \
    --data_dir ./data \
    --output_dir ./results \
    --max_seq_length $2 \
    --rewrite \
    --trust_remote_code \
    --num_eval_examples $3 --topk 1 --starting_layer 0 --attn_type $4
done

# quant
# bash run_infinitebench_pred.sh <model_dir> 131072 -1 minference
