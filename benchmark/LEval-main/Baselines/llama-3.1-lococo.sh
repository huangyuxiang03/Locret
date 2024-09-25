CUDA_VISIBLE_DEVICES=6 python llama-3.1-lococo-test.py  --metric exam_eval --task_name codeU &
CUDA_VISIBLE_DEVICES=1 python llama-3.1-lococo-test.py  --metric ngram_eval --task_name natural_question &
CUDA_VISIBLE_DEVICES=2 python llama-3.1-lococo-test.py  --metric ngram_eval --task_name legal_contract_qa &
CUDA_VISIBLE_DEVICES=7 python llama-3.1-lococo-test.py  --metric ngram_eval --task_name narrative_qa &
CUDA_VISIBLE_DEVICES=4 python llama-3.1-lococo-test.py  --metric ngram_eval --task_name meeting_summ &
CUDA_VISIBLE_DEVICES=5 python llama-3.1-lococo-test.py  --metric ngram_eval --task_name review_summ &
