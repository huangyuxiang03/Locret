CUDA_VISIBLE_DEVICES=0 python phi3-lg-test.py  --metric exam_eval --task_name codeU &
CUDA_VISIBLE_DEVICES=1 python phi3-lg-test.py  --metric ngram_eval --task_name natural_question &
CUDA_VISIBLE_DEVICES=2 python phi3-lg-test.py  --metric ngram_eval --task_name legal_contract_qa &
CUDA_VISIBLE_DEVICES=3 python phi3-lg-test.py  --metric ngram_eval --task_name narrative_qa &
CUDA_VISIBLE_DEVICES=4 python phi3-lg-test.py  --metric ngram_eval --task_name meeting_summ &
CUDA_VISIBLE_DEVICES=5 python phi3-lg-test.py  --metric ngram_eval --task_name review_summ &
