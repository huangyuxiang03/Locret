export VLLM_WORKER_MULTIPROC_METHOD=spawn
python llama-3.1-vanilla-test.py  --metric exam_eval --task_name codeU 
python llama-3.1-vanilla-test.py  --metric ngram_eval --task_name natural_question 
python llama-3.1-vanilla-test.py  --metric ngram_eval --task_name legal_contract_qa 
python llama-3.1-vanilla-test.py  --metric ngram_eval --task_name narrative_qa 
python llama-3.1-vanilla-test.py  --metric ngram_eval --task_name meeting_summ 
python llama-3.1-vanilla-test.py  --metric ngram_eval --task_name review_summ 
