export VLLM_WORKER_MULTIPROC_METHOD=spawn
python phi3-vanilla-test.py  --metric exam_eval --task_name codeU 
python phi3-vanilla-test.py  --metric ngram_eval --task_name natural_question 
python phi3-vanilla-test.py  --metric ngram_eval --task_name legal_contract_qa 
python phi3-vanilla-test.py  --metric ngram_eval --task_name narrative_qa 
python phi3-vanilla-test.py  --metric ngram_eval --task_name meeting_summ 
python phi3-vanilla-test.py  --metric ngram_eval --task_name review_summ 
