import math
from functools import partial

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from modeling_phi3 import Phi3ForCausalLM
import transformers
# -*- coding:utf-8 -*-
import argparse
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from LEval_config import *
from tqdm import tqdm
from omegaconf import OmegaConf
from inf_llm.utils import patch_hf, GreedySearch
import threading, psutil, time
max_cpu_memory = 0
stop_monitor = False
def monitor_memory(pid, interval=0.01):
    process = psutil.Process(pid)
    global max_cpu_memory
    global stop_monitor
    try:
        while True:
            mem_info = process.memory_info()
            max_cpu_memory = max(max_cpu_memory, mem_info.rss)
            time.sleep(interval)
            if stop_monitor:
                return
    except psutil.NoSuchProcess:
        pass


def memory_monitor(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        pid = os.getpid()
        env_alloc = psutil.Process(pid).memory_info().rss


        global stop_monitor
        stop_monitor = False
        def monitor_memory(pid):
            global stop_monitor
            process = psutil.Process(pid)
            global max_cpu_memory
            max_cpu_memory = env_alloc
            while not stop_monitor:
                current_memory = process.memory_info().rss
                max_cpu_memory = max(max_cpu_memory, current_memory)
                time.sleep(0.01)  # Check memory usage every second

        monitor_thread = threading.Thread(target=monitor_memory, args=(pid,))
        monitor_thread.start()

        # Call the decorated function
        result = func(*args, **kwargs)

        # Print GPU and CPU memory usage
        print(f"Max GPU memory: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f}GB")
        stop_monitor = True
        monitor_thread.join()
        global max_cpu_memory
        max_memory_usage = max_cpu_memory
        print(f"Max CPU memory: {(max_memory_usage - env_alloc) / 1024 ** 3:.2f} GB")
        exit(0)
        return result
    
    return wrapper

def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        B_INST, E_INST = "<|user|>\n", "<|end|>\n<|assistant|>\n"
        B_SYS, E_SYS = "<|system|>\n", "<|end|>\n"
        sys_prompt = get_sys_prompt(args, file_name)

        for d in tqdm(data):
            document = d['input']
            cnt = 0
            while num_tokens_from_string(document, tokenizer) > max_length:
                if "code" not in file_name:
                    document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
                else:
                    document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
                cnt += 250                

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context
                elif "topic" in file_name:
                    context = document + "\n\n" + inst
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} \nQuestion: {inst}.  Please directly give the answer without any additional output or explanation "
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                    message += "\nAnswer:"
                else:
                    context = "Document is as follows. {document} Instruction: {inst} " + f"\nAnswer this question with {len(out.split())} words."
                    message = B_INST + B_SYS + sys_prompt + E_SYS + context + E_INST
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long document>")

                inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                # sample = locret(model, inputs.input_ids, max_new_tokens, 32007)
                searcher = GreedySearch(model, tokenizer)
                extra_end_token_ids = []
                extra_end_token_ids.append(tokenizer.encode("<|end|>", add_special_tokens=False)[0])

                @memory_monitor
                def gen(*args, **kwargs):
                    return searcher.generate(*args, **kwargs)

                outputs = gen(input_ids=inputs.input_ids, max_length=max_new_tokens, chunk_size=8192, extra_end_token_ids=extra_end_token_ids)
                searcher.clear()
                # sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                # prompt_length = inputs.input_ids.size()[-1]
                # output = tokenizer.decode(sample[0][prompt_length:])
                output = outputs[0]
                save_d[f'{open_source_model}_pred'] = output.replace('</s>', '')
                save_d['evaluation'] = d['evaluation']

                # test the factuality in scientific fiction
                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "Please directly answer without any additional output or explanation. \nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[0][prompt_length:])
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("[document]:",text_inputs[:100] + "...")
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
                # break
        fw.close()
        # break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument('--max_length', default="128k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--gpu', type=int, default=0)

    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')

    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')
    args = parser.parse_args()

    model_path = f"/home/test/test01/hyx/Phi3-mini-128K"
    open_source_model = f"Phi3-mini-128K-infllm"

    max_length = k_to_number(args.max_length) - max_new_tokens


    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    print(f"Your prediction file will be saved to: {data_save_path}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # breakpoint()
    model = Phi3ForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device)
    model = model.eval()    
    inf_config = OmegaConf.load("inf-llm_phi-3.yaml").model
    model = patch_hf(model, inf_config.type, **inf_config)

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
