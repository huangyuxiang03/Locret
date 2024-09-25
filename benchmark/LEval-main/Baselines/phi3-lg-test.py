import math
from functools import partial

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer
from modeling_phi3 import Phi3ForCausalLM
import transformers
# -*- coding:utf-8 -*-
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from LEval_config import *
from tqdm import tqdm
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


SN = 6000
# SN = 3000
loc = 100
lloc = 2500
# lloc = 0
SD = 3072

# @memory_monitor
def locret(model, input_ids, max_new_tokens, eos_token_id):
    print(f"seq_len: {input_ids.shape[-1]}")
    seq_len = input_ids.shape[-1]

    with torch.no_grad():
        # prefill first
        past_key_values = None
        scores = [None for _ in range(32)]
        global_scores = [None for _ in range(32)]
        block_scores = [None for _ in range(32)]


        prev_ent = 0

        for i in range(0, seq_len - loc, SD):
            b = i
            e = min(i + SD, seq_len - loc)
            position_ids = torch.arange(b, e, dtype=torch.int, device=input_ids.device).unsqueeze(0)
            ipt = input_ids[:, b:e]
            output = model(ipt, position_ids=position_ids, use_cache=True, past_key_values=past_key_values, output_attentions=True)
            past_key_values = output.past_key_values

            pruned_kv_cache = []
            kv_shape = past_key_values[0][0].shape
            for j in range(32):
                if scores[j] is None:
                    cur_score = output.attentions[j][:, :e-b, :].to("cuda:0") 
                    scores[j] = cur_score
                else:
                    cur_score = output.attentions[j][:, :e-b, :].to("cuda:0")
                    scores[j] = torch.cat(
                        (scores[j], cur_score), dim=-2,
                    )
                
                sc = scores[j].clone()
                selected_num = min(SN, sc.shape[-2])
                if b + SD < seq_len - loc:
                    sc[:, -lloc:, :] = torch.finfo(sc.dtype).max ###
                selected_indices = torch.topk(sc[0, :, :], k=selected_num, dim=-2)[1].transpose(0, 1).sort().values # (32, SN)
                selected_indices_ = selected_indices.clone().transpose(0, 1).unsqueeze(0)
                scores[j] = torch.gather(scores[j], 1, selected_indices_.to(sc.device))
                selected_indices = selected_indices.unsqueeze(0).unsqueeze(-1).repeat(kv_shape[0], 1, 1, kv_shape[3])
                k = torch.gather(past_key_values[j][0], 2, selected_indices.to(past_key_values[j][0].device))
                v = torch.gather(past_key_values[j][1], 2, selected_indices.to(past_key_values[j][1].device))
                pruned_kv_cache.append((k, v))
            past_key_values = pruned_kv_cache                
            del pruned_kv_cache
            torch.cuda.empty_cache()
        b = e
        e = seq_len     
        position_ids = torch.arange(b, e, dtype=torch.int, device=input_ids.device).unsqueeze(0)
        output = model(input_ids[:, b:e], use_cache=True, position_ids=position_ids, past_key_values=past_key_values, output_attentions=True)
        del past_key_values
        past_key_values = output.past_key_values
        

        input_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens = [input_tokens.item()]
        position_ids = torch.tensor([[seq_len]]).cuda()
        for i in range(max_new_tokens):
            output = model(input_tokens, position_ids=position_ids, past_key_values=past_key_values)
            position_ids += 1
            past_key_values = output.past_key_values
            input_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if input_tokens.item() == eos_token_id:
                break
            generated_tokens.append(input_tokens.item())
        generated_tokens = torch.tensor(generated_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
        input_ids = torch.cat((input_ids, generated_tokens), dim=-1)
    return input_ids

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
                sample = locret(model, inputs.input_ids, max_new_tokens, 32007)
                # sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
                prompt_length = inputs.input_ids.size()[-1]
                output = tokenizer.decode(sample[0][prompt_length:])

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
    open_source_model = f"Phi3-mini-128K-lg-anti_haystack"

    max_length = k_to_number(args.max_length) - max_new_tokens


    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    print(f"Your prediction file will be saved to: {data_save_path}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # breakpoint()
    model = Phi3ForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device)
    # ckpt = torch.load("/home/test/test01/hyx/train/ckpt_phi3_max_4096/state_0_3000/model.bin")
    # ckpt = torch.load("/home/test/test01/hyx/train/ckpt_phi3_max_longalign/state_0_3000/model.bin")
    ckpt = torch.load("/home/test/test01/hyx/train/ckpt_phi3_max_anti_haystack/state_0_fin/model.bin")
    pruned_ckpt = {}
    for k, v in ckpt.items():
        if 'fc' in k:
            pruned_ckpt[k] = v
    model.load_state_dict(pruned_ckpt, strict=False)
    model = model.eval()

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
