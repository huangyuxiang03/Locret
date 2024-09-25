import math
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_llama import LlamaForCausalLM
import transformers
# -*- coding:utf-8 -*-
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from LEval_config import *
from tqdm import tqdm
from sir_llm.eval_utils import Evaluator
from sir_llm.enable_streaming_llm import enable_streaming_llm


class SirLLMConfig:
    def __init__(self):
        self.start_size = 4
        self.token_entropy_size = 16384
        self.recent_size = 1000
        self.max_gen_len = 20
        self.decay_ratio = 1

BS = 1024
@torch.no_grad()
def gen(model, input_ids, max_new_tokens, eos_token_id):
    s_config = SirLLMConfig()
    s_config.max_gen_len = max_new_tokens
    generator = Evaluator(model, tokenizer, s_config)
    token_entropy = None
    past_key_values = None

    device = input_ids.device

    global_token_entropy = None
    for b in range(0, input_ids.shape[-1], BS):
        e = min(input_ids.shape[-1], b + BS)
        past_key_values, token_entropy, logits = generator._greedy_generate_token_entropy_simple(input_ids[:, b:e], continue_len=1,past_key_values=past_key_values,token_entropy=token_entropy)
        if global_token_entropy is None:
            global_token_entropy = torch.tensor(token_entropy, device=device)
        else:
            cur_token_entropy = torch.tensor(token_entropy, device=device)
            global_token_entropy = torch.cat((global_token_entropy, cur_token_entropy), dim=-1)
        
        if past_key_values[0][0].shape[-2] > s_config.start_size + s_config.recent_size + s_config.token_entropy_size:
            selected_indices = s_config.start_size + torch.topk(global_token_entropy[s_config.start_size:-s_config.recent_size], k=s_config.token_entropy_size, dim=-1)[1].sort().values
            start_indices = torch.arange(0, s_config.start_size).to(device)
            recent_indices = torch.arange(global_token_entropy.shape[-1] - s_config.recent_size, global_token_entropy.shape[-1]).to(device)
            selected_indices = torch.cat((start_indices, selected_indices, recent_indices))
            global_token_entropy = torch.gather(global_token_entropy, -1, selected_indices)
            selected_indices = selected_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            shape = past_key_values[0][0].shape
            selected_indices = selected_indices.repeat(shape[0], shape[1], 1, shape[3])
            pruned_kv = []
            for k, v in past_key_values:
                _k = torch.gather(k, 2, selected_indices)
                _v = torch.gather(v, 2, selected_indices)
                pruned_kv.append((_k, _v))
            past_key_values = pruned_kv


    generated_tokens = []
    input_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens.append(input_id.item())

    for _ in range(max_new_tokens-1):
        output = model(input_id, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        input_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if input_id.item() == eos_token_id:
            break
        generated_tokens.append(input_id.item())
    generated_tokens = torch.tensor(generated_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    input_ids = torch.cat((input_ids, generated_tokens), dim=-1)
    return input_ids


def main():
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    for file_name in key_data_pairs:
        fw = open(file_name, "w")
        data = key_data_pairs[file_name]
        B_INST, E_INST = "<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        B_SYS, E_SYS = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"
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

                sample = gen(model, inputs.input_ids, max_new_tokens, 128009)
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

    model_path = f"/home/test/test01/hyx/Llama-3.1-8B-Instruct"
    open_source_model = f"Llama-3.1-8B-Instruct-sirllm"

    max_length = k_to_number(args.max_length) - max_new_tokens


    data_save_path = f"Predictions/{args.metric}/{open_source_model}"
    print(f"Your prediction file will be saved to: {data_save_path}")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # breakpoint()
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map=device)
    model = model.eval()

    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main())
