# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
os.environ['HF_EVALUATE_OFFLINE'] = '1'
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
torch.cuda.set_per_process_memory_fraction(0.3)
from args import parse_args
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)
from modeling_phi3 import Phi3ForCausalLM
from configuration_phi3 import Phi3Config
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
# from vllm import LLM, SamplingParams

# from minference import MInference
import tracemalloc

# SN = 100000000000000
SN = 6000
# SN = 2500
# SN = 49017
loc = 100

# lloc = 2500
lloc = 3000

SD = 3072

import threading, psutil
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


# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def jaccard_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
    attn_type: str = "vllm",
    ground_truth: str = "",
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    ground_truth = str(ground_truth)
    assert len(ground_truth) > 0
    gt_tokens = tok(ground_truth, max_length=50).input_ids
    input_tokens = truncate_by_tokens(input_text.replace("$$MASK$$", '**MASK**'), tok, max_input_length - len(gt_tokens))
    # breakpoint()
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
    if attn_type == "vllm":
        if len(input_tokens) != 1:
            input_tokens = [input_tokens]
        outputs = model.generate(
            prompt_token_ids=input_tokens,
            sampling_params=generation_config,
        )
        output = outputs[0].outputs[0].text
        output = output.strip()
    else:
        input_tensors = {
            "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
        }
        # cache = SinkCache(window_length=200000, num_sink_tokens=10000)
        # if attn_type == "minference_kv_cache_cpu":
        #     input_tensors["use_cache"] = False
        


        seq_len = input_tensors['input_ids'].shape[-1]
        ans_len = len(gt_tokens)

        with torch.no_grad():
            # prefill first
            past_key_values = None
            scores = [None for _ in range(32)]
            global_scores = [None for _ in range(32)]
            block_scores = [None for _ in range(32)]


            prev_ent = 0

            for i in range(0, seq_len - loc, SD):
                # print(f"{i+1} of {seq_len - loc}")
                # breakpoint()
                b = i
                e = min(i + SD, seq_len - loc)
                ipt = input_tensors['input_ids'][:, b:e]
                output = model(ipt, use_cache=True, past_key_values=past_key_values, output_attentions=True)
                past_key_values = output.past_key_values

                pruned_kv_cache = []
                kv_shape = past_key_values[0][0].shape
                for j in range(32):
                    if scores[j] is None:
                        cur_score = output.attentions[j][:, :e-b, :]
                        scores[j] = cur_score
                        global_scores[j] = cur_score
                    else:

                        cur_score = output.attentions[j][:, :e-b, :]
                        scores[j] = torch.cat(
                            (scores[j], cur_score), dim=-2,
                        )
                        global_scores[j] = torch.cat(
                            (global_scores[j], cur_score), dim=-2,
                        )
                    
                    sc = scores[j].clone()
                    selected_num = min(SN, sc.shape[-2])
                    if b + SD < seq_len - loc:
                        sc[:, -lloc:, :] = torch.finfo(sc.dtype).max ###
                    selected_indices = torch.topk(sc[0, :, :], k=selected_num, dim=-2)[1].transpose(0, 1).sort().values # (32, SN)
                    selected_indices_ = selected_indices.clone().transpose(0, 1).unsqueeze(0)
                    scores[j] = torch.gather(scores[j], 1, selected_indices_)
                    selected_indices = selected_indices.unsqueeze(0).unsqueeze(-1).repeat(kv_shape[0], 1, 1, kv_shape[3])
                    k = torch.gather(past_key_values[j][0], 2, selected_indices.to(past_key_values[j][0].device))
                    v = torch.gather(past_key_values[j][1], 2, selected_indices.to(past_key_values[j][1].device))
                    pruned_kv_cache.append((k, v))
                past_key_values = pruned_kv_cache                
                del pruned_kv_cache
                torch.cuda.empty_cache()

            print("="*20)
            std_gs = torch.load("std_gs.pt", map_location='cuda')
            for ii in [(7, 0), (15, 1), (23, 2), (31, 3)]:
                i, j = ii
                std = std_gs[i][0, :, j]
                cdd = global_scores[i][0, :, j]
                delta = (std - cdd).abs()
                delta = delta.mean()
                print(f"{delta.item():.4f}")

            
            print("="*20)
            exit(0)
            b = e
            e = seq_len     
            output = model(input_tensors['input_ids'][:, b:e], use_cache=True, past_key_values=past_key_values, output_attentions=True)
            del past_key_values
            past_key_values = output.past_key_values

            # hidden_states = output.hidden_states
            # std_hs = torch.load("std.pt", map_location='cuda')
            # print("="*20)
            # for i in range(33):
            #     hs = hidden_states[i][:, -1, :]
            #     shs = std_hs[i][:, -1, :]
            #     delta = shs - hs
            #     print(f"{delta.abs().max().item():.4f}")
            
            # print("="*20)
            # exit(0)

            input_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens = [input_tokens.item()]
            for i in range(generation_config.max_new_tokens):
                output = model(input_tokens, past_key_values=past_key_values)
                past_key_values = output.past_key_values
                input_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if input_tokens.item() == tok.eos_token_id:
                    break
                generated_tokens.append(input_tokens.item())

        output = tok.decode(generated_tokens, skip_special_tokens=True)
        output = output.strip()
    # print(input_text[:5000], input_text[-5000:])
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str,
    topk: int = -1,
    starting_layer: int = -1,
    topk_dims_file_path: str = "",
    use_sparq: bool = False,
    attn_type: str = "vllm",
    max_seq_length: int = None,
    is_search: bool = False,
    use_snapkv: bool = False,
    trust_remote_code: bool = False,
    kv_cache_cpu: bool = False,
    kv_cache_cpu_device: str = "cpu",
):
    tok = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code,
    )
    tok.pad_token = tok.eos_token
    minference_patch = MInference(
        attn_type,
        model_name,
        config_path=topk_dims_file_path,
        starting_layer=starting_layer,
        use_snapkv=use_snapkv,
        is_search=is_search,
        kv_cache_cpu=kv_cache_cpu,
        kv_cache_cpu_device=kv_cache_cpu_device,
    )

    if attn_type == "vllm":
        llm = LLM(
            model_name,
            max_num_seqs=1,
            swap_space=64,
            gpu_memory_utilization=0.98,
            max_model_len=max_seq_length,
        )
    else:
        config = Phi3Config.from_pretrained(
            model_name, resume_download=None, trust_remote_code=trust_remote_code
        )
        if "LWM" in model_name:
            c = {
                "theta": 10000000,
                "max_sequence_length": 131072,
                "scan_attention": True,
                "scan_query_chunk_size": 1024,
                "scan_key_chunk_size": 1024,
                "scan_mlp": True,
                "scan_mlp_chunk_size": 1024,
                "scan_layers": True,
            }
            config.update(c)

        llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            resume_download=None,
            trust_remote_code=trust_remote_code,
        )
    llm = minference_patch(llm)

    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()
    # check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]


    tok = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=True,
    )
    config = Phi3Config.from_pretrained(
        model_name, resume_download=None, trust_remote_code=True
    )
    model = Phi3ForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        resume_download=None,
        trust_remote_code=True,
    )

    # ckpt = torch.load("/home/huangyuxiang/kvcache/train/ckpt2/state_0_8000/pytorch_model.bin")
    ckpt = torch.load("/home/test/test01/hyx/train/ckpt_phi3_max/state_0_3000/model.bin") ### current use
    # ckpt = torch.load("/home/huangyuxiang/kvcache/train/good_ckpt/smooth_0.05_5000_steps.bin")
    # ckpt = torch.load("/home/huangyuxiang/kvcache/train/ckpt_reg_smooth2/state_0_fin/pytorch_model.bin")
    pruned_ckpt = {}
    for k, v in ckpt.items():
        if 'fc' in k:
            pruned_ckpt[k] = v
    model.load_state_dict(pruned_ckpt, strict=False)

    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        if args.attn_type == "vllm":
            generation_config = SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                # temperature=0,
                # top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )

        # Data
        result_dir = Path(args.output_dir, f"{real_model_name}_{args.attn_type}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")

        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, real_model_name, max_seq_length)
        # breakpoint()

        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id:
            # if i < 10:
                continue
            # breakpoint()
            input_text = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            # print(input_text.index(ground_truth), len(input_text), input_text.index(ground_truth) / len(input_text))
            # print(f"====== Example {i} ======")
            # breakpoint()
            # torch.cuda.reset_peak_memory_stats()
            # pid = os.getpid() 
            # env_alloc = psutil.Process(pid).memory_info().rss
            # monitor_thread = threading.Thread(target=monitor_memory, args=(pid,))
            # monitor_thread.start()
            pred = get_pred(
                model,
                tok,
                input_text,
                max_input_length=max_seq_length - max_new_tokens,
                verbose=args.verbose,
                generation_config=generation_config,
                attn_type=args.attn_type,
                ground_truth=ground_truth[0],
            )
            print("Ground Truth", get_answer(eg, data_name))
            # torch.cuda.empty_cache()
            # print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/ 1024/1024/1024:.2f}GB")
            # stop_monitor = True
            # monitor_thread.join()
            # max_memory_usage = max_cpu_memory
            # print(f"Max CPU memory: {(max_memory_usage - env_alloc) / 1024 ** 3:.2f} GB")
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)

        result_file_path = f"{real_model_name}_{args.attn_type}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
