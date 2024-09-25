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
# torch.cuda.set_per_process_memory_fraction(0.3)
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
    Phi3Config,
    Phi3ForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
)
# from modeling_llama import LlamaForCausalLM
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
# from vllm import LLM, SamplingParams
from peft import LoraConfig, get_peft_model

from sir_llm.eval_utils import Evaluator
from sir_llm.enable_streaming_llm import enable_streaming_llm

from transformers import Phi3Config, Phi3ForCausalLM
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


# from minference import MInference
class SirLLMConfig:
    def __init__(self):
        self.start_size = 4
        self.token_entropy_size = 16384
        # self.token_entropy_size = 6000
        self.recent_size = 1000
        self.max_gen_len = 20
        self.decay_ratio = 1

# BS = 1024
BS = 3072
@torch.no_grad()
def gen(model, tokenizer, input_ids, max_new_tokens, eos_token_id):
    s_config = SirLLMConfig()
    s_config.max_gen_len = max_new_tokens
    kv_cache = enable_streaming_llm(
        model, start_size=s_config.start_size, 
        recent_size=s_config.recent_size,
        token_entropy_size=s_config.token_entropy_size,
    )
    generator = Evaluator(model, tokenizer, s_config)
    token_entropy = None
    past_key_values = None
    for b in range(0, input_ids.shape[-1], BS):
        e = min(input_ids.shape[-1], b + BS)
        temp = kv_cache.evict_for_space_token_entropy(past_key_values,token_entropy, e - b)
        past_key_values=temp[0]
        token_entropy=temp[1]
        past_key_values, _token_entropy, logits = generator._greedy_generate_token_entropy_simple(input_ids[:, b:e], continue_len=1,past_key_values=past_key_values,token_entropy=token_entropy)
        if token_entropy is None:
            token_entropy = _token_entropy
        else:
            token_entropy += _token_entropy

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
    return generated_tokens

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
        
        # breakpoint()

        seq_len = input_tensors['input_ids'].shape[-1]
        ans_len = len(gt_tokens)

        cur_len = 0

        generated_tokens = gen(model, tok, input_tensors['input_ids'], generation_config.max_new_tokens, 128009)
        # generated_tokens = gen(model, tok, input_tensors['input_ids'], generation_config.max_new_tokens, 32007)


        output = tok.decode(generated_tokens, skip_special_tokens=True)
        output = output.strip()
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

    if attn_type == "vllm":
        llm = LLM(
            model_name,
            max_num_seqs=1,
            swap_space=64,
            gpu_memory_utilization=0.98,
            max_model_len=max_seq_length,
        )
    else:
        config = LlamaConfig.from_pretrained(
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

        llm = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            resume_download=None,
            trust_remote_code=trust_remote_code,
        )
    # llm = minference_patch(llm)

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


    config = LlamaConfig.from_pretrained(
        model_name, resume_download=None, trust_remote_code=True
    )
    model, tok = load_model(
        model_name,
        args.topk,
        args.starting_layer,
        args.topk_dims_file_path,
        args.use_sparq,
        attn_type=args.attn_type,
        max_seq_length=max_seq_length,
        is_search=args.is_search,
        use_snapkv=args.use_snapkv,
        trust_remote_code=args.trust_remote_code,
        kv_cache_cpu=args.kv_cache_cpu,
        kv_cache_cpu_device=args.kv_cache_cpu_device,
    )

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
            score = compute_scores(output_path, data_name, real_model_name, max_seq_length)
            print(score)
            exit(0)
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
            torch.cuda.reset_peak_memory_stats()
            pid = os.getpid() 
            env_alloc = psutil.Process(pid).memory_info().rss
            monitor_thread = threading.Thread(target=monitor_memory, args=(pid,))
            monitor_thread.start()
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
            torch.cuda.empty_cache()
            print(f"Max GPU memory: {torch.cuda.max_memory_allocated()/ 1024/1024/1024:.2f}GB")
            stop_monitor = True
            monitor_thread.join()
            max_memory_usage = max_cpu_memory
            print(f"Max CPU memory: {(max_memory_usage - env_alloc) / 1024 ** 3:.2f} GB")
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
            torch.cuda.empty_cache()

        result_file_path = f"{real_model_name}_{args.attn_type}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
