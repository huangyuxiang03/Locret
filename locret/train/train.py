# This code is adapted from https://github.com/SafeAILab/EAGLE/blob/main/eagle/train/main.py
import argparse
from typing import Tuple, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List

from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer

from locret.models.phi3.modeling_phi3 import Phi3ForCausalLM
from locret.models.llama.modeling_llama import LlamaForCausalLM

import torch.multiprocessing as mp
import json
from safetensors import safe_open
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed
import time


def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='./output')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    if 'phi' in args.model_dir.lower():
        args.data_dir += '/phi-3'
        args.checkpoint_dir += '/phi-3'
    elif 'llama' in args.model_dir.lower():
        args.data_dir += '/llama-3.1'
        args.checkpoint_dir += '/llama-3.1'
    else:
        raise "Model type not supported yet!"

    train_config = {
        "lr": args.lr,
        "bs": args.bs,
        "gradient_accumulation_steps": 1,
        "datapath": f"{args.data_dir}",
        "is_warmup": True,
        "num_epochs": 1,
        "num_warmup_steps": 2000,
        "total_steps": 3000,
        "num_workers": 1,
        "max_len": 10240,
        "grad_clip": 0.5,
        "b1": 0.9,
        "b2": 0.95,
        "save_freq": 50,
        "kv_head_num": 32 if 'phi' in args.model_dir.lower() else 8,
    }
    return args, train_config

class CustomDataset(Dataset):
    def __init__(self, datapath, max_len):
        self.data = datapath
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        input_ids = data['input_ids'][:self.max_len][None, :]
        weights = data['weights'][:, 0, :, :self.max_len][None, :]
        length = input_ids.shape[1]
        new_data["input_ids"] = input_ids
        new_data["weights"] = weights
        new_data["loss_mask"] = [1] * length

        return new_data


class DataCollatorWithPadding:
    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) == 1
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])

        batch = {
            "input_ids": batch_input_ids,
            "weights": features[0]['weights'],
            "loss_mask": batch_loss_mask,
        }
        return batch

def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath

if __name__ == '__main__':
    mp.set_start_method('spawn')
    train_beg = time.time()
    args, train_config = parse_args()
    set_seed(0)
    torch.manual_seed(0)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=train_config["gradient_accumulation_steps"])

    criterion = nn.SmoothL1Loss(reduction='none')
    mse_loss = nn.MSELoss()


    if 'phi3' in args.model_dir.lower():
        model = Phi3ForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    elif 'llama' in args.model_dir.lower():
        model = LlamaForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.bfloat16)

    datapath = list_files(train_config["datapath"])

    traindatapath = datapath[:train_config["total_steps"]]
    traindataset = CustomDataset(traindatapath, train_config["max_len"])
    train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=False,
                            collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"])

    if accelerator.is_main_process:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

    for param in model.parameters():
        param.requires_grad = False

    for n, param in model.named_parameters():
        if "fc" in n:
            param.requires_grad = True
            
    optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))


    num_epochs = train_config["num_epochs"]
    num_warmup_steps = train_config["num_warmup_steps"]
    total_steps = train_config["total_steps"]
    is_warmup = train_config["is_warmup"]

    if is_warmup:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=total_steps)

        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )


    model.train()

    _acc = 0
    tot_step = 0
    for epoch in range(num_epochs):
        for batch_idx, data in enumerate((train_loader)):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(data['input_ids'], output_attentions = True)
                data['weights'] = data['weights'].to("cuda:0")
                data_weights = (data['weights'])

                scores = output.attentions

                tot_loss = 0
                tot_ent = 0
                sparsity = 0.1

                corr = 0
                tot_cnt = 0

                smooth_loss = 0
                for i, s in enumerate(scores):
                    s = s[0].transpose(0, 1)
                    std_s = data_weights[0, i]
                    s = s[..., :std_s.shape[-1], :]
                    _k = int(sparsity * s.shape[-1])
                    top_s = torch.topk(s, k=_k, dim=-1).indices
                    top_std = torch.topk(std_s, k=_k, dim=-1).indices
                    for j in range(train_config['kv_head_num']):
                        overlap = len(
                            set(top_s[j].tolist()) & set(top_std[j].tolist())
                        )
                        corr += overlap
                        tot_cnt += _k
                    loss = criterion(s.to(std_s.device), std_s.float())
                    smooth_loss += mse_loss(s[:, :-1], s[:, 1:]) 
                    loss = loss.mean(dim=-1)
                    tot_loss += loss.sum()
                    tot_ent += loss.shape[-1]
                
                loss = tot_loss / tot_ent
                    
                print(f"epoch: {epoch:4d} | step: {batch_idx:4d} | loss: {loss:.10f} | smooth_loss: {smooth_loss:.10f} | acc: {(corr / tot_cnt): .4f} | len: {data['input_ids'].shape[-1]} | tot_step: {tot_step}")
                loss += smooth_loss.to(loss.device) * 0.0025
                accelerator.backward(loss)
                accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
                optimizer.step()
                if is_warmup:
                    scheduler.step()
            tot_step += 1
            if accelerator.is_local_main_process and tot_step % train_config['save_freq'] == 0:
                accelerator.save_state(output_dir=f"{args.checkpoint_dir}/state_{epoch}_{tot_step}")
                
    model.save_pretrained(f"{args.checkpoint_dir}/final_model")
    train_end = time.time()
    print(f"training takes {train_end - train_beg}s")
    
"""
python train.py --model_dir /home/test/test01/hyx/Phi3-mini-128K
"""

