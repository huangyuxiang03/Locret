# This code is adapted from https://github.com/SafeAILab/EAGLE/blob/main/eagle/ge_data/ge_data_all_llama3.py
import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from fastchat.model.model_adapter import get_conversation_template
from locret.models.phi3.modeling_phi3 import Phi3ForCausalLM
from locret.models.llama.modeling_llama import LlamaForCausalLM




def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--start', type=int, default=0, help='The staring index of the original dataset.')
    parser.add_argument('--end', type=int, default=5000, help='The ending index of the original dataset. This should be slightly larger than `num`, as some generation might fail.')
    parser.add_argument('--num', type=int, default=3000, help='The number of trained entries generated.')
    parser.add_argument('--chunk_size', type=int, default=4096, help='The chunk size of chunked prefill')
    parser.add_argument('--model_dir', type=str, default=None, help='The directory of model')
    parser.add_argument('--data_path', type=str, default='./longalpaca_sharegpt_type.json', help='The path of dataset. The dataset should be in shareGPT format.')
    parser.add_argument('--save_dir', type=str, default='./output', help='The output directory of generated data entries.')

    args = parser.parse_args()
    return args

def build_dataset_rank(tokenizer, data_path, model_type):
    ds = load_dataset('json', data_files=data_path)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 4

    # THIS FUNCTION NEEDS TO ADAPT TO THE SPECIFIC MODEL!!!
    # Phi3 and Llama-3 happens to share the same function
    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        for i in range(len(examples['id'])):
            conv = get_conversation_template(model_type)
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
            source= examples['conversations'][i]
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]
            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                if sentence["from"]=="gpt":
                    sentence["value"]=" "+sentence["value"]
                conv.append_message(role, sentence["value"])
            conversation=conv.get_prompt()
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id=tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
            ).input_ids[0]
            loss_mask=torch.ones_like(input_ids)

            sep = conv.sep + conv.roles[1] + ": "
            sep2 = conv.sep + conv.roles[0] + ": "

            turns = conversation.split(sep2)
            turns = [turns[0] + sep2 + turns[1]] + turns[2:]
            cur_len = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids)
                # Ignore the user instructions
                loss_mask[max(0, cur_len - 5): cur_len + instruction_len] = 0
                cur_len += turn_len
                cur_len+=5

            loss_mask[cur_len:] = 0


            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])
        return new_examples
    
    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    ds1.set_format(type="torch")
    return ds1

def writedata(name,data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length=len(os.listdir(name))
    idx=current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

@torch.no_grad()
def generate_data(model, group_size, data, chunk_size):
    input_ids = data["input_ids"]
    prefix_len = data['loss_mask'].shape[-1] - data['loss_mask'].sum()
    past_key_values = None
    for b in range(0, prefix_len, chunk_size):
        e = min(b + chunk_size, prefix_len)
        output = model(input_ids[:, b:e].cuda(), use_cache=True, past_key_values=past_key_values)
        past_key_values = output.past_key_values
    output = model(input_ids[:, prefix_len:].cuda(), past_key_values=past_key_values, output_attentions=True, plain_attn=True)
    attentions = output.attentions
    attn = attentions[0]

    loss_mask = data['loss_mask']
    prefix_len = loss_mask.shape[-1] - loss_mask.sum()
    labels = []
    for attn in attentions:
        aa, bb, cc, dd = attn.shape
        attn = attn.reshape(aa, bb // group_size, group_size, cc, dd).max(dim=2).values
        labels.append(
            attn[:, :, :, :prefix_len].max(dim=-2).values
        )
    labels = torch.stack(labels)
    
    td = {"input_ids": input_ids.cpu()[0][:prefix_len], "weights": labels}
    return td


if __name__ == '__main__':
    args = parse_args()

    if 'phi' in args.model_dir.lower():
        model_type = "phi-3-mini-128k-instruct"
        model_id = "phi-3"
        group_size = 1
    elif 'llama' in args.model_dir.lower():
        model_type = "llama-3"
        model_id = "llama-3.1"
        group_size = 4
    else:
        raise "Model type not supported yet!"


    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    dataset = build_dataset_rank(tokenizer, args.data_path, model_type)
    print(dataset)
    if model_type == "phi-3-mini-128k-instruct":
        model = Phi3ForCausalLM.from_pretrained(args.model_dir,  device_map="auto",torch_dtype=torch.bfloat16)
    elif model_type == "llama-3":
        model = LlamaForCausalLM.from_pretrained(args.model_dir,  device_map="auto",torch_dtype=torch.bfloat16)
    model.eval()


    outdir = f'{args.save_dir}/{model_id}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    cnt = 0
    succ = 0
    for data in tqdm(dataset):
        try:
            outdata = generate_data(model, group_size, data, args.chunk_size)
            writedata(outdir,outdata)
            succ += 1
        except:
            print(f"error on idx: {cnt}")
        cnt += 1
        if succ >= args.num:
            break
    print(f"Generation finished. {succ} entries is generated by {cnt} original data. {cnt - succ} failed.")
        
'''
python data_gen.py --model_dir /home/test/test01/hyx/Phi3-mini-128K
'''
