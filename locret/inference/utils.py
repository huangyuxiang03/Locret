from transformers import AutoConfig, AutoTokenizer
from locret.models.phi3.modeling_phi3 import Phi3ForCausalLM
from locret.models.llama.modeling_llama import LlamaForCausalLM
import torch

def load_model_and_tokenizer(model_dir: str, tokenizer_dir: str = None, retainment_head_path: str = None):
    if tokenizer_dir is None:
        tokenizer_dir = model_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if 'phi' in model_dir.lower():
        model = Phi3ForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    elif 'llama' in model_dir.lower():
        model = LlamaForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
    else:
        raise "Model type not supported yet!"
    
    if retainment_head_path is not None:
        ckpt = torch.load(retainment_head_path)
        pruned_ckpt = {}
        for k, v in ckpt.items():
            if 'fc' in k:
                pruned_ckpt[k] = v
        model.load_state_dict(pruned_ckpt, strict=False)
    return model, tokenizer