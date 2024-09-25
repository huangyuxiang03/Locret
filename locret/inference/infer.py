import torch
@torch.no_grad()
def generate(model, input_ids, max_new_tokens, eos_token_id, budget_size=6000, local_len=100, chunk_size=3072, stabilizers=2500):
    seq_len = input_ids.shape[-1]
    # prefill first
    past_key_values = None
    scores = [None for _ in range(100)]

    for i in range(0, seq_len - local_len, chunk_size):
        b = i
        e = min(i + chunk_size, seq_len - local_len)
        ipt = input_ids[:, b:e]
        output = model(ipt, use_cache=True, past_key_values=past_key_values, output_attentions=True)
        past_key_values = output.past_key_values

        pruned_kv_cache = []
        kv_shape = past_key_values[0][0].shape
        for j in range(model.config.num_hidden_layers):
            if scores[j] is None:
                cur_score = output.attentions[j][:, :e-b, :].to("cuda:0") 
                scores[j] = cur_score
            else:
                cur_score = output.attentions[j][:, :e-b, :].to("cuda:0")
                scores[j] = torch.cat(
                    (scores[j], cur_score), dim=-2,
                )
            
            sc = scores[j].clone()
            selected_num = min(budget_size, sc.shape[-2])
            if b + chunk_size < seq_len - local_len:
                sc[:, -stabilizers:, :] = torch.finfo(sc.dtype).max 
            selected_indices = torch.topk(sc[0, :, :], k=selected_num, dim=-2)[1].transpose(0, 1).sort().values # (32, budget_size)
            selected_indices_ = selected_indices.clone().transpose(0, 1).unsqueeze(0)
            scores[j] = torch.gather(scores[j], 1, selected_indices_.to(sc.device))
            selected_indices = selected_indices.unsqueeze(0).unsqueeze(-1).repeat(kv_shape[0], 40, 1, kv_shape[3]) # TODO
            k = torch.gather(past_key_values[j][0], 2, selected_indices.to(past_key_values[j][0].device))
            v = torch.gather(past_key_values[j][1], 2, selected_indices.to(past_key_values[j][1].device))
            pruned_kv_cache.append((k, v))
        past_key_values = pruned_kv_cache                
        del pruned_kv_cache
        torch.cuda.empty_cache()
    b = e
    e = seq_len     
    position_ids = torch.arange(b, e, dtype=torch.int, device=input_ids.device).unsqueeze(0)
    output = model(input_ids[:, b:e], use_cache=True, past_key_values=past_key_values, output_attentions=True)
    del past_key_values
    past_key_values = output.past_key_values
    

    input_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [input_tokens.item()]
    position_ids = torch.tensor([[seq_len]]).cuda()
    for i in range(max_new_tokens - 1):
        output = model(input_tokens, past_key_values=past_key_values)
        position_ids += 1
        past_key_values = output.past_key_values
        input_tokens = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if input_tokens.item() == eos_token_id:
            break
        generated_tokens.append(input_tokens.item())
    generated_tokens = torch.tensor(generated_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    input_ids = torch.cat((input_ids, generated_tokens), dim=-1)
    return input_ids