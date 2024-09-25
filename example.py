from locret import generate, load_model_and_tokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='sp')
    parser.add_argument('--model_dir', type=str, default=None, help='The directory of model')
    parser.add_argument('--retainment_head_path', type=str, default=None, help='The directory of model')
    args = parser.parse_args()
    return args

def make_input(digits):
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * 2000
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * 4000
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query
    

if __name__ == '__main__':
    args = parse_args()
    ans = 76384
    input_str = make_input(ans)
    
    if 'phi' in args.model_dir.lower():
        eos_token_id = 32007
    elif 'llama' in args.model_dir.lower():
        eos_token_id = 128009
    
    model, tokenizer = load_model_and_tokenizer(args.model_dir, retainment_head_path=args.retainment_head_path)
    enc = tokenizer(input_str, return_tensors='pt').to("cuda")
    print(f"Input Sequence Length: {enc.input_ids.shape[-1]}")
    output = generate(model, enc.input_ids, eos_token_id=eos_token_id, max_new_tokens=6, budget_size=100, stabilizers=10)
    output_str = tokenizer.decode(output[0])
    print(f"Standard answer: {ans}")
    print(f"Generated answer: ...{output_str[-100:]}")
    
    
