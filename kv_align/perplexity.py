# Refs: ChatGPT, eval_long_ppl.py of StreamingLLM

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn.functional as F
import time
import os
import argparse
from get_kv_align import get_kv_align, get_llm, get_data, set_up_model
from cache import fix_cache
from parse_model_mode import parse_model_name

@torch.no_grad
def compute_perplexity(mode, cache_size, model, tokenizer, data_ids, num_tokens, key_model, value_model, print_ngll=False, start_size=4): # input: number of label tokens for perplexity calculations
    
    possible_modes = ["with_recompute", "sliding_window", "key_value_net"]   
    assert mode in possible_modes

    recent_size = cache_size - start_size

    assert data_ids.size(-1) >= num_tokens + cache_size + 1
    data_ids_trunc = data_ids[:,:num_tokens + cache_size + 1]

    apply_fix = True if mode == "key_value_net" else False
    recompute = True if mode == "with_recompute" else False

    num_blocks, num_heads, emb_size = set_up_model(model, mode)

    nglls = []
    label_idx = cache_size + 1

    input_ids = data_ids_trunc[:,:label_idx]
    label = data_ids_trunc[:,label_idx].unsqueeze(0)
    outputs = model(input_ids=input_ids, use_cache=True)
    probabilities = F.softmax(outputs.logits[:, -1, :], dim=-1).squeeze()
    ngll = -torch.log(probabilities[label.item()]).item()
    nglls.append(ngll)

    past_key_values = outputs.past_key_values if not recompute else None
    if print_ngll:
        print(f"Neg Log Likelihood: {ngll:.3f}, Label: {tokenizer.decode(label[0])}")

    for _ in range(num_tokens - 1):
        if not recompute:
            # For StreamingLLM or KV-Align
            past_key_values = fix_cache(past_key_values, start_size, recent_size, apply_fix, key_model, value_model, num_blocks, num_heads, emb_size)
            outputs = model(input_ids=label, past_key_values=past_key_values, use_cache=True)
        else:
            input_ids = torch.cat((input_ids, label), dim=-1)
            input_ids = input_ids[:, 1:]
            outputs = model(input_ids=input_ids, use_cache=False)
        label_idx += 1
        label = data_ids_trunc[:,label_idx].unsqueeze(0)
        probabilities = F.softmax(outputs.logits[:, -1, :], dim=-1).squeeze()
        ngll =  -torch.log(probabilities[label.item()]).item()
        nglls.append(ngll)
        if not recompute:
            past_key_values = outputs.past_key_values

        if print_ngll:
            print(f"Neg Log Likelihood: {ngll:.3f}, Label: {tokenizer.decode(label[0])}")
    return nglls

@torch.no_grad
def run_perplexity_evaluation(num_tokens, model_name, dataset_name="wikitext", no_save=False, print_ngll=False):
    os.makedirs(f'nglls', exist_ok=True)

    cache_sizes = [32, 64, 128, 255]

    _, tokenizer = get_llm(model_name)
    key_model, value_model = get_kv_align(model_name)
    data_ids = get_data(dataset_name, tokenizer)

    if args.kv_align_only:
        mode_names = ["key_value_net"]
    else:
        mode_names = ["sliding_window", "key_value_net", "with_recompute"]    
    
    model_name_short = model_name.split("/")[-1]

    print(f"\nStarting perplexity evaluation for model {model_name} for num_tokens {num_tokens}, cache_sizes {cache_sizes}, dataset {dataset_name}, and modes {mode_names}\n")

    for cache_size in cache_sizes:
        for mode in mode_names:
            model, _ = get_llm(model_name)
            model.eval()
            start_time = time.time()
            nglls = torch.tensor(compute_perplexity(mode, cache_size, model, tokenizer, data_ids, num_tokens, key_model, value_model, print_ngll=print_ngll))
            end_time = time.time()
            
            if not no_save:
                torch.save(nglls, f"nglls/nglls_mode_{mode}_tokens_{num_tokens}_cache_size_{cache_size}_model_name_{model_name_short}_data_set_{dataset_name}.pt")

            perplexity = torch.exp(torch.mean(nglls))
            print(f"Perplexity after {num_tokens} Label Tokens and Cache Size {cache_size} with Mode {mode}: {perplexity.item():.4f}. Took {end_time - start_time:.2f} seconds\n")

def main(args):
    model_name = parse_model_name(args)
    run_perplexity_evaluation(args.num_tokens, model_name, args.dataset_name, args.no_save, args.print_ngll)

if __name__ == "__main__":
    """
    Example command for perplexity eval for DistilGPT2 and SmolLM2, resp.:
        python kv_align/perplexity.py --model_name gpt2 --dataset_name wikitext --num_tokens 20000 
        python kv_align/perplexity.py --model_name smollm2 --dataset_name wikitext --num_tokens 20000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert/distilgpt2")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--kv_align_only", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--print_ngll", action="store_true")
    parser.add_argument("--num_tokens", type=int, default=20_000)

    args = parser.parse_args()
    main(args)

