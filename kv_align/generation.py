
# Refs: StreamingLLM, ChatGPT, https://github.com/huggingface/transformers/issues/25420

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn.functional as F
import time
import os
from collections import defaultdict
import json
import argparse
from get_kv_align import get_kv_align, get_llm, set_up_model
from cache import fix_cache
from parse_model_mode import parse_model_name, parse_mode

@torch.no_grad
def generate(mode, prompt, model, tokenizer, key_model, value_model, num_tokens, print_generation=True, start_size=4, cache_size=54, no_repeat_2gram=True, flush=False):
    
    possible_modes = ["with_recompute", "sliding_window", "key_value_net"]
    assert mode in possible_modes

    recent_size = cache_size - start_size

    apply_fix = True if mode == "key_value_net" else False
    recompute = True if mode == "with_recompute" else False
    use_cache = not recompute

    num_blocks, num_heads, emb_size = set_up_model(model, mode)

    if print_generation:
        print(f"\nStarting generation with model {model.config._name_or_path} with mode {mode} and cache size {cache_size} with no_repeat_2gram {no_repeat_2gram}\n")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    if print_generation:
        print(prompt, end="")

    start_time = time.time()

    outputs = model(input_ids=input_ids, use_cache=use_cache)

    probabilities = F.softmax(outputs.logits[:, -1, :], dim=-1)
    probabilities[0][tokenizer.eos_token_id] = 0
    top_probabilities, top_indices = torch.topk(probabilities, k=5, dim=-1)
    pred_token_idx = top_indices[0, torch.multinomial(top_probabilities[0], num_samples=1)].unsqueeze(0)

    input_ids = torch.cat((input_ids, pred_token_idx), dim=-1)
    generated_token = tokenizer.decode(pred_token_idx.item(), skip_special_tokens=True)
    ngrams = set([(tokenizer("\n").input_ids[0], tokenizer("\n").input_ids[0])]) if no_repeat_2gram else None
    past_key_values = outputs.past_key_values if use_cache else None

    end_time = time.time()
    ttft = end_time - start_time        

    if print_generation:
        print(generated_token, end="", flush=flush) # Generate the First Token

    if no_repeat_2gram:
        zero_indices = []

    prompt_tokens = input_ids.size(-1)
    curr_tokens = 1

    start_time = time.time()

    while True:
        if curr_tokens == num_tokens:
            break
        
        if not recompute:
            outputs = model(input_ids=pred_token_idx, past_key_values=past_key_values, use_cache=True)
        else:
            outputs = model(input_ids=input_ids, use_cache=False)

        probabilities = F.softmax(outputs.logits[:, -1, :], dim=-1)
        probabilities[0][tokenizer.eos_token_id] = 0
        if no_repeat_2gram:
            probabilities[0][zero_indices] = 0
        top_probabilities, top_indices = torch.topk(probabilities, k=20, dim=-1)
        token_idx = top_indices[0, torch.multinomial(top_probabilities[0], num_samples=1)].unsqueeze(0)

        if no_repeat_2gram and (pred_token_idx.item(), token_idx.item()) in ngrams:
            zero_indices.append(token_idx.item())
            continue

        past_key_values = outputs.past_key_values

        if recompute:
            curr_size = input_ids.size(-1)
            input_ids = torch.cat((input_ids, token_idx), dim=-1)
            if curr_size > cache_size:
                input_ids = input_ids[:, 1:]
        else:
            curr_size = past_key_values[0][0].size(-2)
            if curr_size > cache_size:
                past_key_values = fix_cache(past_key_values, start_size, recent_size, apply_fix, key_model, value_model, num_blocks, num_heads, emb_size)

        if no_repeat_2gram:
            zero_indices = []

        curr_tokens += 1

        if no_repeat_2gram:
            ngrams.add((pred_token_idx.item(), token_idx.item()))
            
        generated_token = tokenizer.decode(token_idx.item(), skip_special_tokens=True)
        pred_token_idx = token_idx

        if print_generation:
            print(generated_token, end="", flush=flush)

        if print_generation and curr_tokens + prompt_tokens == cache_size:
            print(f"\n\nMaxed out cache size of {cache_size} tokens (including prompt)\n\n")
        if print_generation and curr_tokens + prompt_tokens == 1024:
            print("\n\nReached 1024 tokens (including prompt)\n\n")

    end_time = time.time()
    total_time = end_time - start_time

    if print_generation:
        print(f"\n\nGenerated {num_tokens} tokens (excluding prompt)")
        print(f"Time to First Token: {ttft * 1000:.2f}ms")
        print(f"Total Time after First Token: {total_time:.2f}s")
        print(f"Time per Token after First Token: {total_time / (num_tokens - 1) * 1000:.2f}ms")

    return ttft, total_time

@torch.no_grad
def run_latency_evaluation(cache_sizes, num_tokens, mode_names, num_trials_per_prompt, prompts, model_name, key_model, value_model, print_ttft=False, no_save=False, no_repeats=True):
    os.makedirs(f'latency', exist_ok=True)
    cache_size_to_avg_times = defaultdict(list) # dictionary from cache_size to list of avg time per token for different modes
    print(f"Starting latency evaluation for model {model_name} for num_tokens {num_tokens}, cache_sizes {cache_sizes}, and modes {mode_names}")
    for cache_size in cache_sizes:
        for mode in mode_names:
            model, tokenizer = get_llm(model_name)
            ttfts = []
            total_times = []
            for prompt in prompts:
                for _ in range(num_trials_per_prompt):
                    ttft, total_time = generate(mode, prompt, model, tokenizer, key_model, value_model, num_tokens=num_tokens, print_generation=False, start_size=4, cache_size=cache_size, no_repeat_2gram=no_repeats)
                    ttfts.append(ttft)
                    total_times.append(total_time)

            ttfts = torch.tensor(ttfts)
            total_times = torch.tensor(total_times)
            
            if not no_save:
                torch.save(total_times, f"latency/total_times_mode_{mode}_num_tokens_{num_tokens}_cache_size_{cache_size}_num_trials_per_prompt_{num_trials_per_prompt}.pt")

            if print_ttft:
                print(f"Avg Time to First Token for Cache Size {cache_size} with Mode {mode}: {ttfts.mean().item() * 1000:.2f}ms")
            print(f"Avg Time per Token after First Token for Cache Size {cache_size} with Mode {mode}: {total_times.mean().item() / (num_tokens - 1) * 1000:.2f}ms\n")
            cache_size_to_avg_times[cache_size].append(total_times.mean().item() / (num_tokens - 1) * 1000)
    
    # Save the dictionary to a JSON file
    if not no_save:
        with open('latency/cache_size_to_avg_times.json', 'w') as json_file:
            json.dump(cache_size_to_avg_times, json_file, indent=4)


def main(args):

    model_name, mode = parse_model_name(args), parse_mode(args)
    key_model, value_model = get_kv_align(model_name)

    # Some default prompts from MT Bench (https://huggingface.co/spaces/lmsys/mt-bench), possibly shortened to ensure # of tokens is less than 32
    prompts = [
        "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.", 
        "The vertices of a triangle are (0, 0), (-1, 1), (3, 3). What is its area?",
        "A is the father of B. B is the father of C. What is the relationship between A and C?", 
        "Develop a Python program that reads all the text files under a directory and returns top-5 words with the most number of occurrences.", 
        "How do the stages of life shape our understanding of time and mortality?"
    ]

    if not args.latency:
        model, tokenizer = get_llm(model_name)
        generate(mode, prompts[0], model, tokenizer, key_model, value_model, num_tokens=args.num_tokens, cache_size=args.cache_size, print_generation=True, start_size=4, no_repeat_2gram=not args.allow_repeats, flush=args.flush)
    else:
        cache_sizes = [32, 64, 128, 255]
        if args.kv_align_only:
            mode_names = ["key_value_net"]
        else:
            mode_names = ["sliding_window", "key_value_net", "with_recompute"]
        run_latency_evaluation(cache_sizes, args.num_tokens, mode_names, args.num_trials_per_prompt, prompts, model_name, key_model, value_model, no_save=args.no_save, no_repeats=not args.allow_repeats)

if __name__ == "__main__":
    """
    Example commands for generaton using kv-align, window attention (StreamingLLM for rope embeddings), sliding window with recomputation, resp.:
        python kv_align/generation.py --model_name gpt2 --num_tokens 2000 --cache_size 54 --flush --mode kv
        python kv_align/generation.py --model_name gpt2 --num_tokens 2000 --cache_size 54 --flush --mode sw
        python kv_align/generation.py --model_name gpt2 --num_tokens 2000 --cache_size 54 --flush --mode wr

    Example command for latency eval:
        python kv_align/generation.py --model_name gpt2 --latency --num_tokens 2000
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="distilbert/distilgpt2")
    parser.add_argument("--num_tokens", type=int, default=2000)

    # Argument for latency evaluation
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--kv_align_only", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--num_trials_per_prompt", type=int, default=3)

    # Arguments for generation
    parser.add_argument("--cache_size", type=int, default=54)
    parser.add_argument("--allow_repeats", action="store_true")
    parser.add_argument("--mode", type=str, default="key_value_net") # Options: "sliding_window" (sw), "key_value_net" (kvn), "with_recompute" (wr)
    parser.add_argument("--flush", action="store_true")

    args = parser.parse_args()

    main(args)
 