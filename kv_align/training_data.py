# Ref: https://github.com/karpathy/build-nanogpt, StreamingLLM, ChatGPT

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from datasets import load_dataset
warnings.simplefilter('ignore', category=FutureWarning)
import os

batch_size = 5
cache_size = 256
start_size = 4
stride_size = 32
num_shards = 11
num_data = 1000

model_name = "HuggingFaceTB/SmolLM2-135M"
assert model_name in ["HuggingFaceTB/SmolLM2-135M"]

os.makedirs('training_data', exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

dataset_str = ''.join(dataset["train"]["text"][:num_data])
encoded = tokenizer(dataset_str, return_tensors='pt')

input_ids = encoded.input_ids

num_batches = (input_ids.size(-1) - cache_size) // (stride_size * batch_size)

input_ids_batches = [[input_ids[0, (batch_index * batch_size + j) * stride_size: (batch_index * batch_size + j) * stride_size + cache_size] for j in range(batch_size)] for batch_index in range(num_batches)]
input_ids_batches = [torch.concat(input_ids_batches[i]) for i in range(num_batches)]
input_ids_batches = torch.concat(input_ids_batches)
input_ids_batches = input_ids_batches.view(num_batches, batch_size, cache_size)
torch.save(input_ids_batches, "training_data/input_ids_batches.pt")


input_ids_batches = torch.load("training_data/input_ids_batches.pt")
keys = []
values = []

shard_size = num_batches // num_shards

for idx, batch in enumerate(input_ids_batches):
    initial_output = model(
        input_ids=batch,
        use_cache=True,
    )
    past_key_values = initial_output.past_key_values

    key, value, = list(zip(*past_key_values))
    key = torch.stack(key).transpose(0, 1)
    value = torch.stack(value).transpose(0, 1)
    key.shape, value.shape
    keys.append(key.cpu())
    values.append(value.cpu())

    if (idx + 1) % shard_size == 0:
        print(f"Working on shard {idx // shard_size + 1}")

        keys = torch.cat(keys)
        values = torch.cat(values)

        torch.save(keys, f"rtraining_data/misaligned_keys_shard{idx // shard_size + 1}.pt")
        torch.save(values, f"training_data/misaligned_values_shard{idx // shard_size + 1}.pt")
        
        print(f"Misaligned keys.shape: {keys.shape}, values.shape: {values.shape}")
        
        keys = []
        values = [] 

new_input_ids_batches = torch.cat((input_ids_batches[:,:, :start_size], input_ids_batches[:,:, start_size+1:]), dim=-1)

keys = []
values = []

for idx, batch in enumerate(new_input_ids_batches):
    initial_output = model(
        input_ids=batch,
        use_cache=True,
    )
    past_key_values = initial_output.past_key_values

    key, value, = list(zip(*past_key_values))
    key = torch.stack(key).transpose(0, 1)
    value = torch.stack(value).transpose(0, 1)
    key.shape, value.shape
    keys.append(key.cpu())
    values.append(value.cpu())

    if (idx + 1) % shard_size == 0:
        print(f"Working on shard {idx // shard_size + 1}")

        keys = torch.cat(keys)
        values = torch.cat(values)

        torch.save(keys, f"rtraining_data/aligned_keys_shard{idx // shard_size + 1}.pt")
        torch.save(values, f"training_data/aligned_values_shard{idx // shard_size + 1}.pt")

        print(f"Aligned keys.shape: {keys.shape}, values.shape: {values.shape}")
               
        keys = []
        values = [] 
