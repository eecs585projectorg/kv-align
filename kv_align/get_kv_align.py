# Refs: StreamingLLM, ChatGPT

import torch
from torch import nn
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from modify_llama import enable_llama_pos_shift_attention

warnings.simplefilter('ignore', category=FutureWarning)

class KeyNetworkDistilGPT2(nn.Module):
    def __init__(self, num_pos=256, num_blocks=6, num_heads=12, emb_pos_dim=32, emb_blocks_dim=4, emb_heads_dim=8):
        super().__init__()

        self.emb_pos = nn.Embedding(num_pos, emb_pos_dim)
        self.emb_blocks = nn.Embedding(num_blocks, emb_blocks_dim)
        self.emb_heads = nn.Embedding(num_heads, emb_heads_dim)
        
        self.fc1 = nn.Linear(64 + emb_pos_dim + emb_blocks_dim + emb_heads_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 64)

        
    def forward(self, x, p, b, h):
        p_emb = self.emb_pos(p)
        b_emb = self.emb_blocks(b)
        h_emb = self.emb_heads(h)
        x_z = torch.cat([x, p_emb, b_emb, h_emb], dim=1)  # Combine continuous and embedded categorical data
        x_z = torch.relu(self.fc1(x_z))
        x_z = torch.relu(self.fc2(x_z))
        x = self.fc3(x_z) + x
        return x
        
class ValueNetworkDistilGPT2(nn.Module):
    def __init__(self, num_pos=256, num_blocks=6, num_heads=12, emb_pos_dim=32, emb_blocks_dim=4, emb_heads_dim=8):
        super().__init__()

        self.emb_pos = nn.Embedding(num_pos, emb_pos_dim)
        self.emb_blocks = nn.Embedding(num_blocks, emb_blocks_dim)
        self.emb_heads = nn.Embedding(num_heads, emb_heads_dim)
        
        self.fc1 = nn.Linear(64 + emb_pos_dim + emb_blocks_dim + emb_heads_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)

        
    def forward(self, x, p, b, h):
        p_emb = self.emb_pos(p)
        b_emb = self.emb_blocks(b)
        h_emb = self.emb_heads(h)
        x_z = torch.cat([x, p_emb, b_emb, h_emb], dim=1)  # Combine continuous and embedded categorical data
        x_z = torch.relu(self.fc1(x_z))
        x_z = torch.relu(self.fc2(x_z))
        x = self.fc3(x_z) + x
        return x


class KeyNetworkSmolLM2(nn.Module):
    def __init__(self, num_pos=256, num_blocks=30, num_heads=3, emb_pos_dim=32, emb_blocks_dim=4, emb_heads_dim=8):
        super().__init__()

        self.emb_pos = nn.Embedding(num_pos, emb_pos_dim)
        self.emb_blocks = nn.Embedding(num_blocks, emb_blocks_dim)
        self.emb_heads = nn.Embedding(num_heads, emb_heads_dim)
        
        self.fc1 = nn.Linear(64 + emb_pos_dim + emb_blocks_dim + emb_heads_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)

        
    def forward(self, x, p, b, h):
        p_emb = self.emb_pos(p)
        b_emb = self.emb_blocks(b)
        h_emb = self.emb_heads(h)
        x_z = torch.cat([x, p_emb, b_emb, h_emb], dim=1)  # Combine continuous and embedded categorical data
        x_z = torch.relu(self.fc1(x_z))
        x_z = torch.relu(self.fc2(x_z))
        x = self.fc3(x_z) + x
        return x
        
class ValueNetworkSmolLM2(nn.Module):
    def __init__(self, num_pos=256, num_blocks=30, num_heads=3, emb_pos_dim=32, emb_blocks_dim=4, emb_heads_dim=8):
        super().__init__()

        self.emb_pos = nn.Embedding(num_pos, emb_pos_dim)
        self.emb_blocks = nn.Embedding(num_blocks, emb_blocks_dim)
        self.emb_heads = nn.Embedding(num_heads, emb_heads_dim)
        
        self.fc1 = nn.Linear(64 + emb_pos_dim + emb_blocks_dim + emb_heads_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)

        
    def forward(self, x, p, b, h):
        p_emb = self.emb_pos(p)
        b_emb = self.emb_blocks(b)
        h_emb = self.emb_heads(h)
        x_z = torch.cat([x, p_emb, b_emb, h_emb], dim=1)  # Combine continuous and embedded categorical data
        x_z = torch.relu(self.fc1(x_z))
        x_z = torch.relu(self.fc2(x_z))
        x = self.fc3(x_z) + x
        return x

def get_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

def get_data(dataset_name, tokenizer):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        num_data = 1000
        dataset_str = ''.join(dataset["test"]["text"][:num_data])

    elif dataset_name == "everyday":
        dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")
        num_data = 500
        dataset_str = ' '.join([msg['content'] for msg in [item for sublist in dataset["train"]["messages"][:num_data] for item in sublist]])
    else:
        raise ValueError("Unsupported Dataset")
    encoded = tokenizer(dataset_str, return_tensors='pt')
    data_ids = encoded.input_ids
    return data_ids
    
def get_kv_align(model_name):
    if model_name == "distilbert/distilgpt2":
        key_model = KeyNetworkDistilGPT2().to("cuda")
        key_model.eval()

        value_model = ValueNetworkDistilGPT2().to("cuda")
        value_model.eval()

        checkpoint = torch.load("models/keys_network_distilgpt2.pth")
        key_model.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load("models/values_network_distilgpt2.pth")
        value_model.load_state_dict(checkpoint["model_state_dict"])

        return key_model, value_model
    elif model_name == "HuggingFaceTB/SmolLM2-135M":
        key_model = KeyNetworkSmolLM2().to("cuda")
        key_model.eval()

        value_model = ValueNetworkSmolLM2().to("cuda")
        value_model.eval()

        checkpoint = torch.load("models/keys_network_smollm2.pth")
        key_model.load_state_dict(checkpoint["model_state_dict"])

        checkpoint = torch.load("models/values_network_smollm2.pth")
        value_model.load_state_dict(checkpoint["model_state_dict"])

        return key_model, value_model
    raise ValueError("Invalid Model")

def set_up_model(model, mode):
    if model.config._name_or_path == "HuggingFaceTB/SmolLM2-135M":
        if mode == "sliding_window":
            enable_llama_pos_shift_attention(model)
        num_blocks = 30
        num_heads = 3
        emb_size = 64
    elif model.config._name_or_path == "distilbert/distilgpt2":
        num_blocks = 6
        num_heads = 12
        emb_size = 64
    else:
        raise ValueError("Invalid Model")
    return num_blocks, num_heads, emb_size