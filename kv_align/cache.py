import torch

@torch.no_grad
def k_slice(x, start, end):
    return x[:, :, start:end, ...]

@torch.no_grad
def fix_cache(past_key_values, start_size, recent_size, apply_fix, key_model, value_model, num_blocks, num_heads, emb_size):

    seq_len = past_key_values[0][0].size(-2)

    past_key_values = [
        [
            torch.cat(
                [
                    k_slice(k, 0, start_size),
                    k_slice(k, start_size + 1, seq_len),
                ],
                dim=2,
            ),
            torch.cat(
                [
                    k_slice(v, 0, start_size),
                    k_slice(v, start_size + 1, seq_len),
                ],
                dim=2,
            ),
        ]
        for k, v in past_key_values
    ]

    if not apply_fix:
        return past_key_values    
    
    for block_idx in range(0, num_blocks):
        old_key, old_value = past_key_values[block_idx]
        old_key = old_key.to("cuda")
        old_value = old_value.to("cuda")
        B = torch.full((num_heads * recent_size,), block_idx) 
        P = torch.cat([torch.arange(0, recent_size)] * num_heads)
        H = torch.arange(num_heads).repeat_interleave(recent_size)
        X_key = old_key[:, :, start_size:, :]
        X_value = old_value[:, :, start_size:, :]
        P, B, H = P.to("cuda"), B.to("cuda"), H.to("cuda")
        X_key = X_key.reshape(-1, emb_size)
        X_value = X_value.reshape(-1, emb_size)
        past_key_values[block_idx] = torch.cat( (old_key[:,:,:start_size,:], key_model(X_key, P, B, H).view(1, num_heads, recent_size, -1)), 2).to("cpu"), torch.cat( (old_value[:,:,:start_size,:], value_model(X_value, P, B, H).view(1, num_heads, recent_size, -1)), 2).to("cpu")            

    return past_key_values