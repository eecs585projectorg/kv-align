# Plots for perplexity for input up to 20000 tokens
# use https://github.com/eecs585projectorg/cse585_project/tree/main/final_evaluations/perplexity_eval/distilgpt2/nglls_gpt2
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load data for StreamingLLM
streaming_data = torch.load('nglls_mode_sliding_window_tokens_20000_cache_size_128.pt')

# Load data for KV-Align
kv_align_data = torch.load('nglls_mode_key_value_net_tokens_20000_cache_size_128.pt')

# Apply moving average for smoothing
window_size = 500  # Smoothing window size
smoothed_streaming = np.convolve(streaming_data.numpy(), np.ones(window_size) / window_size, mode='valid')
smoothed_kv_align = np.convolve(kv_align_data.numpy(), np.ones(window_size) / window_size, mode='valid')

# Create a combined plot
plt.figure(figsize=(12, 7))
plt.plot(smoothed_streaming, color='blue', linewidth=2, label='StreamingLLM')
plt.plot(smoothed_kv_align, color='orange', linewidth=2, label='KV-Align')

# Customize the plot
plt.xlabel('Input Length')
plt.ylabel('Log PPL')
plt.ylim(0, 10)  # Scale the y-axis
plt.legend(loc='upper right')

# Set x-axis ticks and labels
max_x = len(smoothed_streaming) * window_size  # Approximate max x-value after smoothing
tick_positions = np.linspace(0, max_x, 5, dtype=int)  # Tick positions
tick_labels = ['0', '5k', '10k', '15k', '20k']  # Corresponding labels
plt.xticks(tick_positions / window_size, tick_labels)  # Adjust for smoothing

plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
