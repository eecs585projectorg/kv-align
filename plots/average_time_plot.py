# This plots figure 3 on the paper
import matplotlib.pyplot as plt
import numpy as np

# Data for average time per token after the first token in JSON format
avg_time_data = {
    "32": [45.70486546039677, 60.80546553646666, 66.1329104199746],
    "64": [49.54282787709886, 59.29366415275243, 65.40216124279544],
    "128": [45.212133857435546, 62.646440420896674, 69.36658327759301],
    "255": [57.8873037791152, 65.31179282731546, 64.3828716915258]
}

cache_sizes = list(avg_time_data.keys())
modes = ["StreamingLLM", "KV-Align", "Sliding Window With\nRecomputation"]
time_values = np.array([avg_time_data[size] for size in cache_sizes]).T

x = np.arange(len(cache_sizes))  # Label locations
width = 0.2  # Width of bars

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, (mode, times) in enumerate(zip(modes, time_values)):
    ax.bar(x + i * width, times, width, label=mode)

# Configure plot
ax.set_xlabel('Cache Sizes')
ax.set_ylabel('Avg Time per Token After First Token (ms)')
ax.set_xticks(x + width)
ax.set_xticklabels(cache_sizes)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Adjust legend position

# Save and display the graph
plt.tight_layout()  # Adjust layout to make room for the legend
plt.savefig('average_time_per_token.png', dpi=300)
plt.show()
