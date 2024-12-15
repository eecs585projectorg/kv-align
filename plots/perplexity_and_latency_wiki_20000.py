# This plots figure 1 on paper
# # use wiki 20000 from this link https://github.com/eecs585projectorg/cse585_project/tree/main/final_evaluations/perplexity_eval/distilgpt2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data for Cache Sizes
cache_sizes = [32, 64, 128, 256]
modes = ["Sliding Window\nwith Recomputation", "StreamingLLM", "KV-Align"]
perplexity_values = [
    [104.0200, 75.6148, 58.5745, 48.8632],  # Sliding Window with Recomputation
    [593.4364, 798.6519, 1507.3171, 1980.2808],  # StreamingLLM
    [318.3841, 145.4735, 148.3063, 333.8580],  # KV-Align
]
time_values = [
    [695.65, 1105.47, 1749.84, 3077.36],  # Sliding Window with Recomputation
    [372.52, 362.66, 370.47, 389.59],  # StreamingLLM
    [563.43, 536.83, 581.63, 649.80],  # KV-Align
]

# Markers and colors for models
markers = ["^", "s", "o"]  # Triangle, Square, Star
colors = ["blue", "orange", "green", "red"]

plt.figure(figsize=(10, 6))

# Plot for each model
for i, mode in enumerate(modes):
    for j, cache_size in enumerate(cache_sizes):
        plt.scatter(
            time_values[i][j],  # Latency on x-axis
            perplexity_values[i][j],  # Perplexity on y-axis
            label=f"{mode} - Cache {cache_size}" if j == 0 else None,  # Add label only once per model
            marker=markers[i],
            color=colors[j],
            s=100,  # Marker size
            edgecolor="black"  # Add border for better visibility
        )

# Add labels, legend, and grid
plt.xlabel("Latency (s)", fontsize=12)
plt.ylabel("Perplexity", fontsize=12)
plt.ylim(0, 2200)  # Increase the y-axis range
plt.grid(True)

# Create custom legend for models (shapes with white background)
model_legend = [
    Line2D([0], [0], marker="^", color="white", markerfacecolor="black", markersize=10, label="Sliding Window"),
    Line2D([0], [0], marker="s", color="white", markerfacecolor="black", markersize=10, label="StreamingLLM"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="black", markersize=10, label="KV-Align"),
]

# Create custom legend for cache sizes (colors)
cache_legend = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor="blue", markersize=10, label="Cache 32"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="orange", markersize=10, label="Cache 64"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="green", markersize=10, label="Cache 128"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="red", markersize=10, label="Cache 256"),
]

# Add legends
combined_legend = model_legend + cache_legend
plt.legend(
    handles=combined_legend,
    loc="upper right",
    title="Models & Cache Sizes",
    fontsize=10,
)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("perplexity_vs_latency_xy_gpt2.png", dpi=300, bbox_inches="tight")
plt.show()
