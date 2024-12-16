# Ref: ChatGPT

# This plots Figure 3 in the paper
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data for Cache Sizes
cache_sizes = [32, 64, 128, 255]
modes = ["Attention Recomputation", "StreamingLLM", "KV-Align"]
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
plt.xlabel("Latency (s)", fontsize=14)
plt.ylabel("Perplexity", fontsize=14)
plt.ylim(0, 2200)  # Increase the y-axis range
plt.grid(True)


# Create custom legend for models (shapes with white background)
model_legend = [
    Line2D([0], [0], marker="^", color="white", markerfacecolor="black", markersize=14, label="Attention Recomputation"),
    Line2D([0], [0], marker="s", color="white", markerfacecolor="black", markersize=14, label="StreamingLLM"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="black", markersize=14, label="KV-Align"),
]

# Create custom legend for cache sizes (colors)
cache_legend = [
    Line2D([0], [0], marker="o", color="white", markerfacecolor="blue", markersize=14, label="Cache 32"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="orange", markersize=14, label="Cache 64"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="green", markersize=14, label="Cache 128"),
    Line2D([0], [0], marker="o", color="white", markerfacecolor="red", markersize=14, label="Cache 255"),
]

# Add legends
combined_legend = model_legend + cache_legend
plt.legend(
    handles=combined_legend,
    loc="upper right",
    title="Models & Cache Sizes",
    fontsize=14,
)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("plots/perplexity_vs_latency_xy_gpt2.png", dpi=300, bbox_inches="tight")
plt.show()
