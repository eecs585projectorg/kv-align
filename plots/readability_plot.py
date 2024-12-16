# Ref: ChatGPT

# This plots Figures 5 and 6 (readability) on paper, depending on wehtehr is_gpt = True or False, respectively
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

is_gpt2 = False # Change this to false for rope-based model

if is_gpt2:
    model_name = "gpt2"
    no_repeat_2gram = True
else:
    model_name = "rope"
    no_repeat_2gram = False

# Load the data
data1 = np.load(f'final_evaluations/readability_data/is_readable_model_{model_name}_mode_kv_limit_2gram_{no_repeat_2gram}.npy')
data2 = np.load(f'final_evaluations/readability_data/is_readable_model_{model_name}_mode_sw_limit_2gram_{no_repeat_2gram}.npy')

# Define a custom colormap for red (True) and green (False)
cmap = ListedColormap(["red", "green"])

# Create a figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)

# Plot for the first dataset
axes[0].imshow(
    np.array(data1).reshape(1, -1),  # Ensure data is 2D for imshow
    aspect="auto",
    cmap=cmap,  # Custom red-green colormap
    interpolation="none"  # Disable interpolation for sharp color boundaries
)
axes[0].set_xlabel("KV-Align", fontsize=14)
axes[0].tick_params(axis="x", labelsize=14)
axes[0].get_yaxis().set_visible(False)  # Hide the y-axis for the first subplot

# Plot for the second dataset
axes[1].imshow(
    np.array(data2).reshape(1, -1),  # Ensure data is 2D for imshow
    aspect="auto",
    cmap=cmap,  # Custom red-green colormap
    interpolation="none"  # Disable interpolation for sharp color boundaries
)
axes[1].set_xlabel("StreamingLLM", fontsize=14)
axes[1].tick_params(axis="x", labelsize=14)
axes[1].get_yaxis().set_visible(False)  # Hide the y-axis for the second subplot

axes[0].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='x', labelsize=14)

# Save and show the plot
plt.savefig(f"plots/readability_{model_name}.png", dpi=300)
plt.show()
