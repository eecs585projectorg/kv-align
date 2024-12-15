# This plots figure 5 and 6 (readability) on paper
# use https://github.com/eecs585projectorg/cse585_project/tree/main/final_evaluations/readability_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load the data
data1 = np.load('is_readable_model_rope_mode_kv_limit_2gram_False.npy')
data2 = np.load('is_readable_model_rope_mode_sw_limit_2gram_False.npy')

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
axes[0].set_xlabel("KV-Align")
axes[0].tick_params(axis="x", labelsize=8)
axes[0].get_yaxis().set_visible(False)  # Hide the y-axis for the first subplot

# Plot for the second dataset
axes[1].imshow(
    np.array(data2).reshape(1, -1),  # Ensure data is 2D for imshow
    aspect="auto",
    cmap=cmap,  # Custom red-green colormap
    interpolation="none"  # Disable interpolation for sharp color boundaries
)
axes[1].set_xlabel("StreamingLLM")
axes[1].tick_params(axis="x", labelsize=8)
axes[1].get_yaxis().set_visible(False)  # Hide the y-axis for the second subplot

# Save and show the plot
plt.savefig("readability_RoPE.png", dpi=300)
plt.show()
