# %%
import json
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# %%
# Configuration
checkpoint_dir = "./svd_results/unfiltered"
output_file = "./svd_plots/evolution.png"

# %%
# Load all checkpoint files
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_batch_*.json"))

if not checkpoint_files:
    raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

checkpoints = []
for file_path in checkpoint_files:
    with open(file_path, "r") as f:
        data = json.load(f)
        checkpoints.append(data)

# Sort by batch index
checkpoints.sort(key=lambda x: x["batch_idx"])

print(f"Loaded {len(checkpoints)} checkpoints")

# Get all layer names from first checkpoint

def key_func(layer):
    # gpt_neox.layers.10.mlp -> (10, 'gpt_neox.layers.mlp')
    # embed -> (-1, 'embed')
    parts = layer.split(".")
    if parts[0] == "gpt_neox" and parts[1] == "layers":
        layer_num = int(parts[2])
        rest = ".".join(parts[3:])
        return (layer_num, rest)
    else:
        return (100, layer)

def should_skip_layer(layer):
    # don't skip non-numeric layers, don't skip 0, 4, 8, ...
    parts = layer.split(".")
    if parts[0] == "gpt_neox" and parts[1] == "layers":
        layer_num = int(parts[2])
        return layer_num % 6 != 0
    return False

layer_names = sorted([k for k in checkpoints[0]["results"].keys() if not should_skip_layer(k)], key=key_func)
num_layers = len(layer_names)

print(f"Found {num_layers} layers")
print("Layer names:", layer_names)

# %%
# Calculate grid dimensions
cols = 4
rows = (num_layers + cols - 1) // cols

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
# fig.suptitle("Singular Value Evolution Across Checkpoints", fontsize=16, y=0.995)

# Flatten axes for easier indexing
if rows == 1 and cols == 1:
    axes = np.array([axes])
axes = axes.flatten() if num_layers > 1 else [axes]

# Create color gradient for checkpoints
num_checkpoints = len(checkpoints)
colors = cm.viridis(np.linspace(0, 1, num_checkpoints))[::-1]

# Plot each layer
for layer_idx, layer_name in enumerate(layer_names):
    ax = axes[layer_idx]

    # Plot singular values for each checkpoint
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        layer_data = checkpoint["results"][layer_name]
        singular_values = layer_data["singular_values"]

        # Plot
        ax.plot(
            singular_values,
            color=colors[checkpoint_idx],
            alpha=0.7,
            linewidth=1.5,
            label=f"Batch {checkpoint['batch_idx']}"
        )

    # Set labels and title
    # ax.set_xlabel("Index", fontsize=8)
    ax.set_ylabel("Singular Value", fontsize=8)
    ax.set_title(layer_name, fontsize=9, pad=3)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)
    if layer_idx == 3:
        ax.legend(fontsize=7)

# Hide unused subplots
for idx in range(num_layers, len(axes)):
    axes[idx].axis("off")

# Add colorbar to show checkpoint progression
# sm = cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_checkpoints - 1))
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", pad=0.02, aspect=50)
# cbar.set_label("Checkpoint Index", fontsize=10)

# Adjust layout
# plt.tight_layout()

# # Save figure
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
# plt.savefig(output_file, dpi=150, bbox_inches="tight")
# print(f"Plot saved to {output_file}")

# plt.show()
# %%
