# %%
import json
from pathlib import Path

path = Path("filtered_output_test/retained_dataset.jsonl")
data = [json.loads(line) for line in path.open("r")]
# %%
print(data[1])
# %%
from matplotlib import pyplot as plt
plt.scatter(
    range(len(data)),
    [len(item['text']) for item in data],
    marker='.',
    alpha=0.1,
)
# %%
"""Explore filtered dataset results from Arrow files."""

from datasets import load_from_disk
import pandas as pd

# Load the filtered dataset
dataset_path = "results/filtering-annealing-mix_20251019-1058/filter_results/train[0%_to_0%]_rank=0"
print(f"Loading dataset from: {dataset_path}")

dataset = load_from_disk(dataset_path)

print(f"\nDataset info:")
print(f"Number of examples: {len(dataset)}")
print(f"\nColumns: {dataset.column_names}")
print(f"\nFeatures: {dataset.features}")

# Show first few examples
print(f"\n{'='*80}")
print("First 5 examples:")
print(f"{'='*80}")
for i, example in enumerate(dataset.select(range(min(5, len(dataset))))):
    print(f"\n--- Example {i} ---")
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 200:
            print(f"{key}: {value[:200]}...")
        else:
            print(f"{key}: {value}")

# Convert to pandas for easier analysis
df = dataset.to_pandas()
print(f"\n{'='*80}")
print("DataFrame info:")
print(f"{'='*80}")
print(df.info())

print(f"\n{'='*80}")
print("DataFrame head:")
print(f"{'='*80}")
print(df.head())

# If there are filter columns, show statistics
filter_cols = [col for col in df.columns if 'filter' in col.lower()]
if filter_cols:
    print(f"\n{'='*80}")
    print("Filter statistics:")
    print(f"{'='*80}")
    for col in filter_cols:
        if df[col].dtype == bool:
            print(f"\n{col}:")
            print(df[col].value_counts())
            print(f"Filtered: {df[col].sum()} / {len(df)} ({100*df[col].sum()/len(df):.2f}%)")

# %%
"""Test JSONLDataset loading and iteration."""
import sys
sys.path.insert(0, '/data2/Users/fabien/deep-ignorance')

from finetune_simple import JSONLDataset
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/deep-ignorance-random-init", trust_remote_code=True)

# Create dataset
dataset = JSONLDataset(
    jsonl_path="filtered_output_test/retained_dataset.jsonl",
    tokenizer=tokenizer,
    seq_length=2048,
    text_field="text",
    batch_size=100
)

# Test iteration
print("Testing dataset iteration...")
for i in range(1000000):
    example = next(dataset)
    print(f"\nExample {i}:")
    print(f"  input_ids shape: {example['input_ids'].shape}")
    # print(f"  labels shape: {example['labels'].shape}")
    # print(f"  First 10 tokens: {example['input_ids'][:10].tolist()}")
    # print(f"  Are input_ids and labels the same? {(example['input_ids'] == example['labels']).all()}")

print("\nDataset iteration test completed successfully!")
# %%
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("EleutherAI/deep-ignorance-random-init", trust_remote_code=True)

# %%
from transformers import AutoModelForCausalLM

modelt = AutoModelForCausalLM.from_pretrained("EleutherAI/deep-ignorance-unfiltered", trust_remote_code=True)

# %%
inputs = tokenizer("Hello, world!" * 100, return_tensors="pt")

# add bos token at the beginning
inputs['input_ids'] = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), inputs['input_ids']], dim=1)
inputs['attention_mask'] = torch.cat([torch.tensor([[1]]), inputs['attention_mask']], dim=1)

outputs = model(**inputs, output_hidden_states=True, return_dict=True)
outputst = modelt(**inputs, output_hidden_states=True, return_dict=True)
outputs.hidden_states[-1].shape
# %%
import torch.nn.functional as F

# compute mse between hs at -1 and -2

mse_loss = F.mse_loss(outputs.hidden_states[10], outputs.hidden_states[1])
print(mse_loss)
# %%
(outputs.hidden_states[10] - outputs.hidden_states[1]).square().mean()
# %%
# mse between student and teacher last hidden states
mse_loss_st = F.mse_loss(outputs.hidden_states[-1], outputst.hidden_states[-1])
print(mse_loss_st)
# %%
from typing import List
import torch

def compute_hidden_supervision_loss(
    student_hidden_states: List[torch.Tensor], teacher_hidden_states: List[torch.Tensor]
) -> torch.Tensor:
    """Compute MSE loss on all hidden states.

    Args:
        student_hidden_states: Hidden states from student (list of tensors)
        teacher_hidden_states: Hidden states from teacher (list of tensors)

    Returns:
        MSE loss averaged over all layers
    """
    losses = []
    # Supervise all layers
    l = 0
    for student_hidden, teacher_hidden in zip(student_hidden_states, teacher_hidden_states):
        loss = F.mse_loss(student_hidden, teacher_hidden)
        # print(loss)
        # # print the max magnitude of any coordinate
        # max_cor = torch.max(torch.abs(student_hidden)).item()
        # max_cor_t = torch.max(torch.abs(teacher_hidden)).item()
        # max_diff = torch.max(torch.abs(student_hidden - teacher_hidden)).item()
        # print(f"Max coord student: {max_cor:.6f}, teacher: {max_cor_t:.6f}, max diff: {max_diff:.6f}")
        # l += 1
        # # print full teacher if layer 5
        # if l == 10:
        #     b, s, d = teacher_hidden.shape
        #     for i in range(b):
        #         for j in range(s):
        #             for k in range(d):
        #                 if abs(teacher_hidden[i,j,k].item()) > 100:
        #                     print(f"teacher_hidden[{i},{j},{k}] = {teacher_hidden[i,j,k].item():.6f}")
        # print(student_hidden.shape, teacher_hidden.shape)
        # print(student_hidden[0, :5, :5])
        # print(teacher_hidden[0, :5, :5])
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=student_hidden_states[0].device)

    return torch.stack(losses).mean()

print(compute_hidden_supervision_loss(
    outputs.hidden_states,
    outputst.hidden_states
))
# %%
# compute per position
for p in range(outputs.hidden_states[-1].shape[1]):
    mse_loss_pos = compute_hidden_supervision_loss(
        [h[:, p:p+1, :] for h in outputs.hidden_states],
        [h[:, p:p+1, :] for h in outputst.hidden_states]
    )
    print(f"Position {p}: MSE loss: {mse_loss_pos.item():.6f}")
# %%
# same for KL divergence, per position
def compute_kd_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0
) -> torch.Tensor:
    """Compute knowledge distillation loss.

    Args:
        student_logits: Logits from student model [batch, seq_len, vocab]
        teacher_logits: Logits from teacher model [batch, seq_len, vocab]
        temperature: Temperature for softening distributions

    Returns:
        KL divergence loss
    """
    # Soften distributions
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence: sum over vocab (dim=-1), then mean over batch and sequence
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction="none")  # [batch, seq_len, vocab]
    kd_loss = kd_loss.sum(dim=-1)  # Sum over vocabulary -> [batch, seq_len]
    kd_loss = kd_loss.mean()  # Mean over batch and sequence
    kd_loss = kd_loss * (temperature**2)  # Scale by temperature^2

    return kd_loss

for p in range(outputs.hidden_states[-1].shape[1]):
    kd_loss_pos = compute_kd_loss(
        outputs.logits[:, p:p+1, :],
        outputst.logits[:, p:p+1, :]
    )
    print(f"Position {p}: KD loss: {kd_loss_pos.item():.6f}")
# %%