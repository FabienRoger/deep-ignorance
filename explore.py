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
