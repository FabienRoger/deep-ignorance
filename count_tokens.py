import json
import os
from argparse import ArgumentParser

import multiprocess as mp
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

import wandb


def get_config():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="Zyphra/dclm-dedup")
    parser.add_argument("--subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--num_tokens", default=500000000000, type=int)
    parser.add_argument("--batch_size", default=10000, type=int)
    parser.add_argument("--num_procs", default=96, type=int)
    parser.add_argument("--use_fast", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_token_counts", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    config = parser.parse_args()

    formatted_dataset_name = config.dataset.split("/")[-1]
    config.experiment_id = f"count_tokens_{formatted_dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = f"results/formatted_datasets/{config.experiment_id}"
    config.output_dir = output_dir
    os.makedirs(config.output_dir, exist_ok=True)
    with open(f"{config.output_dir}/config.json", "w") as f:
        json.dump(vars(config), f, indent=4)

    print(f"Experiment ID: {config.experiment_id}")
    return config


def count_tokens_batched(tokenizer, examples):
    tokens = tokenizer(examples["text"], padding=False, truncation=False, return_tensors=None)
    num_tokens = [len(ids) for ids in tokens["input_ids"]]
    examples["num_tokens"] = num_tokens
    return examples


def main(config):
    wandb.init(
        project="Preventing Dangerous Capabilities with Pre-Training Data Filtering",
        id=config.experiment_id,
        config=vars(config),
        mode="online" if config.use_wandb else "disabled",
    )

    print(f"Experiment ID: {config.experiment_id}")

    print(f"Loading Tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, is_fast=config.use_fast)

    print(f"Loading Dataset: {config.dataset} — {config.subset} - {config.split}")
    full_dataset = load_dataset(config.dataset, config.subset, split=config.split, num_proc=config.num_procs)

    print("Counting Tokens Per Record")
    full_dataset = full_dataset.map(lambda x: count_tokens_batched(tokenizer, x), num_proc=config.num_procs, batched=True, batch_size=config.batch_size)

    print(f"Determing number of records which total {config.num_tokens} tokens")
    running_token_count = 0
    current_index = 0
    token_counts = full_dataset["num_tokens"]
    reached_target_token_count = False
    for index, token_count in enumerate(token_counts):
        running_token_count += token_count
        current_index = index
        reached_target_token_count = running_token_count >= config.num_tokens
        if reached_target_token_count:
            break

    if reached_target_token_count:
        print(f"The first {current_index} samples of the dataset total {running_token_count} tokens")
    else:
        print(f"The entire dataset only makes up {token_counts} tokens")

    results_dict = {
        "reached_target_token_count": reached_target_token_count,
        "final_index": current_index,
        "target_token_count": config.num_tokens,
        "acutal_token_count": running_token_count,
    }
    results_path = f"{config.output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)
        print(f"Saved results to {results_path}")

    for key, value in results_dict.items():
        wandb.summary[key] = value

    if config.save_token_counts:
        token_counts_path = f"{config.output_dir}/token_counts.parquet"
        full_dataset.select_columns(["id", "num_tokens"]).to_parquet(token_counts_path)
        print(f"Saved token counts to {token_counts_path}")



if __name__ == "__main__":
    config = get_config()
    main(config)
