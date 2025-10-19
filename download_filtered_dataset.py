import argparse
import json
import multiprocessing as mp
import os
from multiprocessing import Pool
from collections import Counter

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm

import wandb


def parse_arguments():
    """Parse command line arguments.

    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Filter a dataset based on filter results from another dataset.")

    # Dataset paths
    parser.add_argument(
        "--filter-results-path",
        type=str,
        # default="EleutherAI/dclm-dedup_20250227-004105-filters-only",
        help="Path to the filter results dataset.",
    )
    parser.add_argument(
        "--base-dataset-path",
        type=str,
        # default="(EleutherAI/filtering-pretraining-mix)",
        help="Path to the base dataset that needs to be filtered.",
    )
    parser.add_argument("--insert-dataset-path", type=str, help="Path to the dataset that needs to be inserted into the dataset after filtering.")
    parser.add_argument("--replace-with-retained-docs", action="store_true", help="If set, replace the filtered records with the retained records approved by the decision filter.")
    parser.add_argument("--decision-filter", type=str, default="combined_filter", help="Filter to use for filtering the base dataset.")
    parser.add_argument(
        "--bert-threshold",
        type=float,
        default=None,
        help="Threshold for the BERT score. If set, overwrite the BERT decision using this new threshold. The decision_filter must be combined_filter or bert_filter.",
    )

    # Output options
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the filtered dataset.")
    parser.add_argument("--output-filename", type=str, default="retained_dataset.jsonl", help="Filename for the output jsonl file.")

    # Performance options
    parser.add_argument("--num-proc", type=int, default=100, help="Number of processes for dataset operations.")
    parser.add_argument("--num-samples", type=int, default=None, help="If set, only take the first N samples from each dataset. Useful for testing.")

    # WandB options
    parser.add_argument("--use-wandb", action="store_true", help="Enable logging with Weights & Biases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # Additional output options
    parser.add_argument("--save-filtered-documents", action="store_true", 
                        help="Save filtered-out documents to a separate filtered_documents.jsonl file.")
    parser.add_argument("--disable-retained-dataset", action="store_true", 
                        help="Disable saving the retained dataset.")

    args = parser.parse_args()

    # Create a unique experiment ID
    base_dataset_name = args.base_dataset_path.split("/")[-1]
    args.experiment_id = f"filter_dataset_{base_dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Experiment ID: {args.experiment_id}")
    return args

def load_arrow_file(arrow_file_path):
    return Dataset.from_file(arrow_file_path)

def load_local_dataset(dataset_path):
    """Load a dataset from a local path.

    Args:
        dataset_path: Path to the dataset.

    Returns:
        Dataset: The loaded dataset.
    """
    arrow_files = [file for file in os.listdir(dataset_path) if file.endswith(".arrow")]
    arrow_files_paths = [os.path.join(dataset_path, file) for file in arrow_files]
    print(f"Loading {len(arrow_files_paths)} arrow files from {dataset_path}")

    num_processes = mp.cpu_count()
    with Pool(processes=num_processes) as pool:
        pretraining_dataset_splits = list(tqdm(pool.imap(load_arrow_file, arrow_files_paths), total=len(arrow_files_paths), desc="Loading arrow files"))

    print(f"Concatenating {len(pretraining_dataset_splits)} arrow files")
    return concatenate_datasets(pretraining_dataset_splits)


def load_huggingface_dataset(dataset_path, split="train", num_proc=100):
    """Load a dataset from a local path or a Hugging Face dataset.

    Args:
        dataset_path: Path to the dataset.
        split: Split to load.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        Dataset: The loaded dataset.
    """
    if os.path.exists(dataset_path):
        print(f"Loading dataset from local dataset at {dataset_path}")
        return load_local_dataset(dataset_path)

    print(f"Loading dataset from Hugging Face dataset at {dataset_path}")
    return load_dataset(dataset_path, split=split, num_proc=num_proc)


def main():
    """Filter a dataset based on filter results and optionally insert additional data.

    Loads filter results and base dataset, filters the base dataset to retain only non-filtered records,
    optionally inserts additional data, and saves the resulting dataset. Logs statistics and results
    to Weights & Biases if enabled.
    """
    try:
        # Parse arguments
        args = parse_arguments()

        # Initialize wandb
        wandb.init(
            project="Preventing Dangerous Capabilities with Pre-Training Data Filtering",
            id=args.experiment_id,
            config=vars(args),
            mode="online" if args.use_wandb else "disabled",
        )

        print(f"Experiment ID: {args.experiment_id}")

        print(f"Loading base dataset from {args.base_dataset_path}")
        base_dataset_split = f"train[:{args.num_samples}]" if args.num_samples is not None else "train"
        base_dataset = load_huggingface_dataset(args.base_dataset_path, split=base_dataset_split, num_proc=args.num_proc)
        assert "text" in base_dataset.features

        print(f"Loading filter results from {args.filter_results_path}")
        filter_results_dataset = load_huggingface_dataset(args.filter_results_path, split="train", num_proc=args.num_proc)
        if args.bert_threshold is not None and args.decision_filter == "combined_filter":
            print(f"Overwriting BERT decision using threshold {args.bert_threshold}")
            filter_results_dataset = filter_results_dataset.map(lambda x: {"bert_filter": x["bert_filter_metadata"]["highest_score"] >= args.bert_threshold}, num_proc=args.num_proc)

        # Check that the filter results dataset is a subset or equal to the base dataset
        if args.num_samples is None:
            if len(filter_results_dataset) == len(base_dataset):
                print(f"Filter results dataset has the same length as base dataset: {len(base_dataset)}")
            elif len(filter_results_dataset) < len(base_dataset):
                print(f"Filter results dataset is a subset of base dataset:")
                print(f"  - Base dataset size: {len(base_dataset)}")
                print(f"  - Filter results size: {len(filter_results_dataset)}")
                print(f"  - Coverage: {100 * len(filter_results_dataset) / len(base_dataset):.2f}%")
                print(f"Documents not in filter results will be retained by default")
            else:
                raise ValueError(f"Filter results dataset is larger than base dataset: {len(filter_results_dataset)} > {len(base_dataset)}")

        # If num_samples is specified, take only the first N samples
        if args.num_samples is not None:
            print(f"Limiting to first {args.num_samples} samples of the filter results dataset")
            filter_results_dataset = filter_results_dataset.select(range(min(args.num_samples, len(filter_results_dataset))))

        # Get all IDs that have filter results
        print("Creating set of all IDs in filter results")
        all_filter_result_ids = set(filter_results_dataset["id"])
        print(f"Number of documents with filter results: {len(all_filter_result_ids)}")

        # Restrict base dataset to only documents that have filter results
        print("Restricting base dataset to documents with filter results")
        base_dataset = base_dataset.filter(lambda row: row["id"] in all_filter_result_ids, num_proc=args.num_proc, desc="Restricting to filtered documents")
        print(f"Base dataset restricted to {len(base_dataset)} documents")

        print(f"Extracting filtered records using {args.decision_filter}")
        filtered_records = filter_results_dataset.filter(lambda row: row[args.decision_filter], num_proc=args.num_proc, desc="Extracting filtered records")

        number_filtered_records = len(filtered_records)
        percent_to_filter = 100 * number_filtered_records / len(base_dataset)
        print(f"Number of filtered records: {number_filtered_records}")
        print(f"Percentage of records to filter: {percent_to_filter:.2f}%")

        print("Creating set of filtered IDs")
        filtered_ids = set(filtered_records["id"])
        print(f"Number of filtered IDs: {len(filtered_ids)}")

        # If num_samples is specified, take only the first N samples
        if args.num_samples is not None:
            print(f"Limiting to first {args.num_samples} samples of the base dataset")
            base_dataset = base_dataset.select(range(min(args.num_samples, len(base_dataset))))

        print("Filtering base dataset to retain only non-filtered records")
        retained_dataset = base_dataset.filter(lambda row: row["id"] not in filtered_ids, num_proc=args.num_proc, desc="Filtering records")
        if args.replace_with_retained_docs:
            if args.decision_filter == "word_filter":
                print("WARNING: --replace_with_retained_docs will not do much since the decision filter is word_filter")

            retained_word_filter_escelations = filter_results_dataset.filter(lambda row: row["word_filter"] and not row["bert_filter"], desc="Extracting word filter results", num_proc=args.num_proc)
            retained_word_filter_escelations_ids = set(retained_word_filter_escelations["id"])
            retained_escelated_documents = base_dataset.filter(lambda row: row["id"] in retained_word_filter_escelations_ids, desc="Filtering retained escelated docs", num_proc=args.num_proc)
            retained_dataset = concatenate_datasets([retained_dataset, retained_escelated_documents])
            print(f"Number of retained escelated documents appended to the retained dataset: {len(retained_escelated_documents)}. New retained dataset size: {len(retained_dataset)}")
            if "source" in retained_escelated_documents.features:
                print(f"Appended Document Sources: {Counter(retained_escelated_documents['source'])}")
                print(f"Counterfactual Document Sources: {Counter(retained_dataset[:len(retained_escelated_documents)]['source'])}")

        num_insert_samples = 0
        if args.insert_dataset_path is not None:
            print(f"Loading insert dataset from {args.insert_dataset_path}")
            insert_dataset = load_huggingface_dataset(args.insert_dataset_path, split="train", num_proc=args.num_proc)

            if "id" not in insert_dataset.features:
                print("Insert dataset does not have an 'id' column, creating one")
                insert_dataset = insert_dataset.map(lambda x, i: {"id": f"{args.insert_dataset_path.split('/')[-1]}-{i}"}, with_indices=True)
            if "source" not in insert_dataset.features:
                print("Insert dataset does not have a 'source' column, creating one")
                insert_dataset = insert_dataset.map(lambda x: {"source": args.insert_dataset_path.split("/")[-1]})

            keep_cols = ["id", "source", "text"]
            retained_dataset = concatenate_datasets([retained_dataset, insert_dataset.select_columns(keep_cols)]).shuffle(seed=args.seed)
            retained_dataset = retained_dataset.flatten_indices(num_proc=args.num_proc)
            num_insert_samples = len(insert_dataset)
            print(f"Number of insert samples: {num_insert_samples}")

        # Save filtered documents if requested
        if args.save_filtered_documents:
            print("Extracting filtered documents from base dataset")
            filtered_documents = base_dataset.filter(lambda row: row["id"] in filtered_ids, 
                                                   num_proc=args.num_proc, 
                                                   desc="Extracting filtered documents")
            
            filtered_docs_path = os.path.join(args.output_dir, "filtered_documents.jsonl")
            print(f"Saving {len(filtered_documents)} filtered documents to {filtered_docs_path}")
            filtered_documents.to_json(filtered_docs_path)

        # Save retained dataset if not disabled
        if not args.disable_retained_dataset:
            # Ensure the output filename has a .jsonl extension
            if not args.output_filename.endswith(".jsonl"):
                args.output_filename += ".jsonl"

            file_path = os.path.join(args.output_dir, args.output_filename)

            print(f"Saving retained dataset to {file_path}")
            retained_dataset.to_json(file_path)
        else:
            print("Skipping retained dataset saving (--disable-retained-dataset flag is set)")
            file_path = None

        original_size = len(base_dataset)
        filtered_size = len(filtered_ids)
        retained_size = len(retained_dataset)
        retention_percentage = retained_size / original_size * 100
        print(f"Original dataset size: {original_size}")
        print(f"Filtered records size: {filtered_size}")
        print(f"Retained dataset size: {retained_size}")
        print(f"Percentage retained: {retention_percentage:.2f}%")
        if num_insert_samples is not None:
            print(f"Number of insert samples: {num_insert_samples}")

        # Log results to wandb
        results_dict = {
            "original_dataset_size": original_size,
            "filtered_records_size": filtered_size,
            "retained_dataset_size": retained_size,
            "retention_percentage": retention_percentage,
            "filter_results_path": args.filter_results_path,
            "base_dataset_path": args.base_dataset_path,
            "output_file": file_path,
            "num_insert_samples": num_insert_samples,
            "saved_filtered_documents": args.save_filtered_documents,
            "saved_retained_dataset": not args.disable_retained_dataset,
        }
        
        if args.save_filtered_documents:
            results_dict["filtered_documents_file"] = os.path.join(args.output_dir, "filtered_documents.jsonl")

        # Save results to a JSON file
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)
            print(f"Saved results to {results_path}")

        # Log results to wandb summary
        for key, value in results_dict.items():
            wandb.summary[key] = value

        # Finish wandb run
        wandb.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
