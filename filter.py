"""Primary Data Filtering Pipeline.

This pipeline takes in a HuggingFace dataset and applies a series of filters. Filters are configured
via the command line arguments defined in get_config(). Each filter must inherit from Filter so as
to have a standard interface.

Example Commands:

Command: python filter.py --lm_filter=Skip --log_judgments --use_wandb --save_every=0.01 --filtering_dataset=EleutherAI/filtering-annealing-mix --splits=train
Description: Apply the Word and BERT filters to the annealing dataset. Don't use the LM filter. Track the filtering run in W&B. Split the dataset
into 1% chunks for granular intermediate filtering checkpoints.
"""

import argparse
from itertools import chain
import json
import multiprocessing as mp
import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import timedelta
from multiprocessing import Pool


import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, AutoModelForSequenceClassification

import wandb


class FilterResult:
    """Stores the output from a filter for a given text."""

    def __init__(self, text: str, should_filter: bool, metadata: dict):
        """Initialize a FilterResult.

        Args:
            text (str): The text that was filtered.
            should_filter (bool): Whether the text was filtered.
            metadata (dict): Metadata about the filter.
        """
        self.text = text
        self.should_filter = should_filter
        self.metadata = metadata


class Filter(ABC):
    @abstractmethod
    def should_filter(self, text: str) -> FilterResult:
        """Determine if a text should be filtered.

        Args:
            text (str): The text to filter.

        Returns:
            FilterResult: The result of the filter.
        """

    @staticmethod
    @abstractmethod
    def get_id(self) -> str:
        """Get the ID of the filter.

        Returns:
            str: The ID of the filter.
        """

    @staticmethod
    @abstractmethod
    def get_default_metadata() -> dict:
        """Get the default metadata for the filter.

        Returns:
            dict: The default metadata for the filter.
        """


class BlocklistFilter(Filter):
    """Filter that blocks words from a list."""

    def __init__(self, path: str, pos_occurence_threshold: float, word_count_threshold: int):
        """Initialize a BlocklistFilter.

        Args:
            path (str): The path to the blocklist.
            pos_occurence_threshold (float): The threshold for the positive occurrence of a word.
            word_count_threshold (int): The threshold for the number of words to filter.
        """
        super().__init__()
        self.block_words = self.__read_blocklist(path, pos_occurence_threshold)
        self.word_count_threshold = word_count_threshold

    def should_filter(self, text: str) -> FilterResult:
        """Determine if a text should be filtered.

        Args:
            text (str): The text to filter.

        Returns:
            FilterResult: The result of the filter.
        """
        occuring_words = []
        exceeded_word_threshold = False
        lowered_text = text.lower()
        for word in self.block_words:
            if word.lower() in lowered_text:
                occuring_words.append(word)
                exceeded_word_threshold = len(occuring_words) == self.word_count_threshold
                if exceeded_word_threshold:
                    break
        return FilterResult(text, exceeded_word_threshold, {"keywords": ", ".join(occuring_words)})

    @staticmethod
    def get_id() -> str:
        return "word_filter"

    @staticmethod
    def get_default_metadata() -> dict:
        return {"keywords": ""}

    def __read_blocklist(self, path: str, pos_occurence_threshold: float) -> set:
        is_local_csv = os.path.exists(path) and path.endswith(".csv")
        block_words_frame = pd.read_csv(path) if is_local_csv else load_dataset(path, split="train").to_pandas()
        before_thresholding_count = len(block_words_frame)
        if pos_occurence_threshold is not None:
            block_words_frame = block_words_frame[block_words_frame["pos_ratio"] > pos_occurence_threshold]
        if "is_relevant" in block_words_frame.columns:
            block_words_frame = block_words_frame[block_words_frame["is_relevant"]]
        after_thresholding_count = len(block_words_frame)
        print(f"Loaded threshold from {path}. Size refined from {before_thresholding_count} to {after_thresholding_count} terms")
        block_words = set(block_words_frame["keyword"])
        return block_words


class BERTFilter(Filter):
    def __init__(self, filter_path: str, threshold: float, context_length: int, device_id: int):
        super().__init__()

        # Check CUDA availability first
        assert torch.cuda.is_available(), "CUDA is not available"
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        print(f"Trying to use device_id: {device_id}")

        # Make sure the device_id is valid
        assert device_id < torch.cuda.device_count(), f"Device ID {device_id} is out of range"

        # Set the device explicitly
        cuda_device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)

        # Load model and tokenizer manually


        self.tokenizer = AutoTokenizer.from_pretrained(filter_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            filter_path,
            torch_dtype=torch.bfloat16
        )

        # Explicitly move model to CUDA and verify
        model = model.to(cuda_device)
        print(f"Model device after explicit move: {next(model.parameters()).device}")

        # Create pipeline with the already-moved model
        self.bert_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=self.tokenizer,
            device=cuda_device,
            return_all_scores=True,
            max_length=context_length,
            truncation=True,
            padding=True,
        )

        self.filter_threshold = threshold
        self.context_length = context_length

        # Verify the model's device in the pipeline
        print(f"BERT pipeline device: {self.bert_pipeline.device}")
        print(f"BERT pipeline model device: {next(self.bert_pipeline.model.parameters()).device}")

        assert next(self.bert_pipeline.model.parameters()).device.type == "cuda", "BERT model must be on GPU"

    def should_filter(self, text: str) -> FilterResult:
        input_batch = []
        input_tokens = self.tokenizer.encode(text)
        for i in range(0, len(input_tokens), self.context_length):
            decoded_text = self.tokenizer.decode(input_tokens[i : i + self.context_length], skip_special_tokens=True)
            input_batch.append(decoded_text)

        try:
            bert_should_filter = False
            batch_inference = self.bert_pipeline(input_batch, batch_size=64)
            scores = [judgment_info[-1]["score"] for judgment_info in batch_inference]
            bert_should_filter = any(score > self.filter_threshold for score in scores)
            judgment_metadata = {"highest_score": max(scores), "lowest_score": min(scores), "mean_score": sum(scores) / len(scores)}
            return FilterResult(text, bert_should_filter, judgment_metadata)
        except Exception as e:
            print(f"Error in BERT filtering: {e}")
            return FilterResult(text, True, self.get_default_metadata())

    @staticmethod
    def get_id():
        return "bert_filter"

    @staticmethod
    def get_default_metadata() -> dict:
        return {"highest_score": 0.0, "lowest_score": 0.0, "mean_score": 0.0}


class LMFilter(Filter):
    def __init__(self, model_name: str, prompt_path: str, max_new_tokens: int, device_id: int):
        super().__init__()
        self.model_name = model_name
        self.prompt = self.__get_classification_prompt(prompt_path)
        self.max_new_tokens = max_new_tokens
        self.device_id = device_id
        self.openai_client = self.__initialize_openai_client(model_name)

    def should_filter(self, text: str) -> FilterResult:
        try:
            input_text = text
            if "Now that you've seen the example" in input_text:
                input_text = input_text.split("Now that you've seen the example")[0].strip()
            messages = None
            if isinstance(self.prompt, str):
                messages = [
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": input_text},
                ]
            else:
                messages = self.prompt + [{"role": "user", "content": input_text}]
            completion = self.__get_openai_completion(messages)
            answer_component = completion.split("[")[-1]
            match = re.search(r"\d", answer_component)
            judgment = int(match.group()) == 1
            return FilterResult(text, judgment, {"judgment": judgment, "completion": completion, "error": None})
        except Exception as e:
            error_message = str(e)
            error_completion = f"Error during LM filtering: {error_message}"
            print(f"Error in LM judgment: {error_message}")
            return FilterResult(text, True, {"judgment": True, "completion": error_completion, "error": None})

    @staticmethod
    def get_id() -> str:
        return "lm_filter"

    @staticmethod
    def get_default_metadata() -> dict:
        return {"judgment": False, "completion": "", "error": None}

    def __get_classification_prompt(self, prompt_path: str):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "r") as f:
                prompt = json.load(f)
                return prompt["messages"]
        else:
            with open(prompt_path, "r") as f:
                prompt = f.read()
                return prompt

    def __initialize_openai_client(self, model_name: str):
        if model_name.startswith("gpt-"):
            return OpenAI()
        else:
            return OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="token-abc123",
            )

    def __get_openai_completion(self, messages: list) -> str:
        completion = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        generation = completion.choices[0].message.content
        return generation


def init_distributed(backend: str = ""):
    """Initialize torch distributed group using the provided backend.

    Args:
        backend (str): Backend to use ("nccl", "mpi", or empty string). If empty string, no distributed setup is performed.

    Returns:
        tuple: (global_rank, local_rank, world_size, is_distributed)
    """
    if not backend:
        if "WORLD_SIZE" in os.environ:
            raise ValueError("WORLD_SIZE is set, but no backend is specified")

        print("WARNING: No distributed setup requested")
        return 0, 0, 0, 1, False

    if backend == "nccl":
        node_rank = int(os.environ.get("NODE_RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        global_rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        print(f"Initializing NCCL distributed group with global rank {global_rank} (local rank {local_rank}) and world size {world_size}")
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=60 * 24))
        torch.cuda.set_device(local_rank)
        return node_rank, global_rank, local_rank, world_size, True

    raise ValueError(f"Invalid backend: {backend}")


def read_resume_run_config(config_id: str) -> argparse.Namespace:
    """Read the configuration of a previous run.

    Args:
        config_id (str): The ID of the previous run.

    Returns:
        argparse.Namespace: The configuration of the previous run.
    """
    previous_run_path = f"./results/{config_id}"
    run_configuration = json.load(open(f"{previous_run_path}/config.json"))
    resumed_argparse_config = argparse.Namespace()
    for config_key in run_configuration:
        setattr(resumed_argparse_config, config_key, run_configuration[config_key])
    print(f"Resuming from previous run: {previous_run_path}\n{json.dumps(run_configuration, indent=2)}")
    return resumed_argparse_config


def get_config() -> argparse.Namespace:
    """Get the configuration for the filtering pipeline.

    Returns:
        argparse.Namespace: The configuration for the filtering pipeline.
    """
    should_resume_run = any("--resume_id" in cli_arg for cli_arg in sys.argv)
    if should_resume_run:
        previous_run_id = [cli_arg.split("--resume_id=")[1] for cli_arg in sys.argv if "--resume_id=" in cli_arg][0]
        return read_resume_run_config(previous_run_id)

    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_filter", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--bert_filter", type=str, default="Unlearning/answerdotai-ModernBERT-large_20250204-111340")
    parser.add_argument("--bert_threshold", type=float, default=0.5)
    parser.add_argument("--context_length", type=int, default=8192)
    parser.add_argument("--lm_filter_prompt_path", type=str, default="prompts/classification/classify_v4_0-shot.txt")
    parser.add_argument("--block_words_list", type=str, default="Unlearning/Keyword_Generation_filters_evaluation_20241227-083238_filtered")
    parser.add_argument("--block_words_threshold", type=float, default=0.4)
    parser.add_argument("--block_words_occuring_threshold", type=int, default=2)
    parser.add_argument("--block_words_cpu_limit", type=int, default=None)
    parser.add_argument("--filtering_dataset", type=str, default="Unlearning/filters_evaluation_labeled")
    parser.add_argument("--splits", type=str, default="validation,test")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--disable_lm", action="store_true")
    parser.add_argument("--lm_max_new_tokens", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--log_judgments", action="store_true")
    parser.add_argument("--save_every", default=None, type=float)
    parser.add_argument("--backend", type=str, required=True, choices=["", "nccl"])
    parser.add_argument("--no_gather", action="store_true", help="Do not gather results from other processes. Instead, report and save results for each process separately.")
    parser.add_argument("--upload_to_hub", action="store_true", help="Upload combined filter results to HuggingFace Hub after filtering completes")
    parser.add_argument("--hf_org", type=str, default="EleutherAI", help="HuggingFace organization to upload results to")
    parser.add_argument("--hf_private", action="store_true", help="Make the uploaded dataset private on HuggingFace Hub")
    config = parser.parse_args()

    formatted_dataset_name = config.filtering_dataset.split("/")[-1]
    config.experiment_id = f"{formatted_dataset_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M')}"
    config.output_dir = f"results/{config.experiment_id}"
    os.makedirs(config.output_dir, exist_ok=True)
    with open(f"{config.output_dir}/config.json", "w") as f:
        json.dump(vars(config), f, indent=4)
    print(f"Experiment ID: {config.experiment_id}\n{json.dumps(vars(config), indent=2)}")
    return config


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


def apply_filter(prev_filter_id: str, filter: Filter, document: dict) -> dict:
    """Apply a filter to a document.

    Args:
        prev_filter_id (str): The ID of the previous filter.
        filter (Filter): The filter to apply.
        document (dict): The document to apply the filter to.

    Returns:
        dict: The document with the filter applied.
    """
    current_filter_id = filter.get_id()
    should_judge = document.get(prev_filter_id, True)
    if should_judge:
        filter_result = filter.should_filter(document["text"])
        document[current_filter_id] = filter_result.should_filter
        document[f"{current_filter_id}_metadata"] = filter_result.metadata
    else:
        document[current_filter_id] = False
        document[f"{current_filter_id}_metadata"] = filter.get_default_metadata()
    return document


def get_num_concurrent_filters(current_filter: Filter, block_words_cpu_limit: int, world_size: int) -> int:
    """Get the number of concurrent filters to run for a given filter.

    Args:
        current_filter (Filter): The current filter.
        block_words_cpu_limit (int): The number of CPUs to use for the block words filter.
        world_size (int): The number of processes in the distributed group.

    Returns:
        int: The number of concurrent filters to run.
    """
    if isinstance(current_filter, BlocklistFilter):
        num_cpus = block_words_cpu_limit or mp.cpu_count()
        per_rank_cpus = abs(num_cpus // 8)
        return per_rank_cpus
    return None


def load_filtering_dataset(config: argparse.Namespace, world_size: int, rank: int, save_every: float) -> DatasetDict:
    """Load the filtering dataset.

    Args:
        config (argparse.Namespace): Configuration for the filtering pipeline.
        world_size (int): The number of processes in the distributed group.
        rank (int): The rank of the current process.
        save_every (float): The fraction of the dataset to save as chunks.

    Returns:
        DatasetDict: The filtering dataset.
    """
    is_distributed = world_size > 1
    filtering_splits = config.splits.split(",")
    filtering_dataset = DatasetDict()
    for split in tqdm(filtering_splits, desc="Processing splits"):
        split_dataset = None
        if os.path.exists(config.filtering_dataset):
            split_dataset = load_local_dataset(config.filtering_dataset)
        else:
            split_dataset = load_dataset(config.filtering_dataset, split=split, num_proc=mp.cpu_count())

        if config.num_samples:
            full_dataset_count = len(split_dataset)
            split_dataset = split_dataset.select(range(config.num_samples))
            print(f"Limited {split} to first {config.num_samples} samples out of {full_dataset_count}")
        
        # Add id column if not present
        if "id" not in split_dataset.column_names:
            print(f"Adding id column to {split}")
            # Create incrementing IDs starting from 0
            ids = list(range(len(split_dataset)))
            split_dataset = split_dataset.add_column("id", ids)
        
        # if "is_filter_target" not in split_dataset.column_names:
        #     print(f"Adding is_filter_target column to {split}")
        #     split_dataset = split_dataset.add_column("is_filter_target", [False] * len(split_dataset))
        if save_every is not None:
            current_size = len(split_dataset)
            chunk_size = max(1, int(current_size * save_every))
            num_chunks = (current_size + chunk_size - 1) // chunk_size
            print(f"Splitting {split} ({current_size} samples) into {num_chunks} chunks of ~{chunk_size} samples each")
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, current_size)
                chunk_name = f"{split}[{int(100 * start_idx / current_size)}%:{int(100 * end_idx / current_size)}%]"
                filtering_dataset[chunk_name] = split_dataset.select(range(start_idx, end_idx))
                print(f"Created {chunk_name} with {len(filtering_dataset[chunk_name])} samples")
        else:
            filtering_dataset[split] = split_dataset
    if is_distributed:
        for split in tqdm(filtering_dataset.keys(), desc="Sharding splits for distributed processing"):
            before_count = len(filtering_dataset[split])
            filtering_dataset[split] = filtering_dataset[split].shard(num_shards=world_size, index=rank)
            print(f"Global Rank {rank} has {len(filtering_dataset[split])} out of {before_count} examples for {split}")
    return filtering_dataset


def load_filters(config):
    """Load the filters for the filtering pipeline.

    Args:
        config (argparse.Namespace): Configuration for the filtering pipeline.

    Returns:
        list[Filter]: The list of filters.
    """
    filters: list[Filter] = []
    if config.block_words_list.lower() != "skip":
        blocklist_filter = BlocklistFilter(config.block_words_list, config.block_words_threshold, config.block_words_occuring_threshold)
        filters.append(blocklist_filter)
    if config.bert_filter.lower() != "skip":
        bert_filter = BERTFilter(config.bert_filter, config.bert_threshold, config.context_length, config.local_rank)
        filters.append(bert_filter)
    if config.lm_filter.lower() != "skip":
        lm_filter = LMFilter(config.lm_filter, config.lm_filter_prompt_path, config.lm_max_new_tokens, config.local_rank)
        filters.append(lm_filter)

    return filters


def log_classification_report(y_true: list[bool], y_pred: list[bool], split: str, filter_name: str):
    """Log the classification report for a given filter.

    Args:
        y_true (list[bool]): The true labels.
        y_pred (list[bool]): The predicted labels.
        split (str): The split to log the report for.
        filter_name (str): The name of the filter.
    """
    try:
        y_true = [False] * len(y_pred) if y_true is None else y_true
        report = classification_report(y_true, y_pred, output_dict=True)
        report["filtered_documents"] = sum(y_pred)
        report["percent_filtered"] = report["filtered_documents"] / len(y_true)
        report["tn"], report["fp"], report["fn"], report["tp"] = confusion_matrix(y_true, y_pred).ravel().tolist()
        report["fpr"] = report["fp"] / (report["fp"] + report["tn"])
        file_name_safe_split = split.replace(":", "_to_")
        with open(f"results/{config.experiment_id}/{file_name_safe_split}_{filter_name}_report.json", "w") as f:
            json.dump(report, f, indent=4)

        print(f"{filter_name} report: {split}")
        print(classification_report(y_true, y_pred))
        for metric in [
            "accuracy",
            "f1-score",
            "precision",
            "recall",
            "filtered_documents",
            "percent_filtered",
            "fpr",
            "tn",
            "fp",
            "fn",
            "tp",
        ]:
            metric_value = None
            if metric in ["f1-score", "precision", "recall"]:
                metric_value = report["True"][metric]
            else:
                metric_value = report[metric]
            wandb.summary[f"{split}_{filter_name}_{metric}"] = metric_value
    except Exception as error:
        print(f"Error in logging metrics for {split} | {filter_name}: {error}")


def save_filtering_run_results(filtered_dataset: Dataset, split: str, log_judgments_sample: bool, rank: int):
    """Save the results from all the filters for the given split to local disk.

    Args:
        filtered_dataset (Dataset): The dataset to save locally
        split (str): The split being saved
        log_judgments_sample (bool): Whether a sample of judgments should be logged to W&B
        rank (int): The rank of the current process
    """
    filter_columns = [col_name for col_name in filtered_dataset.features if "_filter" in col_name]
    filter_columns = (["id"] + filter_columns) if "id" in filtered_dataset.features else filter_columns
    file_name_safe_split = split.replace(":", "_to_") + f"_rank={rank}"
    full_save_path = f"{config.output_dir}/filter_results/{file_name_safe_split}"
    filtered_dataset.select_columns(filter_columns).save_to_disk(full_save_path)
    print(f"[Rank {rank}] Saved filter results to {full_save_path}")

    if log_judgments_sample and rank == 0:
        split_dataset_frame = filtered_dataset.take(10000).to_pandas() if len(filtered_dataset) >= 10000 else filtered_dataset.to_pandas()
        split_dataset_frame.to_csv(f"results/{config.experiment_id}/{file_name_safe_split}_judgments.csv", index=False)
        split_dataset_table = wandb.Table(dataframe=split_dataset_frame)
        wandb.log({f"{file_name_safe_split}_judgments": split_dataset_table})


def combine_and_upload_filter_results(config: argparse.Namespace):
    """Combine all filter results and upload to HuggingFace Hub.
    
    Args:
        config (argparse.Namespace): Configuration containing experiment_id and output_dir
    """
    results_path = f"{config.output_dir}/filter_results"
    
    if not os.path.exists(results_path):
        print(f"Filter results directory not found: {results_path}")
        return
    
    # Get all split directories
    splits = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]
    
    if not splits:
        print("No filter result splits found")
        return
    
    # Sort splits by percentage and rank for proper ordering
    def get_split_ordering(split_name):
        # Extract percentage from split name like "train[0%_to_1%]_rank=0"
        if "[" in split_name and "%_to_" in split_name:
            start_percent = int(split_name.split("%_to_")[0].split("[")[-1])
        else:
            start_percent = 0
        # Extract rank
        if "_rank=" in split_name:
            rank = int(split_name.split("_rank=")[-1])
        else:
            rank = 0
        return start_percent, rank
    
    sorted_splits = sorted(splits, key=get_split_ordering)
    print(f"Found {len(sorted_splits)} split results to combine")
    
    # Load and combine all splits
    split_datasets = []
    for split in tqdm(sorted_splits, desc="Loading filter result splits"):
        split_path = os.path.join(results_path, split)
        try:
            split_dataset = load_from_disk(split_path)
            split_datasets.append(split_dataset)
        except Exception as e:
            print(f"Error loading split {split}: {e}")
            continue
    
    if not split_datasets:
        print("No datasets could be loaded")
        return
    
    print(f"Concatenating {len(split_datasets)} datasets...")
    combined_dataset = concatenate_datasets(split_datasets)
    print(f"Combined dataset has {len(combined_dataset)} examples")
    
    # Upload to HuggingFace Hub
    hf_path = f"{config.hf_org}/{config.experiment_id}" if hasattr(config, 'hf_org') else f"EleutherAI/{config.experiment_id}"
    print(f"Uploading to HuggingFace Hub: {hf_path}")
    
    try:
        combined_dataset.push_to_hub(
            hf_path, 
            private=False,  # Make dataset public
            commit_message=f"Filter results for experiment {config.experiment_id}"
        )
        print(f"Successfully uploaded filter results to {hf_path}")
        
        # Stream and print first 100 records
        print("\n" + "="*80)
        print("Streaming first 100 records from uploaded dataset...")
        print("="*80 + "\n")
        
        try:
            # Load dataset in streaming mode
            streaming_dataset = load_dataset(hf_path, split='train', streaming=True)
            
            # Print first 100 records
            for i, record in enumerate(streaming_dataset):
                if i >= 100:
                    break
                
                print(f"\n--- Record {i+1} ---")
                # Print key fields in a readable format
                for key, value in record.items():
                    # Truncate long text fields
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    elif isinstance(value, dict):
                        # For metadata fields, show compact representation
                        value = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    print(f"{key}: {value}")
                
            print(f"\n\nSuccessfully streamed and displayed first 100 records from {hf_path}")
            
        except Exception as e:
            print(f"Error streaming dataset: {e}")
        
        # Print final summary for easy reference
        print("\n" + "="*80)
        print("FILTER RESULTS UPLOAD COMPLETE")
        print("="*80)
        print(f"Dataset path for download_filtered_dataset.py: {hf_path}")
        print("="*80 + "\n")
            
    except Exception as e:
        print(f"Error uploading to HuggingFace Hub: {e}")


def main(config: argparse.Namespace):
    """Run the filtering pipeline.

    Args:
        config (argparse.Namespace): Configuration for the filtering pipeline.
    """
    # config.global_rank = int(os.environ.get("RANK", "0"))
    # config.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # config.world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    # is_distributed = config.world_size != -1

    config.node_rank, config.global_rank, config.local_rank, config.world_size, is_distributed = init_distributed(config.backend)
    if is_distributed:
        print(f"Running with distributed inference on {config.world_size} GPUs and backend as {config.backend}")
    else:
        print("Running without distributed inference")

    # For aggregator (logging, gathering), use the designated global rank (here, global rank 0)
    is_main_rank = config.global_rank == 0 and config.node_rank == 0
    if is_main_rank:
        print("Initializing wandb...")
        wandb.init(
            project="Preventing Dangerous Capabilities with Pre-Training Data Filtering",
            id=config.experiment_id,
            config=vars(config),
            mode="online" if config.use_wandb else "disabled",
        )

    print(f"Loading dataset {config.filtering_dataset}")
    filtering_dataset = load_filtering_dataset(config, config.world_size, config.global_rank, config.save_every)
    print("Loading filters")
    filters = load_filters(config)

    for split in filtering_dataset:
        # if is_distributed:
        #     print(f"[Rank {config.global_rank}] Waiting for all processes to catch up before starting filtering")
        #     dist.barrier()

        file_name_safe_split = split.replace(":", "_to_") + f"_rank={config.global_rank}"
        already_filtered = os.path.exists(f"{config.output_dir}/filter_results/{file_name_safe_split}")
        if already_filtered:
            print(f"Skipping already filtered split: {split} | Rank: {config.global_rank}")
            continue

        for index, filter in enumerate(filters):
            current_filter_id = filter.get_id()
            filtering_stop_log_suffix = f"Split: {split} | Filter Step: {index + 1}/{len(filters)} | Filter ID: {current_filter_id} | Rank: {config.global_rank}"
            print(f"Beginning Filtering — {filtering_stop_log_suffix}")

            prev_filter_id = None if index == 0 else filters[index - 1].get_id()
            num_concurrent_filters = get_num_concurrent_filters(filter, config.block_words_cpu_limit, config.world_size)
            filtering_dataset[split] = filtering_dataset[split].map(lambda document: apply_filter(prev_filter_id, filter, document), num_proc=num_concurrent_filters)

            distributed_filter_results = [None for _ in range(config.world_size)] if config.global_rank == 0 else None
            if is_distributed:
                select_columns_names = ["id", current_filter_id, f"{current_filter_id}_metadata"]
                if "is_filter_target" in filtering_dataset[split].column_names:
                    select_columns_names.append("is_filter_target")

                transport_formatted_filter_results = filtering_dataset[split].select_columns(select_columns_names).to_list()
                serialized_json_filter_results = json.dumps(transport_formatted_filter_results, indent=4)

                if not config.no_gather:
                    dist.gather_object(serialized_json_filter_results, distributed_filter_results, dst=0)
                else:
                    distributed_filter_results = [serialized_json_filter_results]
            if is_main_rank:
                print(f"Reporting Metrics — {filtering_stop_log_suffix}")

                logging_dataset = None
                if is_distributed:
                    deserialized_filter_results = [json.loads(result) for result in distributed_filter_results]
                    flattened_filter_results = list(chain.from_iterable(deserialized_filter_results))
                    logging_dataset = Dataset.from_list(flattened_filter_results)
                else:
                    logging_dataset = filtering_dataset[split]

                filter_judgments = logging_dataset[current_filter_id]
                filter_labels = logging_dataset["is_filter_target"] if "is_filter_target" in logging_dataset.column_names else None
                log_classification_report(filter_labels, filter_judgments, split, current_filter_id)

        distributed_postfilters_results = [None for _ in range(config.world_size)] if config.global_rank == 0 else None
        if is_distributed and not config.no_gather:
            large_data_columns = set(["text", "metadata"])
            postfilter_select_columns_names = [col_name for col_name in filtering_dataset[split].column_names if col_name not in large_data_columns]
            transport_formatted_postfilters_results = filtering_dataset[split].select_columns(postfilter_select_columns_names).to_list()
            dist.gather_object(transport_formatted_postfilters_results, distributed_postfilters_results, dst=0)
        if is_main_rank or config.no_gather:
            postfilter_logging_dataset = None
            if not is_distributed or config.no_gather:
                postfilter_logging_dataset = filtering_dataset[split]
            else:
                postfilter_logging_dataset = Dataset.from_list(list(chain.from_iterable(distributed_postfilters_results)))

            combined_filter_id = "combined_filter"
            final_filter_layer_id = filters[-1].get_id()
            postfilter_logging_dataset = postfilter_logging_dataset.add_column(combined_filter_id, postfilter_logging_dataset[final_filter_layer_id])
            filter_labels = postfilter_logging_dataset["is_filter_target"] if "is_filter_target" in postfilter_logging_dataset.column_names else None
            save_filtering_run_results(postfilter_logging_dataset, split, config.log_judgments, rank=config.global_rank)
            if is_main_rank:
                log_classification_report(filter_labels, postfilter_logging_dataset[combined_filter_id], split, combined_filter_id)

        print(f"FINISHED FILTERING — Global Rank: {config.global_rank} | Split: {split}")
    
    # After all splits are processed, combine and upload if requested
    if config.upload_to_hub and is_main_rank:
        print("\nAll filtering complete. Combining and uploading results to HuggingFace Hub...")
        combine_and_upload_filter_results(config)


if __name__ == "__main__":
    print("\nPreventing Capabilities with Pre-Training Data Filtering\n")
    config = get_config()
    main(config)
