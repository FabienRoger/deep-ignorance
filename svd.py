"""Compute Singular Values of Inputs to Linear Layers via Streaming Covariance.

This script computes the singular values of inputs to each linear layer in a model
by accumulating the covariance matrix A^T @ A in a streaming fashion, then extracting
singular values as sqrt(eigenvalues).

The streaming approach avoids storing all activations in memory, making it feasible
for large datasets.

Example Commands:

# Save results with descriptive name
CUDA_VISIBLE_DEVICES=1 python svd.py \
    --data_path=filtered_output_test/retained_dataset.jsonl \
    --model=EleutherAI/deep-ignorance-unfiltered \
    --output_dir=./svd_results/unfiltered \
    --batch_size=4 \
    --use_bf16 \
    --num_samples=5000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


class SVDHook:
    """Hook to accumulate covariance matrices for linear layer inputs."""

    def __init__(self, layer_name: str, device: torch.device):
        self.layer_name = layer_name
        self.device = device
        self.cov = None
        self.n_samples = 0

    def __call__(self, module, input, output):
        x = input[0].detach()

        if x.dim() >= 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass  # Already correct shape
        else:
            raise ValueError(f"Unexpected input dimension {x.dim()} for layer {self.layer_name}")

        d = x.shape[-1]

        # Initialize covariance matrix on first call
        if self.cov is None:
            self.cov = torch.zeros(d, d, device=self.device, dtype=torch.float32)

        # Update covariance: A^T @ A
        # Use float32 for numerical stability
        x_float = x.float()
        self.cov += x_float.T @ x_float
        self.n_samples += x.shape[0]

    def compute_singular_values(self) -> torch.Tensor:
        """Compute singular values from accumulated covariance matrix."""
        if self.cov is None or self.n_samples == 0:
            return torch.tensor([])

        # Move to CPU and clone to avoid lazy wrapper issues with repeated calls
        cov_cpu = self.cov.cpu().clone()

        # Compute eigenvalues of A^T A (returns ascending order)
        eigenvalues = torch.linalg.eigvalsh(cov_cpu)

        # Singular values are sqrt of eigenvalues (clamp to handle numerical errors)
        singular_values = torch.sqrt(torch.clamp(eigenvalues, min=0))

        # Flip to get descending order (faster than sorting)
        singular_values = torch.flip(singular_values, dims=[0])

        return singular_values

    def get_stats(self) -> Dict:
        """Get statistics about accumulated data."""
        return {
            "layer_name": self.layer_name,
            "n_samples": self.n_samples,
            "hidden_dim": self.cov.shape[0] if self.cov is not None else 0,
        }


def register_hooks(model: nn.Module, device: torch.device) -> Dict[str, SVDHook]:
    """Register forward hooks on all linear layers in the model.

    Args:
        model: The model to instrument
        device: Device to store covariance matrices on

    Returns:
        Dictionary mapping layer names to SVDHook instances
    """
    hooks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = SVDHook(name, device)
            module.register_forward_hook(hook)
            hooks[name] = hook

    print(f"Registered hooks on {len(hooks)} linear layers")
    return hooks


class JSONLDataset:
    """Iterable dataset for loading JSONL text files with lazy tokenization.

    Follows GPT-NeoX conventions:
    - Each document is tokenized on-the-fly in batches
    - Documents get BOS/EOS tokens
    - Uses a deque buffer to tokenize 100 documents at a time
    - Yields sequences until exhausted, then raises StopIteration
    """

    def __init__(
        self, jsonl_path: str, tokenizer, seq_length: int = 2048, text_field: str = "text", batch_size: int = 100
    ):
        """Initialize dataset.

        Args:
            jsonl_path: Path to the JSONL file
            tokenizer: Tokenizer to use
            seq_length: Sequence length for training
            text_field: Name of the text field in JSONL
            batch_size: Number of documents to tokenize at once
        """
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.text_field = text_field
        self.batch_size = batch_size

        # Get special token IDs
        self.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        print(f"Using BOS token ID: {self.bos_token_id}")
        print(f"Using EOS token ID: {self.eos_token_id}")

        # Load dataset using HuggingFace datasets (just the raw data, no tokenization)
        print(f"Loading JSONL dataset from {jsonl_path}")
        self.dataset = load_dataset("json", data_files=jsonl_path, split="train")
        print(f"Loaded {len(self.dataset)} documents")

        # Initialize tokenization state
        self.token_buffer = deque()  # Buffer of tokens
        self.doc_index = 0  # Current document index

        # Pre-fill buffer
        print(f"Pre-filling token buffer with {batch_size} documents...")
        self._fill_buffer()
        print(f"Token buffer initialized with {len(self.token_buffer)} tokens")

    def _fill_buffer(self):
        """Tokenize a batch of documents and add to buffer."""
        if self.doc_index >= len(self.dataset):
            return False

        # Get batch of documents
        end_idx = min(self.doc_index + self.batch_size, len(self.dataset))
        batch_docs = self.dataset[self.doc_index : end_idx]

        # Extract texts
        texts = (
            batch_docs[self.text_field]
            if isinstance(batch_docs[self.text_field], list)
            else [batch_docs[self.text_field]]
        )

        # Batch tokenize
        tokenized = self.tokenizer(texts, add_special_tokens=False)

        # Add BOS/EOS and extend buffer
        for input_ids in tokenized["input_ids"]:
            doc_tokens = [self.eos_token_id] + input_ids  # + [self.eos_token_id]
            self.token_buffer.extend(doc_tokens)

        self.doc_index = end_idx
        return True

    def __iter__(self):
        """Return iterator."""
        return self

    def __next__(self):
        """Get next training example.

        Returns:
            dict with 'input_ids' and 'labels'

        Raises:
            StopIteration when dataset is exhausted
        """
        # Fill buffer if needed
        while len(self.token_buffer) < self.seq_length:
            if not self._fill_buffer():
                # No more documents to tokenize
                if len(self.token_buffer) < self.seq_length:
                    raise StopIteration
                break

        # Extract sequence from front of buffer
        tokens = []
        for _ in range(self.seq_length):
            if len(self.token_buffer) == 0:
                raise StopIteration
            tokens.append(self.token_buffer.popleft())

        # HuggingFace models handle shifting internally, so input_ids and labels are the same
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)

        return {"input_ids": tokens_tensor, "labels": tokens_tensor}


def compute_svd(args):
    """Main function to compute SVD."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32, trust_remote_code=True
    ).to(device)

    model.eval()

    # Disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Register hooks
    hooks = register_hooks(model, device)

    # Load dataset (iterable)
    dataset = JSONLDataset(args.data_path, tokenizer, seq_length=args.seq_length, text_field=args.text_field)

    print(
        f"\nProcessing up to {args.num_samples} samples..." if args.num_samples > 0 else "\nProcessing all samples..."
    )
    print(f"Batch size: {args.batch_size}")

    # Calculate expected number of batches for progress bar
    if args.num_samples > 0:
        expected_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    else:
        # Estimate based on dataset size if available
        expected_batches = None

    # Generate checkpoint batch indices using powers of 2 with half-steps
    # [1, 2, 2, 4, 4, 8, 8, 16, 16, 32, ...] -> [1, 2, 4, 8, 16, 32, ...]
    max_checkpoint = expected_batches if expected_batches else 100000
    checkpoint_batches = set()
    i = 0
    while True:
        checkpoint_batch = int(2 ** (i / 2))
        if checkpoint_batch > max_checkpoint:
            break
        checkpoint_batches.add(checkpoint_batch)
        i += 1

    checkpoint_batches = sorted(checkpoint_batches)
    print(f"Will save checkpoints at batches: {checkpoint_batches[:20]}{'...' if len(checkpoint_batches) > 20 else ''}")

    # Process data
    progress_bar = tqdm(total=expected_batches, desc="Processing batches")

    with torch.no_grad():
        sample_idx = 0
        batch_idx = 0

        while True:
            # Collect batch
            batch_input_ids = []

            for _ in range(args.batch_size):
                try:
                    example = next(dataset)
                    batch_input_ids.append(example["input_ids"])
                    sample_idx += 1

                    # Check if we've hit the sample limit
                    if args.num_samples > 0 and sample_idx >= args.num_samples:
                        break
                except StopIteration:
                    break

            # If no examples collected, we're done
            if len(batch_input_ids) == 0:
                break

            # Stack into batch and move to device
            input_ids = torch.stack(batch_input_ids).to(device)

            # Forward pass (hooks will accumulate covariance)
            _ = model(input_ids=input_ids)

            batch_idx += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"samples": sample_idx, "batch": batch_idx})

            # Save checkpoint at specific batch indices (powers of 2 with half-steps)
            if batch_idx in checkpoint_batches:
                progress_bar.write(f"Saving checkpoint at batch {batch_idx}...")
                checkpoint_file = os.path.join(args.output_dir, f"checkpoint_batch_{batch_idx}.json")
                os.makedirs(args.output_dir, exist_ok=True)

                # Compute singular values for checkpoint in parallel using threads
                def compute_layer_svd(name_hook_pair):
                    name, hook = name_hook_pair
                    sv = hook.compute_singular_values()
                    return name, {
                        "singular_values": sv.cpu().numpy().tolist(),
                        "n_samples": hook.n_samples,
                        "hidden_dim": hook.cov.shape[0] if hook.cov is not None else 0,
                    }

                checkpoint_results = {}
                with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = {executor.submit(compute_layer_svd, item): item[0] for item in hooks.items()}
                    svd_pbar = tqdm(total=len(futures), desc="Computing SVD", leave=False)
                    for future in as_completed(futures):
                        name, result = future.result()
                        checkpoint_results[name] = result
                        svd_pbar.update(1)
                    svd_pbar.close()

                checkpoint = {
                    "batch_idx": batch_idx,
                    "samples_processed": sample_idx,
                    "results": checkpoint_results,
                }

                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint, f, indent=2)
                progress_bar.write(f"Checkpoint saved to {checkpoint_file}")

            # Check if we've hit the sample limit
            if args.num_samples > 0 and sample_idx >= args.num_samples:
                break

    progress_bar.close()

    print("\nComputing final singular values...")

    # Compute singular values for each layer in parallel
    def compute_final_svd(name_hook_pair):
        layer_name, hook = name_hook_pair
        singular_values = hook.compute_singular_values()
        stats = hook.get_stats()
        return layer_name, {
            "singular_values": singular_values.cpu().numpy().tolist(),
            "n_samples": stats["n_samples"],
            "hidden_dim": stats["hidden_dim"],
        }

    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(compute_final_svd, item): item[0] for item in hooks.items()}
        svd_pbar = tqdm(total=len(futures), desc="Computing SVD")
        for future in as_completed(futures):
            layer_name, result = future.result()
            results[layer_name] = result
            svd_pbar.update(1)
        svd_pbar.close()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "singular_values.json")

    print(f"\nSaving results to {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save metadata
    metadata = {
        "model": args.model,
        "data_path": args.data_path,
        "num_samples": sample_idx,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "num_layers": len(results),
    }

    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSummary:")
    print(f"  Total layers analyzed: {len(results)}")
    print(f"  Total samples processed: {sample_idx}")
    print(f"  Results saved to: {output_file}")
    print(f"  Metadata saved to: {metadata_file}")

    # Print sample statistics
    print("\nSample statistics (first 3 layers):")
    for i, (layer_name, data) in enumerate(list(results.items())[:3]):
        sv = data["singular_values"]
        print(f"  {layer_name}:")
        print(f"    Hidden dim: {data['hidden_dim']}")
        print(f"    N samples: {data['n_samples']}")
        print(f"    Top 5 singular values: {sv[:5]}")
        print(f"    Condition number: {sv[0] / sv[-1] if len(sv) > 0 and sv[-1] > 0 else float('inf')}")


def main():
    parser = argparse.ArgumentParser(description="Compute singular values of linear layer inputs")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--text_field", type=str, default="text", help="Field name for text in JSONL")
    parser.add_argument("--seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to process (0=all)")

    # Model arguments
    parser.add_argument(
        "--model", type=str, required=True, help="Model to analyze (e.g., EleutherAI/deep-ignorance-e2e-strong-filter)"
    )
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")

    # Processing arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")

    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("SVD Computation Configuration")
    print("=" * 80)
    for key, value in sorted(vars(args).items()):
        print(f"{key:30s}: {value}")
    print("=" * 80)

    compute_svd(args)
    print("\nDone!")


if __name__ == "__main__":
    main()
