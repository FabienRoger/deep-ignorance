"""Compute Singular Values of Inputs to Linear Layers via Streaming Covariance.

This script computes the singular values of inputs to each linear layer in a model
by accumulating the covariance matrix A^T @ A in a streaming fashion, then extracting
singular values as sqrt(eigenvalues).

The streaming approach avoids storing all activations in memory, making it feasible
for large datasets.

Example Commands:

# Compute SVD for student model on a dataset
CUDA_VISIBLE_DEVICES=0 python svd.py \
    --data_path=filtered_output_test/retained_dataset.jsonl \
    --model=EleutherAI/deep-ignorance-random-init \
    --output_dir=./svd_results \
    --num_samples=10000 \
    --batch_size=4

# Save results with descriptive name
python svd.py \
    --data_path=data.jsonl \
    --model=EleutherAI/deep-ignorance-unfiltered \
    --output_dir=./svd_results/unfiltered \
    --num_samples=50000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

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
        """Hook function called on forward pass."""
        # Get input tensor (input is a tuple, we want the first element)
        x = input[0].detach()

        # Flatten batch and sequence dimensions: [batch, seq, hidden] -> [batch*seq, hidden]
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass  # Already correct shape
        else:
            return  # Skip unexpected shapes

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

        # Compute eigenvalues of A^T A
        eigenvalues = torch.linalg.eigvalsh(self.cov)

        # Singular values are sqrt of eigenvalues (clamp to handle numerical errors)
        singular_values = torch.sqrt(torch.clamp(eigenvalues, min=0))

        # Sort in descending order
        singular_values = torch.sort(singular_values, descending=True).values

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


def load_data(data_path: str, tokenizer, seq_length: int = 2048, text_field: str = "text"):
    """Load JSONL dataset.

    Args:
        data_path: Path to JSONL file
        tokenizer: Tokenizer to use
        seq_length: Maximum sequence length
        text_field: Field name in JSONL containing text

    Returns:
        Dataset iterator
    """
    print(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Loaded {len(dataset)} documents")

    def tokenize_function(examples):
        return tokenizer(
            examples[text_field],
            truncation=True,
            max_length=seq_length,
            padding="max_length",
            return_tensors="pt"
        )

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


def compute_svd(args):
    """Main function to compute SVD."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
        trust_remote_code=True
    ).to(device)

    model.eval()

    # Disable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Register hooks
    hooks = register_hooks(model, device)

    # Load dataset
    dataset = load_data(args.data_path, tokenizer, args.seq_length, args.text_field)

    # Determine number of samples to process
    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 else len(dataset)
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size

    print(f"\nProcessing {num_samples} samples in {num_batches} batches...")
    print(f"Batch size: {args.batch_size}")

    # Process data
    with torch.no_grad():
        sample_idx = 0
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            # Get batch
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, num_samples)
            batch = dataset[batch_start:batch_end]

            # Move to device
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)

            # Forward pass (hooks will accumulate covariance)
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

            sample_idx = batch_end

            if sample_idx >= num_samples:
                break

    print("\nComputing singular values...")

    # Compute singular values for each layer
    results = {}
    for layer_name, hook in tqdm(hooks.items(), desc="Computing SVD"):
        singular_values = hook.compute_singular_values()
        stats = hook.get_stats()

        results[layer_name] = {
            "singular_values": singular_values.cpu().numpy().tolist(),
            "n_samples": stats["n_samples"],
            "hidden_dim": stats["hidden_dim"],
        }

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
        "num_samples": num_samples,
        "seq_length": args.seq_length,
        "batch_size": args.batch_size,
        "num_layers": len(results),
    }

    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSummary:")
    print(f"  Total layers analyzed: {len(results)}")
    print(f"  Total samples processed: {num_samples}")
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
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to process (0=all)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to analyze (e.g., EleutherAI/deep-ignorance-random-init)"
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
