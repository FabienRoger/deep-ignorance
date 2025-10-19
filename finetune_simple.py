"""Simple Fine-tuning Script with Lion Optimizer and Optional Teacher Supervision.

This script fine-tunes a language model using:
- Lion optimizer (from lion-pytorch) with constant LR and linear warmup
- Single GPU training
- JSONL text files as input
- Optional teacher model for knowledge distillation (temperature=1.0)
- Optional supervision on all hidden intermediate states
- Periodic evaluation on WMDP benchmarks

Installation:
    pip install lion-pytorch

Example Commands:

# Basic fine-tuning from JSONL file
CUDA_VISIBLE_DEVICES=7 python finetune_simple.py \
    --data_path=filtered_output_test/retained_dataset.jsonl \
    --student_model=EleutherAI/deep-ignorance-random-init \
    --output_dir=./checkpoints/finetuned \
    --num_steps=10000 \
    --batch_size=2 \
    --use_bf16 \
    --eval_every=500 \
    --use_wandb

# Fine-tuning with teacher supervision and evaluation
python finetune_simple.py \
    --data_path=filtered_output_test/retained_dataset.jsonl \
    --teacher_model=EleutherAI/deep-ignorance-unfiltered \
    --student_model=EleutherAI/deep-ignorance-random-init \
    --hidden_supervision \
    --kd_alpha=0.5 \
    --hidden_loss_weight=0.1 \
    --output_dir=./checkpoints/distilled \
    --num_steps=50000 \
    --batch_size=2 \
    --use_bf16 \
    --eval_every=500 \
    --eval_task=wmdp_bio \
    --use_wandb
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm
import wandb
from lion_pytorch import Lion
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


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
            doc_tokens = [self.bos_token_id] + input_ids + [self.eos_token_id]
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
    for student_hidden, teacher_hidden in zip(student_hidden_states, teacher_hidden_states):
        loss = F.mse_loss(student_hidden, teacher_hidden)
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=student_hidden_states[0].device)

    return torch.stack(losses).mean()


def evaluate_model(model, task="wmdp_bio", limit=None):
    """Evaluate the model on a specific task using lm_eval.

    Args:
        model: The model to evaluate
        task: Task name (default: wmdp_bio)
        limit: Limit number of examples (default: None)

    Returns:
        dict: Evaluation results
    """
    model.eval()
    with torch.no_grad():
        hflm_model = HFLM(model)
        eval_results = evaluator.simple_evaluate(
            model=hflm_model,
            tasks=[task],
            device=model.device,
            verbosity="ERROR",
            limit=limit,
            num_fewshot=0,
        )
        del hflm_model
        try:
            acc = {task: eval_results["results"][task]["acc,none"]}
        except:
            acc = eval_results["results"]
        return acc


def get_linear_warmup_constant_lr(optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1):
    """Create a schedule with linear warmup and then constant learning rate.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps (unused, for compatibility)
        last_epoch: Last epoch number

    Returns:
        Learning rate scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def train(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name or os.path.basename(args.output_dir), config=vars(args)
        )

    # Load tokenizer from student model
    print(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    # Load student model
    print(f"Loading student model: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32, trust_remote_code=True
    ).to(device)

    # Load teacher model if specified
    teacher_model = None
    if args.teacher_model:
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model, torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32, trust_remote_code=True
        ).to(device)
        teacher_model.eval()

        # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False

    # Load dataset (iterable, no DataLoader needed)
    print(f"Loading dataset from {args.data_path}")
    dataset = JSONLDataset(args.data_path, tokenizer, seq_length=args.seq_length, text_field=args.text_field)

    # Initialize optimizer
    print(f"Initializing Lion optimizer with lr={args.lr}")
    optimizer = Lion(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initialize scheduler
    num_warmup_steps = int(args.num_steps * args.warmup_ratio)
    scheduler = get_linear_warmup_constant_lr(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=args.num_steps
    )

    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Batch size: {args.batch_size}")
    if args.hidden_supervision:
        print("Hidden supervision enabled on all layers")

    student_model.train()
    global_step = 0
    epoch = 0

    progress_bar = tqdm(total=args.num_steps, desc="Training")

    while global_step < args.num_steps:
        # Collect batch_size examples
        batch_input_ids = []
        batch_labels = []

        for _ in range(args.batch_size):
            try:
                example = next(dataset)
            except StopIteration:
                # Dataset exhausted, restart
                dataset = JSONLDataset(
                    args.data_path, tokenizer, seq_length=args.seq_length, text_field=args.text_field
                )
                example = next(dataset)
                epoch += 1
                print(f"\n--- Starting epoch {epoch} ---\n")

            batch_input_ids.append(example["input_ids"])
            batch_labels.append(example["labels"])

        # Stack into batches and move to device
        input_ids = torch.stack(batch_input_ids).to(device)
        labels = torch.stack(batch_labels).to(device)

        # Forward pass - student
        student_outputs = student_model(
            input_ids=input_ids, labels=labels, output_hidden_states=args.hidden_supervision, return_dict=True
        )

        # Compute base loss
        ntp_loss = student_outputs.loss
        loss = ntp_loss

        # Knowledge distillation if teacher is provided
        kd_loss = None
        hidden_loss = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, output_hidden_states=args.hidden_supervision, return_dict=True
                )

            # Logit distillation (temperature=1.0)
            kd_loss = compute_kd_loss(student_outputs.logits, teacher_outputs.logits, temperature=1.0)

            # Hidden state supervision
            hidden_loss = torch.tensor(0.0, device=device)
            if args.hidden_supervision:
                hidden_loss = compute_hidden_supervision_loss(
                    student_outputs.hidden_states, teacher_outputs.hidden_states
                )

            # Combine losses
            loss = (1 - args.kd_alpha) * ntp_loss + args.kd_alpha * kd_loss + args.hidden_loss_weight * hidden_loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.gradient_clipping)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        metrics = {
            "loss": loss.item(),
            "ntp_loss": ntp_loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }

        if teacher_model is not None and kd_loss is not None:
            metrics["kd_loss"] = kd_loss.item()
            if args.hidden_supervision and hidden_loss is not None:
                metrics["hidden_loss"] = hidden_loss.item()

        # Print metrics
        if global_step % 10 == 0:  # Print every 10 steps to avoid spam
            metric_str = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            print(f"Step {global_step} | {metric_str}")

        if args.use_wandb:
            wandb.log(metrics, step=global_step)

        progress_bar.set_postfix({"loss": f"{metrics['loss']:.4f}", "lr": f"{metrics['lr']:.6f}"})

        # Evaluation
        if args.eval_every > 0 and (global_step + 1) % args.eval_every == 0:
            print(f"\n[Step {global_step}] Running evaluation...")
            eval_results = evaluate_model(student_model, task=args.eval_task, limit=args.eval_limit)
            print(f"[Step {global_step}] Evaluation results: {eval_results}")

            if args.use_wandb:
                wandb.log({f"eval/{args.eval_task}": eval_results[args.eval_task]}, step=global_step)

            student_model.train()

        # Checkpointing
        if args.save_interval > 0 and (global_step + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            print(f"\nSaving checkpoint to {checkpoint_dir}")
            student_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save training state
            torch.save(
                {
                    "step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(checkpoint_dir, "training_state.pt"),
            )

        global_step += 1
        progress_bar.update(1)

    # Save final model
    print(f"\nSaving final model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print("Training complete!")

    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Simple fine-tuning with Lyon optimizer")

    # Data arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--text_field", type=str, default="text", help="Field name for text in JSONL (default: text)")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")

    # Model arguments
    parser.add_argument(
        "--student_model",
        type=str,
        default="EleutherAI/deep-ignorance-random-init",
        help="Student model to fine-tune (tokenizer will be loaded from here)",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default=None,
        help="Optional teacher model for distillation (e.g., EleutherAI/deep-ignorance-unfiltered)",
    )

    # Knowledge distillation arguments
    parser.add_argument(
        "--kd_alpha", type=float, default=0.5, help="Weight for KD loss vs. CE loss (0=no KD, 1=only KD)"
    )
    parser.add_argument("--hidden_supervision", action="store_true", help="Enable supervision on all hidden states")
    parser.add_argument(
        "--hidden_loss_weight", type=float, default=0.1, help="Weight for hidden state supervision loss"
    )

    # Optimization arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Lion typically uses lower LR than Adam)")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay (Lion typically uses higher weight decay than Adam)",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--gradient_clipping", type=float, default=1.0, help="Gradient clipping value")

    # Training arguments
    # >185_675/batchsize will multi epoch
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (number of sequences per step)")
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")

    # Logging and checkpointing
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="deep-ignorance-finetune", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=0, help="Evaluate every N steps (0=no evaluation)")
    parser.add_argument("--eval_task", type=str, default="wmdp_bio", help="Evaluation task name")
    parser.add_argument("--eval_limit", type=int, default=None, help="Limit number of eval examples (None=all)")

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("Fine-tuning Configuration")
    print("=" * 80)
    for key, value in sorted(vars(args).items()):
        print(f"{key:30s}: {value}")
    print("=" * 80)

    train(args)


if __name__ == "__main__":
    main()
