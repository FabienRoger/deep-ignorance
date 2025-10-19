"""Simple Fine-tuning Script with Lion Optimizer and Optional Teacher Supervision.

This script fine-tunes a language model using:
- Lion optimizer (from lion-pytorch) with constant LR and linear warmup
- Single GPU training
- Optional teacher model for knowledge distillation (temperature=1.0)
- Optional supervision on all hidden intermediate states
- Same data format as GPT-NeoX pretraining

Installation:
    pip install lion-pytorch

Example Commands:

# Basic fine-tuning without teacher
python finetune_simple.py \
    --data_path=/path/to/data_text_document \
    --student_model=EleutherAI/deep-ignorance-random-init \
    --output_dir=./checkpoints/finetuned \
    --num_steps=10000 \
    --batch_size=8 \
    --use_bf16

# Fine-tuning with teacher supervision on all hidden states
python finetune_simple.py \
    --data_path=/path/to/data_text_document \
    --teacher_model=EleutherAI/deep-ignorance-unfiltered \
    --student_model=EleutherAI/deep-ignorance-random-init \
    --hidden_supervision \
    --kd_alpha=0.5 \
    --hidden_loss_weight=0.1 \
    --output_dir=./checkpoints/distilled \
    --num_steps=50000 \
    --use_bf16 \
    --use_wandb
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List

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
from tqdm import tqdm
import wandb
from lion_pytorch import Lion


class MMapIndexedDataset(Dataset):
    """Memory-mapped dataset for efficient loading of tokenized text."""

    def __init__(self, data_path: str, seq_length: int = 2048):
        """Initialize dataset.

        Args:
            data_path: Path to the data file (without extension)
            seq_length: Sequence length for training
        """
        self.seq_length = seq_length

        # Load index file
        index_path = f"{data_path}.idx"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Read index
        with open(index_path, 'rb') as f:
            # Skip magic bytes and version
            magic = f.read(9)
            version = np.frombuffer(f.read(1), dtype=np.uint8)[0]

            # Read dtype
            dtype_code = np.frombuffer(f.read(1), dtype=np.uint8)[0]
            self.dtype = np.dtype(np.int32) if dtype_code == 3 else np.dtype(np.int64)

            # Read number of sequences
            self.num_sequences = np.frombuffer(f.read(8), dtype=np.int64)[0]

            # Read document indices
            self.doc_idx = np.frombuffer(f.read(8 * (self.num_sequences + 1)), dtype=np.int64)

        # Memory-map the data file
        data_file = f"{data_path}.bin"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.data = np.memmap(data_file, dtype=self.dtype, mode='r')

        print(f"Loaded dataset with {self.num_sequences} documents and {len(self.data)} tokens")

    def __len__(self):
        """Return approximate number of training examples."""
        # Approximate by total tokens / seq_length
        return max(1, len(self.data) // self.seq_length)

    def __getitem__(self, idx):
        """Get a training example.

        Returns:
            dict with 'input_ids' and 'labels'
        """
        # Get random starting position
        max_start = max(0, len(self.data) - self.seq_length - 1)
        start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0

        # Extract sequence
        tokens = self.data[start_idx:start_idx + self.seq_length + 1]

        # Create input and target (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0
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

    # Compute KL divergence
    kd_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

    return kd_loss


def compute_hidden_supervision_loss(
    student_hidden_states: List[torch.Tensor],
    teacher_hidden_states: List[torch.Tensor]
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


def get_linear_warmup_constant_lr(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
):
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
            project=args.wandb_project,
            name=args.wandb_run_name or os.path.basename(args.output_dir),
            config=vars(args)
        )

    # Load tokenizer from student model
    print(f"Loading tokenizer from {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        trust_remote_code=True
    )

    # Load student model
    print(f"Loading student model: {args.student_model}")
    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
        trust_remote_code=True
    ).to(device)

    # Load teacher model if specified
    teacher_model = None
    if args.teacher_model:
        print(f"Loading teacher model: {args.teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
            trust_remote_code=True
        ).to(device)
        teacher_model.eval()

        # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = MMapIndexedDataset(args.data_path, seq_length=args.seq_length)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize optimizer
    print(f"Initializing Lion optimizer with lr={args.lr}")
    optimizer = Lion(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Initialize scheduler
    num_warmup_steps = int(args.num_steps * args.warmup_ratio)
    scheduler = get_linear_warmup_constant_lr(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_steps
    )

    # Training loop
    print(f"\nStarting training for {args.num_steps} steps...")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Batch size: {args.batch_size}")
    if args.hidden_supervision:
        print("Hidden supervision enabled on all layers")

    student_model.train()
    global_step = 0

    data_iter = iter(dataloader)
    progress_bar = tqdm(total=args.num_steps, desc="Training")

    while global_step < args.num_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass - student
        student_outputs = student_model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=args.hidden_supervision,
            return_dict=True
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
                    input_ids=input_ids,
                    output_hidden_states=args.hidden_supervision,
                    return_dict=True
                )

            # Logit distillation (temperature=1.0)
            kd_loss = compute_kd_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                temperature=1.0
            )

            # Hidden state supervision
            hidden_loss = torch.tensor(0.0, device=device)
            if args.hidden_supervision:
                hidden_loss = compute_hidden_supervision_loss(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states
                )

            # Combine losses
            loss = (
                (1 - args.kd_alpha) * ntp_loss +
                args.kd_alpha * kd_loss +
                args.hidden_loss_weight * hidden_loss
            )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if args.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                student_model.parameters(),
                args.gradient_clipping
            )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging
        if global_step % args.log_interval == 0:
            metrics = {
                'loss': loss.item(),
                'ntp_loss': ntp_loss.item(),
                'lr': scheduler.get_last_lr()[0],
                'step': global_step
            }

            if teacher_model is not None and kd_loss is not None:
                metrics['kd_loss'] = kd_loss.item()
                if args.hidden_supervision and hidden_loss is not None:
                    metrics['hidden_loss'] = hidden_loss.item()

            if args.use_wandb:
                wandb.log(metrics, step=global_step)

            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['lr']:.6f}"
            })

        # Checkpointing
        if args.save_interval > 0 and (global_step + 1) % args.save_interval == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step + 1}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            print(f"\nSaving checkpoint to {checkpoint_dir}")
            student_model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            # Save training state
            torch.save({
                'step': global_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'training_state.pt'))

        global_step += 1
        progress_bar.update(1)

    # Save final model
    print(f"\nSaving final model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config
    with open(os.path.join(args.output_dir, 'training_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("Training complete!")

    if args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Simple fine-tuning with Lyon optimizer")

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to mmap dataset (without .bin/.idx extension)')
    parser.add_argument('--seq_length', type=int, default=2048,
                        help='Sequence length')

    # Model arguments
    parser.add_argument('--student_model', type=str, default='EleutherAI/deep-ignorance-random-init',
                        help='Student model to fine-tune (tokenizer will be loaded from here)')
    parser.add_argument('--teacher_model', type=str, default=None,
                        help='Optional teacher model for distillation (e.g., EleutherAI/deep-ignorance-unfiltered)')

    # Knowledge distillation arguments
    parser.add_argument('--kd_alpha', type=float, default=0.5,
                        help='Weight for KD loss vs. CE loss (0=no KD, 1=only KD)')
    parser.add_argument('--hidden_supervision', action='store_true',
                        help='Enable supervision on all hidden states')
    parser.add_argument('--hidden_loss_weight', type=float, default=0.1,
                        help='Weight for hidden state supervision loss')

    # Optimization arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (Lion typically uses lower LR than Adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay (Lion typically uses higher weight decay than Adam)')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                        help='Warmup ratio (fraction of total steps)')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='Gradient clipping value')

    # Training arguments
    parser.add_argument('--num_steps', type=int, default=10000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--use_bf16', action='store_true',
                        help='Use bfloat16 precision')

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log metrics every N steps')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='deep-ignorance-finetune',
                        help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')

    # Data loading
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("Fine-tuning Configuration")
    print("=" * 80)
    for key, value in sorted(vars(args).items()):
        print(f"{key:30s}: {value}")
    print("=" * 80)

    train(args)


if __name__ == '__main__':
    main()
