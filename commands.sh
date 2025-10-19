#!/bin/bash
# Commands for filtering and fine-tuning

# ============================================================================
# 1. FILTERING: Apply filters to dataset and save results
# ============================================================================

# Basic filtering with blocklist and BERT filters, save every 1%
python filter.py --lm_filter=skip --bert_filter=skip --log_judgments --use_wandb --save_every=0.01 --filtering_dataset=EleutherAI/filtering-annealing-mix --backend="" --splits=train

# ============================================================================
# 2. DOWNLOAD FILTERED DATASET: Create retained dataset from filter results
# ============================================================================

# Basic usage: Filter out dangerous documents and save retained dataset
python download_filtered_dataset.py \
    --filter-results-path=results/filtering-annealing-mix_20251019-1058/filter_results/train[0%_to_0%]_rank=0 \
    --base-dataset-path=EleutherAI/filtering-annealing-mix \
    --output-dir=./filtered_output \
    --output-filename=retained_dataset.jsonl \
    --decision-filter=combined_filter \
    --num-proc=32 \
    --use-wandb

# With all options: Replace filtered docs with escalated ones, save both retained and filtered
python download_filtered_dataset.py \
    --filter-results-path=results/filtering-annealing-mix_20251019-1058/filter_results/train[0%_to_0%]_rank=0 \
    --base-dataset-path=EleutherAI/filtering-annealing-mix \
    --output-dir=./filtered_output \
    --save-filtered-documents \
    --num-proc=32 \
    --use-wandb

# Test with first 1000 samples only
python download_filtered_dataset.py \
    --filter-results-path=results/filtering-annealing-mix_20251019-1058/filter_results/train[0%_to_0%]_rank=0 \
    --base-dataset-path=EleutherAI/filtering-annealing-mix \
    --output-dir=./filtered_output_test \
    --num-proc=16

# ============================================================================
# 3. FINE-TUNING: Train models with Lion optimizer
# ============================================================================

# Basic fine-tuning without teacher (requires mmap dataset)
python finetune_simple.py \
    --data_path=/path/to/data_text_document \
    --student_model=EleutherAI/deep-ignorance-random-init \
    --output_dir=./checkpoints/finetuned \
    --num_steps=10000 \
    --batch_size=8 \
    --use_bf16 \
    --use_wandb

# Fine-tuning with teacher model and hidden state supervision
python finetune_simple.py \
    --data_path=/path/to/data_text_document \
    --teacher_model=EleutherAI/deep-ignorance-unfiltered \
    --student_model=EleutherAI/deep-ignorance-random-init \
    --hidden_supervision \
    --kd_alpha=0.5 \
    --hidden_loss_weight=0.1 \
    --output_dir=./checkpoints/distilled \
    --num_steps=50000 \
    --batch_size=8 \
    --use_bf16 \
    --use_wandb \
    --save_interval=5000 \
    --log_interval=10

# ============================================================================
# 4. EVALUATION: Evaluate models on WMDP benchmarks
# ============================================================================

# Evaluate a single model
make eval_hf MODEL=EleutherAI/deep-ignorance-unfiltered

# Evaluate with Docker
sudo -E make eval_hf_docker MODEL=EleutherAI/deep-ignorance-unfiltered

# Evaluate all final models
make eval_hf_final_models
