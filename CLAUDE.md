# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the "Deep Ignorance" project - a machine learning research project focused on preventing dangerous capabilities in language models through pre-training data filtering. The project specifically targets filtering out dangerous biological knowledge (WMDP - Weapons of Mass Destruction Protection) from training datasets.

## Architecture

### Core Components

1. **Data Filtering Pipeline** (`filter.py`)
   - Main entry point for filtering HuggingFace datasets
   - Implements multiple filter types: BlocklistFilter, BERT-based filters, LM filters
   - Supports distributed processing and checkpoint saving
   - Example: `python filter.py --lm_filter=Skip --log_judgments --use_wandb --save_every=0.01 --filtering_dataset=EleutherAI/filtering-annealing-mix --splits=train`

2. **Token Counting** (`count_tokens.py`)
   - Counts tokens in large datasets with multiprocessing support
   - Used for dataset budgeting and analysis

3. **Dataset Management** (`download_filtered_dataset.py`)
   - Downloads and manages filtered datasets
   - Can insert additional datasets and replace filtered records

4. **Evaluation Framework** (`lm_eval_tasks/`)
   - Contains YAML configurations for evaluating models on dangerous knowledge benchmarks
   - Main tasks: wmdp_bio_categorized_mcqa, wmdp_bio_cloze_verified
   - Categories: bioweapons, virology, pandemic pathogens, etc.
   - Variants: categorized MCQA (multiple choice by category), cloze verified (fill-in-the-blank)

5. **Training Configuration** (`pretraining/`)
   - GPT-NeoX configuration files for model pretraining and annealing
   - `pretraining_neox_config.yml`: Main pretraining config
   - `annealing_neox_config.yml`: Annealing phase config

6. **Fine-tuning Attack Testing** (`finetune_attack.py`)
   - Tests model resistance to adversarial fine-tuning
   - Includes LoRA-based fine-tuning with evaluation callbacks
   - Supports WMDP evaluation during training to monitor safety degradation
   - Example: `python finetune_attack.py --model_name=EleutherAI/deep-ignorance-e2e-strong-filter`

## Common Commands

### Running Evaluations

```bash
# Evaluate a single model on host
make eval_hf MODEL=EleutherAI/deep-ignorance-unfiltered

# Evaluate model with Docker (requires WANDB_API_KEY and HF_TOKEN)
sudo -E make eval_hf_docker MODEL=EleutherAI/deep-ignorance-unfiltered

# Evaluate all final models (host)
make eval_hf_final_models

# Evaluate all final models (Docker)
make eval_hf_docker_final_models
```

### Development Commands

```bash
# Install dependencies (Python 3.11+ required)
pip install -e .

# Run linting
ruff check .

# Format code
ruff format .

# Run tests
pytest
```

### Data Filtering

```bash
# Basic filtering with blocklist and BERT filters
python filter.py --filtering_dataset=<dataset_name> --splits=train

# Filter with all options including LM filter
python filter.py --lm_filter=LM --log_judgments --use_wandb --filtering_dataset=<dataset_name>

# Filter with intermediate checkpoints
python filter.py --save_every=0.01 --filtering_dataset=<dataset_name>
```

### Attack Testing

```bash
# Test model resistance to fine-tuning attacks
python finetune_attack.py --model_name=EleutherAI/deep-ignorance-e2e-strong-filter

# Count tokens in datasets
python count_tokens.py --dataset_path=<path> --num_workers=8
```

## Model Checkpoints

The project references these key model variants on HuggingFace:

**Core Models:**
- `EleutherAI/deep-ignorance-unfiltered`: Baseline unfiltered model
- `EleutherAI/deep-ignorance-e2e-strong-filter`: End-to-end strong filtering
- `EleutherAI/deep-ignorance-e2e-weak-filter`: End-to-end weak filtering
- `EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal`: Strong filter pretraining + weak filter annealing
- `EleutherAI/deep-ignorance-weak-filter-pt-strong-filter-anneal`: Weak filter pretraining + strong filter annealing

**Constitutional AI Variants:**
- `EleutherAI/deep-ignorance-*-cb`: Constitutional Bedside (CB) variants
- `EleutherAI/deep-ignorance-*-cb-lat`: Constitutional Bedside Latent (CB-Lat) variants

**Pretraining Stage Models:**
- `EleutherAI/deep-ignorance-pretraining-stage-unfiltered`
- `EleutherAI/deep-ignorance-pretraining-stage-strong-filter`
- `EleutherAI/deep-ignorance-pretraining-stage-weak-filter`

## Environment Variables

- `WANDB_API_KEY`: Required for Weights & Biases tracking
- `HF_TOKEN`: Required for accessing HuggingFace models
- `CUDA_VISIBLE_DEVICES`: For GPU selection
- `LM_EVAL_TASKS_PATH`: Path to evaluation tasks (defaults to local lm_eval_tasks/)

## Key Dependencies

- PyTorch (must be installed separately before other dependencies)
- Transformers, Accelerate, VLLM
- lm_eval for evaluations
- Flash Attention for efficient attention
- Weights & Biases for experiment tracking

## Docker Support

Three Docker images are provided:
- `Dockerfile.filtering`: For data filtering tasks
- `Dockerfile.training`: For model training
- `Dockerfile.evals`: For model evaluation

## Code Style

- Line length: 200 characters
- Linting: Uses ruff with D, E, F rules
- Python 3.11+ required