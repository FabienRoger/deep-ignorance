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

## Common Commands

### Running Evaluations

```bash
# Evaluate a single model
make eval_hf MODEL=EleutherAI/camus

# Evaluate model with Docker (requires WANDB_API_KEY and HF_TOKEN)
sudo -E make eval_hf_docker MODEL=EleutherAI/camus

# Evaluate all final models
make eval_hf_final_models
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
# Basic filtering with Word and BERT filters
python filter.py --filtering_dataset=<dataset_name> --splits=train

# Filter with all options including LM filter
python filter.py --lm_filter=LM --log_judgments --use_wandb --filtering_dataset=<dataset_name>

# Filter with intermediate checkpoints
python filter.py --save_every=0.01 --filtering_dataset=<dataset_name>
```

## Model Checkpoints

The project references these key models:
- `EleutherAI/camus`: Baseline model
- `EleutherAI/stranger`: Blocklist pretraining + Blocklist annealing
- `EleutherAI/sisyphus`: Blocklist pretraining + ModernBERT annealing
- `EleutherAI/plague`: ModernBERT pretraining + ModernBERT annealing
- `EleutherAI/absurd`: ModernBERT pretraining + Blocklist annealing

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