# Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs

This repository contains the filtering, training, and evaluation logic from our paper "Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs" by Kyle O'Brien*, Stephen Casper*, Quentin Anthony, Tomek Korbak, Robert Kirk, Ishan Mishra, Yarin Gal, and Stella Biderman (*Equal Contribution).

## Paper Abstract

Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine-tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text – outperforming existing post-training baselines by over an order of magnitude – with no observed degradation to unrelated capabilities.

## Repository Contents

This repository shares the core implementation components from our research:

### 1. Data Filtering Pipeline (`filter.py`)
The main filtering pipeline that processes HuggingFace datasets through multiple stages:

**Filtering Stages:**
1. **Blocklist Filter**: Reviews all documents for biothreat-related keywords
   - Documents without keywords pass through unfiltered
   - Documents with 2+ keywords are escalated to the next stage

2. **ModernBERT Classifier** (optional): Semantic analysis of escalated documents
   - Fine-tuned on expert-labeled examples
   - Reduces false positives from keyword matching

3. **LM Filter** (optional): Additional language model-based validation

**Key Classes:**
- `BlocklistFilter`: Checks documents against curated biothreat keywords
- `BERTFilter`: Uses fine-tuned ModernBERT for semantic content analysis
- `LMFilter`: Optional GPT-based filtering

**Usage:**
```bash
# Basic filtering with blocklist only
python filter.py --filtering_dataset=EleutherAI/dataset-name --splits=train

# Full pipeline with all filters
python filter.py --lm_filter=LM --log_judgments --use_wandb --filtering_dataset=EleutherAI/dataset-name

# With checkpoint saving for large datasets
python filter.py --save_every=0.01 --filtering_dataset=EleutherAI/dataset-name
```

### 2. Dataset Processing (`download_filtered_dataset.py`)
Processes filter results to create training datasets:

**Features:**
- Manages filtered and retained documents
- Replaces filtered documents with "escalated" ones (flagged by blocklist but approved by classifier)
- Maintains dataset size and diversity
- Supports insertion of additional datasets

**Usage:**
```bash
python download_filtered_dataset.py \
    --filter-results-path=path/to/filter-results \
    --base-dataset-path=path/to/original-dataset \
    --output-dir=/output/path
```

### 3. Evaluation Framework (`lm_eval_tasks/`)
- WMDP-Bio evaluation tasks for measuring biothreat proxy knowledge
- Custom safety evaluations to assess filtering effectiveness

**Evaluation Tasks:**

1. **WMDP-Bio Categorized MCQA** (`wmdp_bio_categorized_mcqa`)
   - Multiple-choice questions testing biothreat proxy knowledge
   - Split into two subsets to mitigate shortcut exploitation:
     - **Robust subset**: Questions resistant to multiple-choice heuristics
     - **Shortcut subset**: Questions that can be gamed using answer patterns
   - Categories: bioweapons, virology, pandemic pathogens, expanding access, reverse genetics, viral vectors

2. **WMDP-Bio Cloze Verified** (`wmdp_bio_cloze_verified`)
   - Fill-in-the-blank style evaluation (more challenging than MCQA)
   - Tests genuine knowledge without multiple-choice shortcuts
   - Uses perplexity-based scoring for answer selection

### 4. Training Infrastructure
- **Dockerfiles**: For filtering (`Dockerfile.filtering`), training (`Dockerfile.training`), and evaluation (`Dockerfile.evals`) environments
- **GPT-NeoX Configs**: For pretraining and annealing phases (`pretraining/`)
- **Makefile**: For running model evaluations with lm-eval harness

### 5. Utility Scripts
- `count_tokens.py`: Analyzes token counts in datasets for training planning
  - Multiprocessing support for large datasets
  - Helps determine training epochs and batch sizes
- Additional analysis tools for dataset statistics

## Installation

```bash
# Python 3.11+ required
pip install -e .

# Note: PyTorch must be installed separately before other dependencies
```

## Running Evaluations

```bash
# Evaluate a single model
make eval_hf MODEL=EleutherAI/camus

# Evaluate with Docker (requires WANDB_API_KEY and HF_TOKEN)
sudo -E make eval_hf_docker MODEL=EleutherAI/camus
```

## Citation

If you use this code in your research, please cite:

```bibtex
TBD
```

## Contact

For questions about the code or paper, please contact:
- Kyle O'Brien: kyledevinobrien1@gmail.com
- Stephen Casper: scasper@mit.edu