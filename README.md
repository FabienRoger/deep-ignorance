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
- Multiple evaluation variants: categorized MCQA, verified cloze
- Categories include: bioweapons, virology, pandemic pathogens, etc.
- Custom safety evaluations to assess filtering effectiveness

### 4. Training Infrastructure
- **Dockerfiles**: For filtering (`Dockerfile.filtering`), training (`Dockerfile.training`), and evaluation (`Dockerfile.evals`) environments
- **GPT-NeoX Configs**: For pretraining and annealing phases (`pretraining/`)
- **Makefile**: For running model evaluations with lm-eval harness

### 5. Utility Scripts
- `count_tokens.py`: Analyzes token counts in datasets for training planning
  - Multiprocessing support for large datasets
  - Helps determine training epochs and batch sizes
- Additional analysis tools for dataset statistics

## Filtering Logic Flow

```
1. All documents → Blocklist Filter
   ├─ No keywords → Keep document
   └─ Has keywords (2+) → Send to ModernBERT Classifier
       │
       └─ ModernBERT Scoring
           ├─ Score ≥ threshold → Filter out document
           └─ Score < threshold → Keep document (escalated)
```

## Key Results

Our filtering approach demonstrates:
- **State-of-the-art tamper resistance**: Models resist up to 10,000 steps and 300M tokens of adversarial fine-tuning
- **Preserved general capabilities**: No degradation on benchmarks like MMLU, PIQA, LAMBADA, and HellaSwag
- **Complementary defenses**: Works synergistically with Circuit-Breaking techniques for defense-in-depth
- **Scalability**: Our filtering pipeline processed 500B+ tokens efficiently on 80 H100 GPUs

## Installation

```bash
# Python 3.11+ required
pip install -e .

# Note: PyTorch must be installed separately before other dependencies
```

## Dependencies

Key dependencies include:
- PyTorch (install separately first)
- Transformers, Accelerate, VLLM
- Datasets (HuggingFace)
- lm_eval for evaluations
- Flash Attention for efficient training
- Weights & Biases for experiment tracking
- See `pyproject.toml` for complete list

## Example Workflow

```bash
# 1. Filter a dataset
python filter.py --filtering_dataset=EleutherAI/my-dataset --splits=train

# 2. Process filtered results
python download_filtered_dataset.py \
    --filter-results-path=results/ \
    --output-dir=processed/

# 3. Count tokens for training planning
python count_tokens.py --dataset-path=processed/dataset

# 4. Run evaluations
make eval_hf MODEL=EleutherAI/camus

# Or with Docker
sudo -E make eval_hf_docker MODEL=EleutherAI/camus

# Evaluate all final models
make eval_hf_final_models
```

## Model Checkpoints

We evaluate and reference these models in our experiments:
- `EleutherAI/camus`: Baseline (unfiltered)
- `EleutherAI/stranger`: Blocklist pretraining + Blocklist annealing
- `EleutherAI/sisyphus`: Blocklist pretraining + ModernBERT annealing  
- `EleutherAI/plague`: ModernBERT pretraining + ModernBERT annealing
- `EleutherAI/absurd`: ModernBERT pretraining + Blocklist annealing

### Filtering Configurations
- **Strong Filter**: Single-stage blocklist filtering (more aggressive)
- **Weak Filter**: Multi-stage pipeline with ModernBERT review (more precise)

## Data and Model Availability

- **Code**: This repository contains all filtering, training, and evaluation code
- **Models**: Available at [huggingface.co/collections/EleutherAI/deep-ignorance-685441040d024a0fee593d68](https://huggingface.co/collections/EleutherAI/deep-ignorance-685441040d024a0fee593d68)
- **Filtered Datasets**: Not publicly released for safety reasons
- **Blocklist & Classifiers**: Available upon request for research purposes

## Requirements

- Python 3.11+
- PyTorch (install separately before other dependencies)
- CUDA-capable GPU for training and evaluation
- For filtering: 80+ GPUs recommended for web-scale datasets
- Environment variables:
  - `WANDB_API_KEY`: For experiment tracking
  - `HF_TOKEN`: For accessing HuggingFace models
  - `LM_EVAL_TASKS_PATH`: Path to evaluation tasks (optional)

## Configuration Options

Most scripts support command-line arguments. Use `--help` for details:

```bash
python filter.py --help
python download_filtered_dataset.py --help
python count_tokens.py --help
```

Key configuration parameters:
- `--bert_threshold`: Threshold for ModernBERT classifier (default: 0.5)
- `--save_every`: Checkpoint frequency for large datasets (e.g., 0.01 for 1%)
- `--use_wandb`: Enable Weights & Biases tracking
- `--log_judgments`: Log detailed filtering decisions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{obrien2025deep,
  title={Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs},
  author={O'Brien, Kyle and Casper, Stephen and Anthony, Quentin and Korbak, Tomek and Kirk, Robert and Mishra, Ishan and Gal, Yarin and Biderman, Stella},
  journal={arXiv preprint},
  year={2025}
}
```

## License

[Add appropriate license information]

## Safety Notice

This system is designed to identify and filter potentially dangerous content in training data. The filtered datasets and resulting models should be used responsibly and in accordance with applicable laws and ethical guidelines.

## Contact

For questions about the code or paper, please contact:
- Kyle O'Brien: kyledevinobrien1@gmail.com
- Stephen Casper: scasper@mit.edu

## Acknowledgments

This research was enabled by GPU donations from CoreWeave to EleutherAI and compute support from Prime Intellect and the GW4/UL Met office Isembard cluster.