---
language:
- en
tags:
- pytorch
- causal-lm
- pythia
license: apache-2.0
datasets:
- EleutherAI/deep-ignorance-pretraining-mix
- EleutherAI/deep-ignorance-annealing-mix
---

# Deep Ignorance Model Suite

We explore an intuitive yet understudied question: Can we prevent LLMs from learning unsafe technical capabilities (such as CBRN) by filtering out enough of the relevant pretraining data before we begin training a model? Research into this question resulted in the **Deep Ignorance Suite**. In our experimental setup, we find that filtering pretraining data prevents undesirable knowledge, doesn't sacrifice general performance, and results in models that are resistant to tampering.

Deep Ignorance is a collection of 6.9B models developed to facilitate research into pretraining, interpretability, training data, and unlearning [(see paper)](TODO).
It contains 18 models composing of a baseline model trained on unfiltered data, and 17 models trained on filtered datasets or with other safety interventions being applied.

## Uses and Limitations

### Quickstart

We recommend starting with the following models as these are the ones studied most extensively in our paper.

| Model | Pretraining Filtering | Annealing Filtering | Post-training |
|:------|:---------------------|:-------------------|:--------------|
| [deep-ignorance-unfiltered](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered) | - | - | - |
| [deep-ignorance-strong-filter-pt-weak-filter-anneal](https://huggingface.co/EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal) | Strong Filter | Weak Filter | - |
| [deep-ignorance-e2e-strong-filter](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter) | Strong Filter | Strong Filter | - |
| [deep-ignorance-unfiltered-cb-lat](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered-cb-lat) | - | - | Circuit Breaking + Latent Adversarial Training |

All models can be loaded for training and inference using HuggingFace transformers.

```python
from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal",
  revision="global_step11921",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal",
  revision="global_step11921",
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])
```

Revision/branch `global_step11921` corresponds exactly to the model checkpoint on the `main` branch of each model. Specifying the revision allows you to load intermediate checkpoints. These are useful for studying how filtering affects model behavior across training time.

### Full Model List

| Model | Pretraining Filtering | Annealing Filtering | Post-training |
|:------|:---------------------|:-------------------|:--------------|
| **Unfiltered Baseline Models** | | | |
| [deep-ignorance-unfiltered](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered) | - | - | - |
| [deep-ignorance-unfiltered-cb](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered-cb) | - | - | Circuit Breaking |
| [deep-ignorance-unfiltered-cb-lat](https://huggingface.co/EleutherAI/deep-ignorance-unfiltered-cb-lat) | - | - | Circuit Breaking + Latent Adversarial Training |
| **Pretraining-Stage Only Models** | | | |
| [deep-ignorance-pretraining-stage-unfiltered](https://huggingface.co/EleutherAI/deep-ignorance-pretraining-stage-unfiltered) | - | - | - |
| [deep-ignorance-pretraining-stage-extra-weak-filter](https://huggingface.co/EleutherAI/deep-ignorance-pretraining-stage-extra-weak-filter) | Extra Weak Filter | - | - |
| [deep-ignorance-pretraining-stage-weak-filter](https://huggingface.co/EleutherAI/deep-ignorance-pretraining-stage-weak-filter) | Weak Filter | - | - |
| [deep-ignorance-pretraining-stage-strong-filter](https://huggingface.co/EleutherAI/deep-ignorance-pretraining-stage-strong-filter) | Strong Filter | - | - |
| **End-to-End Filtered Models** | | | |
| [deep-ignorance-e2e-extra-weak-filter](https://huggingface.co/EleutherAI/deep-ignorance-e2e-extra-weak-filter) | Extra Weak Filter | Extra Weak Filter | - |
| [deep-ignorance-e2e-weak-filter](https://huggingface.co/EleutherAI/deep-ignorance-e2e-weak-filter) | Weak Filter | Weak Filter | - |
| [deep-ignorance-weak-filter-pt-strong-filter-anneal](https://huggingface.co/EleutherAI/deep-ignorance-weak-filter-pt-strong-filter-anneal) | Weak Filter | Strong Filter | - |
| [deep-ignorance-strong-filter-pt-weak-filter-anneal](https://huggingface.co/EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal) | Strong Filter | Weak Filter | - |
| [deep-ignorance-strong-filter-pt-weak-filter-anneal-cb](https://huggingface.co/EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb) | Strong Filter | Weak Filter | Circuit Breaking |
| [deep-ignorance-strong-filter-pt-weak-filter-anneal-cb-lat](https://huggingface.co/EleutherAI/deep-ignorance-strong-filter-pt-weak-filter-anneal-cb-lat) | Strong Filter | Weak Filter | Circuit Breaking + Latent Adversarial Training |
| [deep-ignorance-e2e-strong-filter](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter) | Strong Filter | Strong Filter | - |
| [deep-ignorance-e2e-strong-filter-cb](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter-cb) | Strong Filter | Strong Filter | Circuit Breaking |
| [deep-ignorance-e2e-strong-filter-cb-lat](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter-cb-lat) | Strong Filter | Strong Filter | Circuit Breaking + Latent Adversarial Training |
| [deep-ignorance-e2e-strong-filter-weak-knowledge-corrupted](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter-weak-knowledge-corrupted) | Strong Filter | Strong Filter | Weak Knowledge Corruption via Synthetic Document Fine-Tuning |
| [deep-ignorance-e2e-strong-filter-strong-knowledge-corrupted](https://huggingface.co/EleutherAI/deep-ignorance-e2e-strong-filter-strong-knowledge-corrupted) | Strong Filter | Strong Filter | Strong Knowledge Corruption via Synthetic Document Fine-Tuning |

### Intended Use

Deep Ignorance is primarily intended for research into the behavior, functionality, and limitations of large language models, providing a controlled setting for conducting scientific experiments, with intermediate checkpoints for most models made available as branches hosted on Hugging Face.

Deep Ignorance models have not undergone any post-training. They often fall into repetition. They do not follow user instructions. Structured benchmarks work best for evaluating them. Applying post-training to these models could be valuable future work.

### Out-of-scope use

The Deep Ignorance Suite is not intended for deployment and is not a product for human-facing interactions. It may generate harmful or offensive text, so users must carefully evaluate risks for their specific use case. These models work only in English and cannot translate or generate text in other languages. They have not been fine-tuned for common uses like writing prose or powering commercial chatbots. Unlike ChatGPT, Deep Ignorance will not respond to prompts as expected because it lacks fine-tuning through methods like Reinforcement Learning from Human Feedback (RLHF).

## Training

All of our models undergo identical pretraining and annealing setups except for some data being removed by filters. All other hyperparameters are identical. This allows practitioners to make causal claims about data filtering's impact on training dynamics and behavior. Models trained on filtered datasets are trained for a little more than one epoch until they reach 550B training tokens in total.

### Training data

**[Pretraining](https://huggingface.co/datasets/EleutherAI/deep-ignorance-pretraining-mix)**: We utilize a deduplicated version of DCLM provided by ZyphraAI as our pretraining dataset. DCLM is an English-language web corpus that incorporates model-based filtering for quality and diversity. It has demonstrated success in training high-performing open-source language models. Our implementation uses approximately 500B tokens with the GPT-NeoX tokenizer, encompassing 409,935,485 documents.

**[Annealing/Midtraining](https://huggingface.co/datasets/EleutherAI/deep-ignorance-annealing-mix)**: Following pretraining, we perform an annealing phase with an additional 50B high-quality tokens. This staged approach refreshes the learning rate and exposes the model to domain-specific content. Our annealing mixture allocates 25B tokens (50%) to previously unseen DCLM data and 25B tokens to specialized content. The domain-specific portion emphasizes scientific and instructional data, including Flan (16.87%), StackExchange (2.82%), Pes2o (22.90%), Wikipedia (7.37%), and small amounts of Camel Bio, Chemistry, and Physics datasets (0.02% each). This composition targets improvements in knowledge benchmarks while maintaining broad capabilities.

### Training procedure

All models were trained on the exact same data, in the exact same order. Each
model saw 299,892,736,000 tokens during training, and 143 checkpoints for each
model are saved every 2,097,152,000 tokens, spaced evenly throughout training,
from `step1000` to `step143000` (which is the same as `main`). In addition, we
also provide frequent early checkpoints: `step0` and `step{1,2,4...512}`.
This corresponds to training for just under 1 epoch on the Pile for
non-deduplicated models, and about 1.5 epochs on the deduplicated Pile.

All *Pythia* models trained for 143000 steps at a batch size
of 2M (2,097,152 tokens).<br>
See [GitHub](https://github.com/EleutherAI/pythia) for more details on training
 procedure, including [how to reproduce
 it](https://github.com/EleutherAI/pythia/blob/main/README.md#reproducing-training).<br>
Pythia uses the same tokenizer as [GPT-NeoX-
20B](https://huggingface.co/EleutherAI/gpt-neox-20b).

## Evaluations

We evaluate our models across two primary dimensions: (1) retention of general capabilities and (2) reduction of biothreat proxy knowledge. This dual evaluation approach ensures that our filtering techniques effectively remove unwanted knowledge while preserving beneficial capabilities.

### Biothreat Proxy Knowledge Benchmarks
We assess biothreat-related knowledge using the WMDP-Bio benchmark, focusing on two robust evaluation formats designed to minimize shortcut exploitation:

**WMDP-Bio Robust MCQA (868 Questions)**: A curated subset of the original WMDP-Bio benchmark that excludes questions vulnerable to heuristic exploitation. We removed 405 questions (31.81%) where three different models could correctly answer based solely on the answer choices without seeing the question text. This subset provides a more reliable assessment of genuine biothreat proxy knowledge.

**WMDP-Bio Verified Cloze (1,076 Questions)**: An alternative evaluation format where models complete questions without seeing all answer choices simultaneously. We evaluate the length-normalized log probability of each answer separately, preventing models from using comparative heuristics between choices. Questions incompatible with cloze-style evaluation (e.g., "All of the above" or "Which of the following is most...") are excluded.

### General Capability Benchmarks

To ensure our filtering approach preserves beneficial knowledge, we evaluate on standard benchmarks:

- **MMLU-No-Bio**: 53 topics from MMLU excluding biology-related subjects, measuring broad knowledge retention
- **MMLU-Bio**: High school and college biology topics from MMLU, assessing benign biological knowledge
- **PIQA**: Physical commonsense reasoning tasks
- **LAMBADA**: Text comprehension requiring full-context understanding
- **HellaSwag**: Commonsense natural language inference

| Model                                                                | Pretraining Filtering   | Annealing Filtering   | WMDP Bio Average (Robust MCQA, Verified Cloze) (↓)   | Average (MMLU, PIQA, Lambada, HellaSwag) (↑)   | WMDP Bio Robust MCQA (↓)   | WMDP Bio Verified Cloze (↓)   | MMLU (↑)       | PIQA (↑)       | Lambada (↑)    | HellaSwag (↑)   |
|:---------------------------------------------------------------------|:------------------------|:----------------------|:-----------------------------------------------------|:-----------------------------------------------|:---------------------------|:------------------------------|:---------------|:---------------|:---------------|:----------------|
| deep-ignorance-unfiltered                                 | -                    | -                  | 39.66%                                        | 56.05%                                  | 42.97%              | 36.34%                 | 44.92%  | 76.44%  | 47.08%  | 55.75%   |
| deep-ignorance-pretraining-stage-unfiltered               | -                    | -                  | 37.16% (-2.50)                                       | 60.24% (4.19)                                  | 38.25% (-4.72)             | 36.06% (-0.28)                | 42.80% (-2.12) | 79.05% (2.61)  | 63.03% (15.95) | 56.06% (0.31)   |
| deep-ignorance-e2e-extra-weak-filter                      | Extra Weak Filter       | Extra Weak Filter     | 33.70% (-5.96)                                       | 55.83% (-0.22)                                 | 38.02% (-4.95)             | 29.37% (-6.97)                | 44.13% (-0.79) | 77.04% (0.60)  | 46.85% (-0.23) | 55.29% (-0.46)  |
| deep-ignorance-weak-filter-pt-strong-filter-anneal        | Weak Filter             | Strong Filter         | 30.97% (-8.69)                                       | 56.22% (0.17)                                  | 36.75% (-6.22)             | 25.19% (-11.15)               | 43.16% (-1.76) | 77.20% (0.76)  | 48.86% (1.78)  | 55.67% (-0.08)  |
| deep-ignorance-e2e-weak-filter                            | Weak Filter             | Weak Filter           | 30.50% (-9.16)                                       | 57.37% (1.32)                                  | 35.25% (-7.72)             | 25.74% (-10.60)               | 43.91% (-1.01) | 78.35% (1.91)  | 51.81% (4.73)  | 55.41% (-0.34)  |
| deep-ignorance-strong-filter-pt-weak-filter-anneal        | Strong Filter           | Weak Filter           | 30.38% (-9.28)                                       | 57.88% (1.83)                                  | 33.99% (-8.98)             | 26.77% (-9.57)                | 44.82% (-0.10) | 76.88% (0.44)  | 54.05% (6.97)  | 55.78% (0.03)   |
| deep-ignorance-e2e-strong-filter                          | Strong Filter           | Strong Filter         | 29.90% (-9.76)                                       | 55.53% (-0.52)                                 | 35.37% (-7.60)             | 24.44% (-11.90)               | 43.21% (-1.71) | 75.73% (-0.71) | 47.29% (0.21)  | 55.90% (0.15)   |
| deep-ignorance-pretraining-stage-strong-filter            | Strong Filter           | -                  | 29.47% (-10.19)                                      | 60.02% (3.97)                                  | 33.29% (-9.68)             | 25.65% (-10.69)               | 43.46% (-1.46) | 79.27% (2.83)  | 60.82% (13.74) | 56.53% (0.78)   |
| deep-ignorance-unfiltered-cb                              | -                    | -                  | 29.29% (-10.37)                                      | 54.11% (-1.94)                                 | 29.49% (-13.48)            | 29.09% (-7.25)                | 43.61% (-1.31) | 76.50% (0.06)  | 45.84% (-1.24) | 50.50% (-5.25)  |
| deep-ignorance-pretraining-stage-weak-filter              | Weak Filter             | -                  | 29.12% (-10.54)                                      | 58.98% (2.93)                                  | 33.53% (-9.44)             | 24.72% (-11.62)               | 41.04% (-3.88) | 78.78% (2.34)  | 60.57% (13.49) | 55.53% (-0.22)  |
| deep-ignorance-strong-filter-pt-weak-filter-anneal-cb-lat | Strong Filter           | Weak Filter           | 26.92% (-12.74)                                      | 58.00% (1.95)                                  | 29.95% (-13.02)            | 23.88% (-12.46)               | 43.52% (-1.40) | 76.61% (0.17)  | 56.01% (8.93)  | 55.84% (0.09)   |
| deep-ignorance-strong-filter-pt-weak-filter-anneal-cb     | Strong Filter           | Weak Filter           | 26.12% (-13.54)                                      | 56.46% (0.41)                                  | 25.46% (-17.51)            | 26.77% (-9.57)                | 41.45% (-3.47) | 76.33% (-0.11) | 53.64% (6.56)  | 54.40% (-1.35)  |
| deep-ignorance-unfiltered-cb-lat                          | -                    | -                  | 25.93% (-13.73)                                      | 56.43% (0.38)                                  | 27.42% (-15.55)            | 24.44% (-11.90)               | 42.73% (-2.19) | 76.22% (-0.22) | 51.85% (4.77)  | 54.92% (-0.83)  |
| deep-ignorance-e2e-strong-filter-cb-lat                   | Strong Filter           | Strong Filter         | 25.87% (-13.79)                                      | 56.60% (0.55)                                  | 27.76% (-15.21)            | 23.98% (-12.36)               | 42.08% (-2.84) | 75.41% (-1.03) | 52.75% (5.67)  | 56.18% (0.43)   |
| deep-ignorance-e2e-strong-filter-cb                       | Strong Filter           | Strong Filter         | 25.56% (-14.10)                                      | 52.60% (-3.45)                                 | 25.00% (-17.97)            | 26.12% (-10.22)               | 39.45% (-5.47) | 75.35% (-1.09) | 47.56% (0.48)  | 48.03% (-7.72)  |