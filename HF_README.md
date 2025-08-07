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

> *Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. Currently, there is not yet a robust science of open-weight model risk management. Existing safety fine-tuning methods and other post-training techniques have struggled to make LLMs resistant to more than a few dozen steps of adversarial fine-tuning. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine- tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text – outperforming existing post-training baselines by over an order of magnitude – with no notable degradation to unrelated capabilities. However, while filtered models lack internalized dangerous knowledge, we find that they can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach. Overall, these findings help to establish pretraining data curation as a promising layer of defense for open-weight AI systems.* - O'Brien et al., 2025

The *Deep Ignorance Suite* is a collection of 6.9B models developed to facilitate research into interpretability, training data, and unlearning [(see paper)](TODO).
It contains 18 models composing of a baseline model trianed on unfiltered data, and 17 models trained on filtered datasets or with other safety  interventions being applied.

## Model Details

- Developed by: [EleutherAI](http://eleuther.ai)
- Model type: Transformer-based Language Model
- Language: English
- Learn more: [Deep Ignorance's GitHub repository](https://github.com/EleutherAI/deep-ignorance/tree/main)
 for config files, filtering logic, and details on how to use.
 [See paper](TODO) for more evals and implementation
 details.
- Library: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
- License: Apache 2.0
- Contact: to ask questions about this model, join the [EleutherAI
Discord](https://discord.gg/zBGx3azzUn), and post them in `#release-discussion`.
 Please read the existing *Deep Ignorance* documentation before asking about it in the
 EleutherAI Discord. For general correspondence: [kyledevinobrien1@gmail.com](mailto:kyledevinobrien1@gmail.com) and or Stephan Casper [scasper@mit.edu](mainto:scasper@mit.edu) .

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


## Uses and Limitations

### Intended Use

The primary intended use of Deep Ignorance is research on the behavior, functionality,
and limitations of large language models. This suite is intended to provide
a controlled setting for performing scientific experiments. We also provide
intermediate checkpoints for most models. These checkpoints are hosted on Hugging Face as branches.

### Out-of-scope use

The Deep Ignorance Suite is **not** intended for deployment. It is not a in itself
a product and cannot be used for human-facing interactions. For example,
the model may generate harmful or offensive text. Please evaluate the risks
associated with your particular use case.

Deep Ignorance models are English-language only, and are not suitable for translation
or generating text in other languages.

Deep Ignorance models have not been fine-tuned for downstream contexts in which
language models are commonly deployed, such as writing genre prose,
or commercial chatbots. This means Deep Ignorance will **not**
respond to a given prompt the way a product like ChatGPT does. This is because,
 unlike this model, ChatGPT was fine-tuned using methods such as Reinforcement
Learning from Human Feedback (RLHF) to better “follow” human instructions.

### Quickstart

Pythia models can be loaded and used via the following code, demonstrated here
for the third `pythia-70m-deduped` checkpoint:

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

Revision/branch `step143000` corresponds exactly to the model checkpoint on
the `main` branch of each model.<br>
For more information on how to use all Pythia models, see [documentation on
GitHub](https://github.com/EleutherAI/pythia).

## Training

### Training data

[The Pile](https://pile.eleuther.ai/) is a 825GiB general-purpose dataset in
English. It was created by EleutherAI specifically for training large language
models. It contains texts from 22 diverse sources, roughly broken down into
five categories: academic writing (e.g. arXiv), internet (e.g. CommonCrawl),
prose (e.g. Project Gutenberg), dialogue (e.g. YouTube subtitles), and
miscellaneous (e.g. GitHub, Enron Emails). See [the Pile
paper](https://arxiv.org/abs/2101.00027) for a breakdown of all data sources,
methodology, and a discussion of ethical implications. Consult [the
datasheet](https://arxiv.org/abs/2201.07311) for more detailed documentation
about the Pile and its component datasets. The Pile can be downloaded from
the [official website](https://pile.eleuther.ai/), or from a [community
mirror](https://the-eye.eu/public/AI/pile/).<br>
The Pile was **not** deduplicated before being used to train Pythia-12B.

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

All 16 *Pythia* models were evaluated using the [LM Evaluation
Harness](https://github.com/EleutherAI/lm-evaluation-harness). You can access
the results by model and step at `results/json/*` in the [GitHub
repository](https://github.com/EleutherAI/pythia/tree/main/results/json/).<br>
Expand the sections below to see plots of evaluation results for all
Pythia and Pythia-deduped models compared with OPT and BLOOM.

<details>
  <summary>LAMBADA – OpenAI</summary>
  <img src="/EleutherAI/pythia-12b/resolve/main/eval_plots/lambada_openai_v1.png" style="width:auto"/>
</details>

<details>
  <summary>Physical Interaction: Question Answering (PIQA)</summary>
  <img src="/EleutherAI/pythia-12b/resolve/main/eval_plots/piqa_v1.png" style="width:auto"/>
</details>

<details>
  <summary>WinoGrande</summary>
  <img src="/EleutherAI/pythia-12b/resolve/main/eval_plots/winogrande_v1.png" style="width:auto"/>
</details>

<details>
  <summary>AI2 Reasoning Challenge—Easy Set</summary>
  <img src="/EleutherAI/pythia-12b/resolve/main/eval_plots/arc_easy_v1.png" style="width:auto"/>
</details>

<details>
  <summary>SciQ</summary>
  <img src="/EleutherAI/pythia-12b/resolve/main/eval_plots/sciq_v1.png" style="width:auto"/>
</details>
