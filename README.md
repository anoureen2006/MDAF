# MDAF: Multi-Dialect Arabic Fake News Detection

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/)


> **Multi-Dialect Arabic Fake News Detection via LLM-Augmented Dialectal Training**

This repository contains the full implementation of the MDAF framework, which addresses the critical vulnerability of Arabic fake news detectors to dialectal variation. We use large language models (Fanar-1-9B) to generate dialectal and adversarial augmentations, then train a robust MARBERTv2-based classifier that generalises across Arabic dialects.

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Experimental Setup](#experimental-setup)

---

## Overview

Arabic fake news detection models are typically trained on Modern Standard Arabic (MSA) and suffer significant performance degradation when encountering dialectal Arabic text — the dominant form of online communication across the Arab world. MDAF tackles this gap through:

1. **Dialectal Data Augmentation** — Using Fanar-1-9B-Instruct to paraphrase MSA articles into 7 Arabic dialects (Moroccan, Algerian, Tunisian, Egyptian, Syrian, Lebanese, Gulf).
2. **Adversarial Augmentation** — Generating stylistically diverse rewrites to improve general robustness.
3. **Robust Classification** — A MARBERTv2-based model with triple pooling (CLS + Mean + Max) and multi-sample dropout, trained on the augmented data.

## Key Findings

| Experiment | Train Data | Val Data | Accuracy | F1 Score | AUC |
|:----------:|:----------:|:--------:|:--------:|:--------:|:---:|
| **Test A** | Original | Original | 82.31% ± 0.76 | 82.20% ± 0.76 | 91.37% ± 0.77 |
| **Test B** | Original | Dialect | 65.60% ± 1.00 | 65.53% ± 1.04 | 72.08% ± 1.42 |
| **Test C** | Original + Adversarial | Dialect | 66.10% ± 0.49 | 66.03% ± 0.48 | 73.12% ± 0.77 |
| **Test D** | Original + Dialect | Dialect | **67.84% ± 1.37** | **67.70% ± 1.39** | **75.00% ± 1.40** |

- **Test B** reveals a **~17% accuracy drop** when MSA-trained models encounter dialectal text.
- **Test D** (proposed MDAF approach) recovers **+2.24%** accuracy over the no-augmentation baseline (Test B).
- Dialectal augmentation (Test D) outperforms adversarial-only augmentation (Test C) across all metrics.

---


---

## Datasets

### AFND (Arabic Fake News Dataset)
- **Source:** 135 Arabic news sources with credibility labels
- **Size:** 374,532 articles after preprocessing
- **Labels:** Binary — `1` (Credible) / `0` (Fake)
- **Format:** JSONL with fields: `id`, `source`, `label`, `title`, `text`
- **Link:** https://www.kaggle.com/datasets/murtadhayaseen/arabic-fake-news-dataset-afnd
- **Paper:**
  ```bibtex
   @article{KHALIL2022108141,
  title = {AFND: Arabic fake news dataset for the detection and classification of articles credibility},
  journal = {Data in Brief},
  volume = {42},
  pages = {108141},
  year = {2022},
  issn = {2352-3409},
  doi = {https://doi.org/10.1016/j.dib.2022.108141},
  url = {https://www.sciencedirect.com/science/article/pii/S2352340922003493},
  author = {Ashwaq Khalil and Moath Jarrah and Monther Aldwairi and Manar Jaradat},
  keywords = {Arabic news dataset, Arabic fake news, Article credibility, Weak labeling, Detection, Classification},
  abstract = {The news credibility detection task has started to gain more attention recently due to the rapid increase of news on different social media platforms. This article provides a large, labeled, and diverse Arabic Fake News Dataset (AFND) that is collected from public Arabic news websites. This dataset enables the research community to use supervised and unsupervised machine learning algorithms to classify the credibility of Arabic news articles. AFND consists of 606912 public news articles that were scraped from 134 public news websites of 19 different Arab countries over a 6-month period using Python scripts. The Arabic fact-check platform, Misbar, is used manually to classify each public news source into credible, not credible, or undecided. Weak supervision is applied to label news articles with the same label as the public source. AFND is imbalanced in the number of articles in each class. Hence, it is useful for researchers who focus on finding solutions for imbalanced datasets. The dataset is available in JSON format and can be accessed from Mendeley Data repository.}
  }

### VeraArab
- **Source:** VeraArab benchmark dataset (Excel format)
- **Size:** 20,070 samples after cleaning
- **Labels:** Binary — `1` (Real) / `0` (Fake)
- **Paper:**
  ```bibtex
   @article{10.7717/peerj-cs.2432,
   title = {VERA-ARAB: unveiling the Arabic tweets credibility by constructing balanced news dataset for veracity analysis},
   author = {Mostafa, Mohamed A. and Almogren, Ahmad},
   year = 2024,
   month = oct,
   keywords = {Social computing, Social media, Fake news, Arabic dataset, Topic classification, Named entity recognition},
   abstract = {
  The proliferation of fake news on social media platforms necessitates the development of reliable datasets for effective fake news detection and veracity analysis. In this article, we introduce a veracity dataset of Arabic tweets called “VERA-ARAB”, a pioneering large-scale dataset designed to enhance fake news detection in Arabic tweets. VERA-ARAB is a balanced, multi-domain, and multi-dialectal dataset, containing both fake and true news, meticulously verified by fact-checking experts from Misbar. Comprising approximately 20,000 tweets from 13,000 distinct users and covering 884 different claims, the dataset includes detailed information such as news text, user details, and spatiotemporal data, spanning diverse domains like sports and politics. We leveraged the X API to retrieve and structure the dataset, providing a comprehensive data dictionary to describe the raw data and conducting a thorough statistical descriptive analysis. This analysis reveals insightful patterns and distributions, visualized according to data type and nature. We also evaluated the dataset using multiple machine learning classification models, exploring various social and textual features. Our findings indicate promising results, particularly with textual features, underscoring the dataset’s potential for enhancing fake news detection. Furthermore, we outline future work aimed at expanding VERA-ARAB to establish it as a benchmark for Arabic tweets in fake news detection. We also discuss other potential applications that could leverage the VERA-ARAB dataset, emphasizing its value and versatility for advancing the field of fake news detection in Arabic social media. Potential applications include user veracity assessment, topic modeling, and named entity recognition, demonstrating the dataset's wide-ranging utility for broader research in information quality management on social media.
  },
   volume = 10,
   pages = {e2432},
   journal = {PeerJ Computer Science},
   issn = {2376-5992},
   url = {https://doi.org/10.7717/peerj-cs.2432},
   doi = {10.7717/peerj-cs.2432}
  }

### Augmented Data Format
Each sample produces a **triplet** stored in JSONL:
```json
{"id": "batch_0_idx_0", "text": "...", "label": 0, "type": "original"}
{"id": "batch_0_idx_0", "text": "...", "label": 0, "type": "dialect", "dialect_name": "المصرية"}
{"id": "batch_0_idx_0", "text": "...", "label": 0, "type": "adversarial"}
```

# Model Architecture
```
MARBERTv2 (UBC-NLP/MARBERTv2)
    │
    ├── Embedding Layer (frozen)
    ├── Encoder Layers 0–3 (frozen)
    ├── Encoder Layers 4–11 (trainable)
    │
    └── Triple Pooling Head
        ├── [CLS] token pooling
        ├── Mean pooling (masked)
        └── Max pooling (masked)
            │
            ├── Concatenation → [768 × 3 = 2304]
            ├── 5× Multi-Sample Dropout (p=0.2)
            ├── Linear → 2 classes
            └── Label-Smoothed Cross-Entropy (ε=0.1)
```
## Prerequisites
Python 3.13+
CUDA-capable GPU (recommended) or Apple Silicon (MPS)


## Setup
### Clone the repository
git clone https://github.com/<your-username>/MDAF.git
cd MDAF

### Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
 or: venv\Scripts\activate  # Windows

### Install dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate datasets
pip install scikit-learn seaborn matplotlib
pip install openpyxl tqdm
pip install llama-cpp-python          # For local GGUF inference
pip install bitsandbytes              # For 8-bit quantisation (GPU only)

## Experimental Setup

| Hyperparameter | Value |
|:---------------|:------|
| Base Model | `UBC-NLP/MARBERTv2` |
| Augmentation Model | `QCRI/Fanar-1-9B-Instruct` |
| Max Sequence Length | 80 tokens |
| Batch Size | 8 |
| Gradient Accumulation Steps | 4 |
| Effective Batch Size | 32 |
| Learning Rate | 2 × 10⁻⁵ |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Warmup Ratio | 15% |
| Max Gradient Norm | 1.0 |
| Number of Epochs | 10 (with early stopping) |
| Early Stopping Patience | 3 epochs |
| Cross-Validation | 5-Fold Stratified Group K-Fold |
| Label Smoothing | 0.1 |
| Dropout Rate | 0.2 (5 independent masks) |
| Pooling Strategy | CLS + Mean + Max (Triple Pooling) |
| Frozen Layers | Embeddings + Encoder Layers 0–3 |
| Trainable Layers | Encoder Layers 4–11 + Classification Head |
| Number of Dialects | 7 (Moroccan, Algerian, Tunisian, Egyptian, Syrian, Lebanese, Gulf) |
| Augmentation Quantisation | 8-bit (GPU) / 4-bit Q4_K_S (CPU) |
| Dialect Generation Temperature | 0.6 |
| Adversarial Generation Temperature | 0.5 |
| Max New Tokens (Augmentation) | 100 |
| Random Seed | 42 |


