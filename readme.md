# Interest Entropy: Rethinking Contrastive Learning for Sequential Recommendation with Interest Uncertainty

This repository contains the official implementation of **IERec**, our proposed model for contrastive learning-based sequential recommendation. The code is built upon [RecBole](https://github.com/RUCAIBox/RecBole).

## Abstract

Sequential Recommendation models predict the next item of interest based on their past behavior. However, sparse user behavior data makes it hard for the model to accurately learn user preferences.
Recently, contrastive learning has shown promise in this area. It augments data to form positive pairs and maximizing their similarity, allowing the model to learn more generalizable user interests. However, current methods mainly adopt uniform augmentation and alignment across all sequences, with limited consideration of the challenges arising from the distinct interest structure within them, namely semantic discrepancy and semantic bias. 
In this paper, we first investigate the impact of augmentation on sequence's semantic through interest entropy, which measures the diversity and density of interest distribution. 
Our finding shows only a small fraction of sequences are stable under perturbation. These sequences mainly exhibit low or high entropy, reflecting focused or casual interests.
This limits the effectiveness of contrastive learning, which relies on semantically consistent positive pairs.
Furthermore, with spectral analysis, we show that positive alignment may cause low-entropy sequences to overlook niche interests, while high-entropy sequences may amplify interest-irrelevant signal, which we term semantic bias.
Finally, based on Interest Entropy, we propose IERec, a simple yet effective mutual retrieval augmented contrastive learning framework that mitigates the above issues in a unified manner.
For each anchor sequence (those with low or high entropy), we retrieve a semantically similar sequence with complementary interest entropy, and concatenate them to form a positive view. Sequences that are easily affected, mainly those with medium entropy, are excluded from augmentation.
This approach can avoid harmful semantic discrepancy of positive pairs and reduce the effect of the semantic bias, leading to improved performance.
Moreover, leveraging interest entropy to guide contrastive learning can enhance the performance of existing CL-based SR methods.

## Requirements

The code is tested with Python `3.8.20`. Install dependencies via:

```bash
pip install -r requirements.txt
```

## Datasets

The `ML-100K` datasets are already located in the `./dataset` folder.

## Run IERec

To run the base IERec model on `ML-100K` with default settings:
```bash
python run_recbole.py --model=NEW --dataset=ml-100k --train_neg_sample_args=None
```

## Run IERec*

We also provide IERec*, an efficient variant that precomputes interest entropy values derived from a well-trained SR model (e.g., SASRec). This requires a separate pre-training stage but significantly reduces runtime by avoiding per-epoch IE recomputation.
While IERec* achieves slightly lower performance than IERec, it offers a favorable trade-off between efficiency and effectiveness.
```bash
python run_recbole.py --model=NEWV2 --dataset=ml-100k --train_neg_sample_args=None
```

## EGPF Module (New)

This repository additionally includes an **EGPF** (Entropy-Guided Prototype Fusion) enhancement in `NEW`.

### Motivation

IERec mainly addresses semantic discrepancy/bias through selective augmentation and selective alignment.
EGPF further stabilizes representation learning for entropy-sensitive samples by injecting prototype-level priors.

### Core Formulation

For each sequence, we first build interest prototypes and then perform entropy-guided fusion:

- $\tau(H)=\beta H$
- $\alpha_j=\mathrm{softmax}(w_j/\tau(H))$
- $z_{proto}=\sum_j\alpha_j p_j$
- $z_s=\gamma z_{enc} + (1-\gamma) z_{proto}$

Prototype constraint (currently applied to medium-entropy samples in this implementation):

- $L_{proto}=\lambda_{proto}\|z_s-z_{proto}\|_2^2$

Final training objective:

- $L = L_{rec} + \lambda_{cl} L_{SA-CL} + L_{proto}$

### Implementation Notes

- File: `recbole/model/sequential_recommender/new.py`
- Added module: `class EGPF(nn.Module)`
- Integrated into `NEW.forward`, `calculate_loss`, `predict`, and `full_sort_predict`
- Training stability options are provided: warmup epochs, apply interval, running entropy quantiles

### Main Hyperparameters (`recbole/properties/model/NEW.yaml`)

- `egpf_beta`: entropy-temperature scale
- `egpf_gamma`: encoder/prototype fusion ratio
- `lambda_proto`: prototype constraint strength
- `egpf_warmup_epochs`: epochs before activating EGPF
- `egpf_apply_interval`: apply EGPF every N training steps
- `egpf_quantile_low`, `egpf_quantile_high`: entropy band thresholds
- `egpf_use_running_quantile`, `egpf_quantile_momentum`: threshold smoothing for stability

### Reproduction

Train with EGPF-enabled `NEW`:

```bash
python run_recbole.py --model=NEW --dataset=ml-100k --train_neg_sample_args=None
```

Evaluate entropy subsets (overall/low/medium/high):

```bash
python eval_medium_subset.py --split valid
python eval_medium_subset.py --split test
```

Extract entropy statistics from a checkpoint:

```bash
python extract_entropy.py
```

## Custom Configuration
You can also add your own conguration, named `NEW.yaml`, into the `./recbole/properties/model` folder and run the above command. The results of the experiment will be stored in the `./log` directory.

## Citation
If IERec is useful for your research, please cite the paper:
```
@inproceedings{
    bin2025interest,
    title={Interest Entropy: Rethinking Contrastive Learning for Sequential Recommendation with Interest Uncertainty},
    author={Binquan Wu, KunZeng, Yicheng Luo, Junhao Zheng, and Qianli Ma},
    booktitle={32nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2026 - Research Track (First Cycle Deadline)},
    year={2025},
}
```
