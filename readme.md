# IERec + EGPF for Sequential Recommendation

This repository is based on [RecBole](https://github.com/RUCAIBox/RecBole) and extends IERec with a new **EGPF** (Entropy-Guided Prototype Fusion) module.

## Quick Test Summary (ML-100K)

| Model | NDCG@10 | NDCG@20 | Hit@10 | Hit@20 |
|---|---:|---:|---:|---:|
| CL4SRec† | 0.0519 | 0.0716 | 0.1106 | 0.1888 |
| IERec (baseline) | 0.0532 | 0.0746 | 0.1106 | 0.1953 |
| IERec + EGPF | **0.0550** | **0.0761** | **0.1151** | **0.1995** |

† CL4SRec numbers are taken from the IERec paper's ML-100K test table.

## EGPF First (Main Contribution in This Repo)

### Motivation

IERec improves contrastive learning by entropy-guided augmentation/alignment, while EGPF further stabilizes sequence representations by adding prototype priors.
In this implementation, EGPF mainly targets medium-entropy samples (the most augmentation-sensitive group).

### Core Formulation

For each sequence, we build prototypes and perform entropy-guided fusion:

- $\tau(H)=\beta H$
- $\alpha_j=\mathrm{softmax}(w_j/\tau(H))$
- $z_{proto}=\sum_j\alpha_j p_j$
- $z_s=\gamma z_{enc} + (1-\gamma) z_{proto}$

Prototype constraint:

- $L_{proto}=\lambda_{proto}\|z_s-z_{proto}\|_2^2$

Final objective:

- $L = L_{rec} + \lambda_{cl} L_{SA-CL} + L_{proto}$

### Where It Is Implemented

- `recbole/model/sequential_recommender/new.py` (`EGPF`, fusion path, proto loss)
- `recbole/properties/model/NEW.yaml` (EGPF hyperparameters)
- `eval_medium_subset.py` (overall/low/medium/high entropy subset evaluation)

### EGPF Hyperparameters

- `egpf_beta`: entropy-temperature scale
- `egpf_gamma`: encoder/prototype fusion ratio
- `lambda_proto`: prototype constraint strength
- `egpf_warmup_epochs`: epochs before activating EGPF
- `egpf_apply_interval`: apply EGPF every N training steps
- `egpf_quantile_low`, `egpf_quantile_high`: entropy band thresholds
- `egpf_use_running_quantile`, `egpf_quantile_momentum`: threshold smoothing

### Reproduction

Train EGPF-enabled `NEW`:

```bash
python run_recbole.py --model=NEW --dataset=ml-100k --train_neg_sample_args=None
```

Evaluate entropy subsets:

```bash
python eval_medium_subset.py --split valid
python eval_medium_subset.py --split test
```

Extract entropy statistics:

```bash
python extract_entropy.py
```

## IERec (Brief)

IERec is the base framework from the KDD paper, with two key ideas:

- **SLA**: retrieval-based sequence-level augmentation for low/high entropy samples
- **SA-CL**: selective contrastive alignment on low/high entropy samples

This repo keeps IERec training logic and adds EGPF as an additional representation regularization path.

## Requirements

The code is tested with Python `3.8.20`. Install dependencies via:

```bash
pip install -r requirements.txt
```

## Datasets

The `ML-100K` datasets are already located in the `./dataset` folder.

## Run IERec*

We also provide IERec*, an efficient variant that precomputes interest entropy values derived from a well-trained SR model (e.g., SASRec). This requires a separate pre-training stage but significantly reduces runtime by avoiding per-epoch IE recomputation.
While IERec* achieves slightly lower performance than IERec, it offers a favorable trade-off between efficiency and effectiveness.
```bash
python run_recbole.py --model=NEWV2 --dataset=ml-100k --train_neg_sample_args=None
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
