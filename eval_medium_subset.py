import argparse
import math
import sys
from typing import Dict, Tuple

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model


BASELINE_CKPT = "saved/NEW-Apr-23-2026_18-14-26.pth"
EGPF_CKPT = "saved/NEW-Apr-25-2026_02-26-41.pth"


def load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


def build_config(split_name: str, config_overrides=None):
    config = Config(
        model="NEW",
        dataset="ml-100k",
        config_file_list=["recbole/properties/model/NEW.yaml"],
        config_dict=config_overrides or {},
    )
    return config


def load_model(config, dataset, ckpt_path: str):
    model_cls = get_model(config["model"])
    model = model_cls(config, dataset).to(config["device"])
    checkpoint = load_checkpoint(ckpt_path, config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_entropy(model, eval_data) -> torch.Tensor:
    entropies = []
    for batch in eval_data:
        interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
        interaction = interaction.to(model.device)
        entropy, _, _, _ = model.build_entropy_prototypes(
            interaction[model.ITEM_SEQ], interaction[model.ITEM_SEQ_LEN], theta=model.theta
        )
        entropies.append(entropy.detach().cpu())
    return torch.cat(entropies, dim=0)


@torch.no_grad()
def evaluate_medium_subset(model, eval_data, medium_mask: torch.Tensor) -> Dict[str, float]:
    max_k = 20
    results = {"hit@10": 0.0, "hit@20": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0}
    total = 0
    offset = 0

    for batch in eval_data:
        if isinstance(batch, (tuple, list)):
            interaction = batch[0]
            positive_i = batch[3]
        else:
            interaction = batch
            positive_i = batch[model.ITEM_ID]

        interaction = interaction.to(model.device)
        scores = model.full_sort_predict(interaction)
        pos_items = positive_i.detach().cpu()
        topk_items = torch.topk(scores, k=max_k, dim=1).indices.detach().cpu()

        batch_size = scores.size(0)
        batch_mask = medium_mask[offset : offset + batch_size]
        offset += batch_size

        for i in range(batch_size):
            if not bool(batch_mask[i]):
                continue

            total += 1
            pos_item = int(pos_items[i].item())
            matches = (topk_items[i] == pos_item).nonzero(as_tuple=False)
            if matches.numel() == 0:
                continue

            rank = int(matches[0].item())
            if rank < 10:
                results["hit@10"] += 1.0
                results["ndcg@10"] += 1.0 / math.log2(rank + 2)
            results["hit@20"] += 1.0
            results["ndcg@20"] += 1.0 / math.log2(rank + 2)

    if total == 0:
        raise ValueError("Medium subset is empty. Please check the entropy quantiles or split setting.")

    for key in results:
        results[key] /= total
    results["count"] = float(total)
    return results


@torch.no_grad()
def evaluate_subset(model, eval_data, subset_mask: torch.Tensor = None) -> Dict[str, float]:
    """Evaluate model on the whole set or a masked subset of instances.

    Args:
        model: trained recommender.
        eval_data: full-sort eval dataloader.
        subset_mask: boolean tensor aligned with eval samples. If None, evaluates all samples.
    """
    max_k = 20
    results = {"hit@10": 0.0, "hit@20": 0.0, "ndcg@10": 0.0, "ndcg@20": 0.0}
    total = 0
    offset = 0

    for batch in eval_data:
        if isinstance(batch, (tuple, list)):
            interaction = batch[0]
            positive_i = batch[3]
        else:
            interaction = batch
            positive_i = batch[model.ITEM_ID]

        interaction = interaction.to(model.device)
        scores = model.full_sort_predict(interaction)
        pos_items = positive_i.detach().cpu()
        topk_items = torch.topk(scores, k=max_k, dim=1).indices.detach().cpu()

        batch_size = scores.size(0)
        if subset_mask is None:
            batch_mask = torch.ones(batch_size, dtype=torch.bool)
        else:
            batch_mask = subset_mask[offset : offset + batch_size]
        offset += batch_size

        for i in range(batch_size):
            if not bool(batch_mask[i]):
                continue

            total += 1
            pos_item = int(pos_items[i].item())
            matches = (topk_items[i] == pos_item).nonzero(as_tuple=False)
            if matches.numel() == 0:
                continue

            rank = int(matches[0].item())
            if rank < 10:
                results["hit@10"] += 1.0
                results["ndcg@10"] += 1.0 / math.log2(rank + 2)
            results["hit@20"] += 1.0
            results["ndcg@20"] += 1.0 / math.log2(rank + 2)

    if total == 0:
        raise ValueError("Selected subset is empty. Please check the entropy quantiles or split setting.")

    for key in results:
        results[key] /= total
    results["count"] = float(total)
    return results


def format_result(result: Dict[str, float]) -> str:
    return (
        f"count={int(result['count'])}, "
        f"ndcg@10={result['ndcg@10']:.4f}, ndcg@20={result['ndcg@20']:.4f}, "
        f"hit@10={result['hit@10']:.4f}, hit@20={result['hit@20']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    args = parser.parse_args()

    # 防止 RecBole 的 Config 读取到脚本参数
    sys.argv = [sys.argv[0]]

    # 统一数据划分，保证 baseline / EGPF 使用同一批样本
    # baseline-like: use the same EGPF code path, but neutralize the fusion loss/fusion effect
    # gamma=1.0 and lambda_proto=0.0 make z_s == z_enc and proto_loss == 0.
    base_config = build_config(args.split, {"egpf_gamma": 1.0, "lambda_proto": 0.0})
    dataset = create_dataset(base_config)
    train_data, valid_data, test_data = data_preparation(base_config, dataset)
    eval_data = valid_data if args.split == "valid" else test_data

    # baseline：用 4.23 的旧 checkpoint 定义“中熵子集”
    baseline_model = load_model(base_config, train_data.dataset, BASELINE_CKPT)
    baseline_entropy = collect_entropy(baseline_model, eval_data)
    low_thresh = torch.quantile(baseline_entropy, 0.3)
    high_thresh = torch.quantile(baseline_entropy, 0.7)
    low_mask = baseline_entropy < low_thresh
    medium_mask = (baseline_entropy >= low_thresh) & (baseline_entropy <= high_thresh)
    high_mask = baseline_entropy > high_thresh

    baseline_overall = evaluate_subset(baseline_model, eval_data, None)
    baseline_low = evaluate_subset(baseline_model, eval_data, low_mask)
    baseline_medium = evaluate_subset(baseline_model, eval_data, medium_mask)
    baseline_high = evaluate_subset(baseline_model, eval_data, high_mask)

    # EGPF：加载新 checkpoint
    egpf_config = build_config(args.split)
    egpf_model = load_model(egpf_config, train_data.dataset, EGPF_CKPT)
    egpf_overall = evaluate_subset(egpf_model, eval_data, None)
    egpf_low = evaluate_subset(egpf_model, eval_data, low_mask)
    egpf_medium = evaluate_subset(egpf_model, eval_data, medium_mask)
    egpf_high = evaluate_subset(egpf_model, eval_data, high_mask)

    groups = [
        ("overall", baseline_overall, egpf_overall),
        ("low", baseline_low, egpf_low),
        ("medium", baseline_medium, egpf_medium),
        ("high", baseline_high, egpf_high),
    ]

    print("=" * 80)
    print(f"Split: {args.split}")
    for name, base_res, egpf_res in groups:
        print(f"[{name}] baseline: {format_result(base_res)}")
        print(f"[{name}] EGPF    : {format_result(egpf_res)}")
        print("-" * 80)
        for key in ["ndcg@10", "ndcg@20", "hit@10", "hit@20"]:
            delta = egpf_res[key] - base_res[key]
            rel = (delta / base_res[key] * 100.0) if base_res[key] != 0 else float("nan")
            print(f"[{name}] {key}: {delta:+.4f} ({rel:+.2f}%)")
        print("-" * 80)
    print("=" * 80)


if __name__ == "__main__":
    main()
