# fl_server.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_loader import MyDataset
from src.focal_loss import FocalLoss
from src import utils


def fedavg(state_dicts: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    if len(state_dicts) == 0:
        raise ValueError("No client models provided to FedAvg")

    total = float(sum(weights))
    avg: Dict[str, torch.Tensor] = {}

    for k in state_dicts[0].keys():
        avg[k] = sum((sd[k].float() * (w / total)) for sd, w in zip(state_dicts, weights))
    return avg


def _build_criterion(args, train_labels: List[int], num_classes: int):
    if not bool(args.focal_loss):
        if bool(args.class_weights):
            class_counts = dict(Counter(train_labels))
            m = max(class_counts.values())
            for c in class_counts:
                class_counts[c] = m / class_counts[c]
            weights = torch.Tensor([class_counts[k] for k in sorted(class_counts.keys())])
            if torch.cuda.is_available():
                weights = weights.cuda()
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    if args.alpha is None:
        return FocalLoss(gamma=args.gamma, alpha=None)
    return FocalLoss(gamma=args.gamma, alpha=[args.alpha] * num_classes)


def evaluate_server(model, texts: List[str], labels: List[int], args, num_classes: int, criterion) -> Tuple[float, float, float]:
    model.eval()
    val_set = MyDataset(texts, labels, args)
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    losses = utils.AverageMeter()
    accs = utils.AverageMeter()

    y_true, y_pred = [], []

    with torch.no_grad():
        for features, y in loader:
            if torch.cuda.is_available():
                features = features.cuda()
                y = y.cuda()

            logits = model(features)
            loss = criterion(logits, y)

            y_true += y.detach().cpu().numpy().tolist()
            y_pred += torch.max(logits, 1)[1].detach().cpu().numpy().tolist()

            metrics = utils.get_evaluation(
                np.asarray(y.detach().cpu().numpy()),
                logits.detach().cpu().numpy(),
                list_metrics=["accuracy", "f1"],
            )

            losses.update(loss.item(), features.size(0))
            accs.update(metrics["accuracy"], features.size(0))

    f1 = float(utils.metrics.f1_score(y_true, y_pred, average="weighted")) if len(y_true) else 0.0
    return losses.avg, accs.avg, f1
