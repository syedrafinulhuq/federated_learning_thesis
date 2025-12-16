# fl_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import MyDataset
from src.focal_loss import FocalLoss
from src import utils


@dataclass
class ClientResult:
    client_id: int
    num_samples: int
    state_dict: Dict[str, torch.Tensor]
    train_loss: float
    train_acc: float
    train_f1: float


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


def _build_optimizer(args, model: torch.nn.Module):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    raise ValueError("optimizer must be 'sgd' or 'adam'")


def train_one_client(
    client_id: int,
    global_model: torch.nn.Module,
    client_texts: List[str],
    client_labels: List[int],
    client_sample_weights: Optional[np.ndarray],
    num_classes: int,
    args,
) -> ClientResult:
    # Clone the global model weights into a local model (same instance type)
    local_model = type(global_model)(args, num_classes)
    local_model.load_state_dict({k: v.detach().cpu() for k, v in global_model.state_dict().items()})

    if torch.cuda.is_available():
        local_model = local_model.cuda()

    criterion = _build_criterion(args, client_labels, num_classes)
    optimizer = _build_optimizer(args, local_model)

    # DataLoader
    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.workers,
        "drop_last": True,
    }

    train_set = MyDataset(client_texts, client_labels, args)

    if bool(args.use_sampler) and client_sample_weights is not None:
        sw = torch.from_numpy(client_sample_weights)
        sampler = WeightedRandomSampler(sw.type("torch.DoubleTensor"), len(sw), replacement=True)
        training_params["sampler"] = sampler
        training_params["shuffle"] = False

    train_loader = DataLoader(train_set, **training_params)

    local_model.train()
    losses = utils.AverageMeter()
    accs = utils.AverageMeter()

    y_true, y_pred = [], []

    for _ in range(args.local_epochs):
        for features, labels in train_loader:
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            logits = local_model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            y_true += labels.detach().cpu().numpy().tolist()
            y_pred += torch.max(logits, 1)[1].detach().cpu().numpy().tolist()

            metrics = utils.get_evaluation(
                np.asarray(labels.detach().cpu().numpy()),
                logits.detach().cpu().numpy(),
                list_metrics=["accuracy", "f1"],
            )
            losses.update(loss.item(), features.size(0))
            accs.update(metrics["accuracy"], features.size(0))

    f1 = float(utils.metrics.f1_score(y_true, y_pred, average="weighted")) if len(y_true) else 0.0  # safe

    # Returning CPU state_dict for aggregation
    cpu_state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}

    return ClientResult(
        client_id=client_id,
        num_samples=len(train_set),
        state_dict=cpu_state,
        train_loss=losses.avg,
        train_acc=accs.avg,
        train_f1=f1,
    )
