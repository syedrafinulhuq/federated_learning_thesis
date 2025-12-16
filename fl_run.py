# fl_run.py
from __future__ import annotations
import argparse
import json
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from src.data_loader import load_data
from src.model import CharacterLevelCNN

from fl_partition import make_client_splits
from fl_client import train_one_client
from fl_server import fedavg, evaluate_server


def main(args):
    # Load dataset (same as your centralized pipeline)
    texts, labels, num_classes, sample_weights = load_data(args)

    # Centralized split (server keeps val/test)
    train_texts, temp_texts, train_labels, temp_labels, train_sw, temp_sw = train_test_split(
        texts, labels, sample_weights, test_size=0.30, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels, _, _ = train_test_split(
        temp_texts, temp_labels, temp_sw, test_size=0.50, random_state=42, stratify=temp_labels
    )

    # Build global model
    global_model = CharacterLevelCNN(args, num_classes)
    if args.init_from_checkpoint is not None:
        state = torch.load(args.init_from_checkpoint, map_location="cpu")
        global_model.load_state_dict(state)

    if torch.cuda.is_available():
        global_model = global_model.cuda()

    # Partition training set to clients
    client_splits = make_client_splits(
        train_texts,
        train_labels,
        train_sw,
        num_clients=args.num_clients,
        split_type=args.client_split,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )

    # Server criterion for evaluation
    # (We reuse training labels distribution for optional class-weights; it’s centralized here)
    from fl_server import _build_criterion
    criterion = _build_criterion(args, train_labels, num_classes)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    best_val_f1 = -1.0

    for rnd in range(args.rounds):
        # Sample clients each round
        rng = np.random.default_rng(args.seed + rnd)
        m = max(1, int(args.client_fraction * args.num_clients))
        selected = rng.choice(len(client_splits), size=m, replace=False)

        client_states = []
        client_sizes = []

        for idx in selected:
            cs = client_splits[idx]
            result = train_one_client(
                client_id=cs.client_id,
                global_model=global_model,
                client_texts=cs.texts,
                client_labels=cs.labels,
                client_sample_weights=cs.sample_weights,
                num_classes=num_classes,
                args=args,
            )
            client_states.append(result.state_dict)
            client_sizes.append(result.num_samples)

        # Aggregate (FedAvg)
        new_state = fedavg(client_states, client_sizes)
        global_model.load_state_dict(new_state)

        # Server-side validation
        val_loss, val_acc, val_f1 = evaluate_server(global_model, val_texts, val_labels, args, num_classes, criterion)
        print(f"[Round {rnd+1}/{args.rounds}] val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_best is not None:
                torch.save(global_model.state_dict(), args.save_best)
                print(f"[INFO] Saved best global model to: {args.save_best}")

    # Final test
    test_loss, test_acc, test_f1 = evaluate_server(global_model, test_texts, test_labels, args, num_classes, criterion)
    print("=" * 60)
    print(f"[FINAL TEST] loss={test_loss:.4f} acc={test_acc:.4f} f1={test_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Federated Learning (FedAvg) - Character CNN URL classifier")

    # Keep your existing args (same defaults as train.py)
    parser.add_argument("--data_path", type=str, default="./src/url.csv")
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--steps", nargs="+", default=["lower"])
    parser.add_argument("--balance", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ratio", type=float, default=1.0)

    parser.add_argument("--alphabet", type=str,
                        default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
    parser.add_argument("--number_of_characters", type=int, default=69)
    parser.add_argument("--extra_characters", type=str, default="")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--dropout_input", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)

    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--class_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--focal_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--use_sampler", type=int, default=1, choices=[0, 1])

    # Federated args
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--client_fraction", type=float, default=1.0, help="fraction of clients per round")
    parser.add_argument("--local_epochs", type=int, default=1)

    parser.add_argument("--client_split", type=str, default="dirichlet", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    # Optional: warm start from centralized checkpoint
    parser.add_argument("--init_from_checkpoint", type=str, default=None)

    # Save best global model path
    parser.add_argument("--save_best", type=str, default="./models/fedavg_best.pth")

    args = parser.parse_args()
    main(args)
