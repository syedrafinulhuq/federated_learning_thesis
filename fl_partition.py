# fl_partition.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

@dataclass
class ClientSplit:
    client_id: int
    texts: List[str]
    labels: List[int]
    sample_weights: Optional[np.ndarray] = None


def _dirichlet_split_indices(labels: List[int], num_clients: int, alpha: float, seed: int) -> List[np.ndarray]:
    """
    Non-IID split: each class is distributed to clients according to a Dirichlet(alpha).
    Smaller alpha => more non-IID.
    """
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    classes = np.unique(labels_np)

    client_indices = [np.array([], dtype=np.int64) for _ in range(num_clients)]

    for c in classes:
        idx_c = np.where(labels_np == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))
        # convert proportions into counts
        counts = (proportions * len(idx_c)).astype(int)
        # fix rounding to match total
        diff = len(idx_c) - counts.sum()
        if diff > 0:
            counts[:diff] += 1
        elif diff < 0:
            # remove extras
            for i in range(abs(diff)):
                j = i % num_clients
                if counts[j] > 0:
                    counts[j] -= 1

        start = 0
        for client_id, k in enumerate(counts):
            if k > 0:
                client_indices[client_id] = np.concatenate([client_indices[client_id], idx_c[start:start+k]])
            start += k

    for i in range(num_clients):
        rng.shuffle(client_indices[i])

    return client_indices


def _iid_split_indices(n: int, num_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, num_clients)


def make_client_splits(
    texts: List[str],
    labels: List[int],
    sample_weights: np.ndarray,
    num_clients: int,
    split_type: str = "dirichlet",  # "iid" or "dirichlet"
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> List[ClientSplit]:
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")

    if split_type == "iid":
        indices_list = _iid_split_indices(len(texts), num_clients, seed)
    elif split_type == "dirichlet":
        indices_list = _dirichlet_split_indices(labels, num_clients, dirichlet_alpha, seed)
    else:
        raise ValueError("split_type must be either 'iid' or 'dirichlet'")

    splits: List[ClientSplit] = []
    for cid, idxs in enumerate(indices_list):
        idxs = np.asarray(idxs, dtype=np.int64)
        splits.append(
            ClientSplit(
                client_id=cid,
                texts=[texts[i] for i in idxs],
                labels=[labels[i] for i in idxs],
                sample_weights=sample_weights[idxs] if sample_weights is not None else None,
            )
        )

    return splits
