!pip install torch tensorboardX tqdm scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import torch
from torch.utils.data import TensorDataset, DataLoader
dataset = pd.read_csv('Data.csv')
X_raw = dataset.iloc[:, 1].astype(str).values  # Raw URLs
y_raw = dataset.iloc[:, -1].values

NUM_CHARACTERS = 256
MAX_LENGTH = 256

def one_hot_encode_url(url, max_length):
    encoded = np.zeros((max_length, NUM_CHARACTERS), dtype=np.float32)
    for i, char in enumerate(url[:max_length]):
        if char in CHAR2IDX:
            encoded[i, CHAR2IDX[char]] = 1.0
    return encoded

# Encode URLs
X_encoded = np.array([one_hot_encode_url(url, MAX_LENGTH) for url in X_raw])

# Convert to tensors
X_tensor = torch.tensor(X_encoded)
# Encode target: Binary 'Adult' = 1, Others = 0
y = np.array([1 if label.lower() == "adult" else 0 for label in y_raw], dtype=np.float32)
y_tensor = torch.tensor(y).unsqueeze(1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
from torch.utils.data import random_split

# Define dataset again (if not already)
full_dataset = TensorDataset(X_tensor, y_tensor)

# Calculate lengths
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Extract training labels for sampler
# Note: random_split returns Subset objects which have `.indices` attribute
train_indices = train_dataset.indices
train_labels = y_tensor[train_indices].numpy().astype(int).flatten()

# Compute class counts and weights
class_counts = np.bincount(train_labels)
class_weights = 1. / class_counts
sample_weights = class_weights[train_labels]

# Create WeightedRandomSampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


