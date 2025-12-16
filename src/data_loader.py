
import json
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from.import utils
import torch
with open('config.json') as f:
    config = json.load(f)

def get_sample_weights(labels):
    counter = Counter(labels)
    for k in counter:
        counter[k] = 1 / counter[k]
    sample_weights = np.array([counter[l] for l in labels])
    return sample_weights


def load_data(args):
    chunks = pd.read_csv(
        args.data_path,
        names=["id", "url", "label"],  # Manually define columns
        usecols=["url", "label"],      # Only use relevant columns
        chunksize=args.chunksize,
        encoding=args.encoding,
        nrows=args.max_rows,
        sep=args.sep,
    )

    texts = []
    labels = []

    for df_chunk in tqdm(chunks):
        aux_df = df_chunk.copy()
        aux_df = aux_df.sample(frac=1)
        aux_df = aux_df[~aux_df["url"].isnull()]
        aux_df = aux_df[(aux_df["url"].map(len) > 1)]
        aux_df["processed_text"] = aux_df["url"].map(
            lambda text: utils.process_text(args.steps, text)
        )

        texts += aux_df["processed_text"].tolist()
        labels += [1 if str(label).lower() == "adult" else 0 for label in aux_df["label"].tolist()]

    if bool(args.balance):
        counter = Counter(labels)
        keys = list(counter.keys())
        values = list(counter.values())
        count_minority = np.min(values)

        balanced_labels = []
        balanced_texts = []

        for key in keys:
            balanced_texts += [text for text, label in zip(texts, labels) if label == key][: int(args.ratio * count_minority)]
            balanced_labels += [label for text, label in zip(texts, labels) if label == key][: int(args.ratio * count_minority)]

        texts = balanced_texts
        labels = balanced_labels
 
    number_of_classes = len(set(labels))
    print(f"Data loaded successfully with {len(texts)} rows and {number_of_classes} labels")
    print("Class distribution:", Counter(labels))

    sample_weights = get_sample_weights(labels)

    return texts, labels, number_of_classes, sample_weights


class MyDataset(Dataset):
    def __init__(self, texts, labels, args):
        self.data_path = 'src/url.csv'
        self.texts = texts
        self.labels = labels
        self.length = len(self.texts)

        self.vocabulary = args.alphabet + args.extra_characters
        self.char2idx = {c: i for i, c in enumerate(self.vocabulary)}
        self.number_of_characters = len(self.vocabulary)
        self.max_length = args.max_length
        self.identity_mat = np.identity(self.number_of_characters)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        raw_text = raw_text[::-1]  # Reverse as per original logic

        # One-hot encode the URL using CHAR2IDX
        data = np.zeros((self.max_length, self.number_of_characters), dtype=np.float32)
        for i, char in enumerate(raw_text[:self.max_length]):
            if char in self.char2idx:
                data[i, self.char2idx[char]] = 1.0

        label = self.labels[index]
        return torch.tensor(data), label

if __name__ == "__main__":
    class Args:
        data_path = "src/url.csv"
        chunksize = 50000
        encoding = "utf-8"
        max_rows = 10000000           # Or None to load all rows
        sep = ","
        steps = ["lower", "remove_hashtags", "remove_user_mentions", "remove_urls", "remove_extensions","remove_stop_words"]
        alphabet = config["alphabet"]["en"]["both"]["alphabet"]
        extra_characters = config.get("extra_characters", "")
        max_length = config["data"]["max_length"]
        balance = False             # Set True if you want balanced sampling
        ratio = 1.0

    args = Args()

    # Load and show data samples
    texts, labels, number_of_classes, sample_weights = load_data(args)
    print("Sample Text:", texts[0])
    print("Sample Label:", labels[0])
    print("Number of Classes:", number_of_classes)
    print("Sample Weight:", sample_weights[0])

    # Create Dataset object
    dataset = MyDataset(texts, labels, args)
    print("Dataset size:", len(dataset))

    # Show one sample from the dataset
    sample_data, sample_label = dataset[0]
    print("Sample tensor shape:", sample_data.shape)
    print("Sample label:", sample_label)