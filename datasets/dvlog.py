#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
from pathlib import Path
from typing import Union, Optional, List
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def normalize_data(x):
    """
    Normalize the input data array x to zero mean and unit variance.
    Args:
        x (np.ndarray)  : Input data.
    Returns:
        np.ndarray      : Normalized data.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.clip(std, a_min=1e-8, a_max=None)  # Avoid division by zero
    return (x - mean) / std

class DVlog(data.Dataset):
    def __init__(
        self, root: Union[str, Path], fold: str="train", 
        gender: str="both", transform=None, target_transform=None
    ):
        self.root = root if isinstance(root, Path) else Path(root)
        self.fold = fold
        self.gender = gender
        self.transform = transform
        self.target_transform = target_transform

        self.features = []
        self.labels = []
        with open(self.root / "labels.csv", "r") as f:
            for line in f:
                sample = line.strip().split(",")
                if self.is_sample(sample):
                    s_id = sample[0]
                    s_label = int(sample[1]=="depression")
                    self.labels.append(s_label)

                    v_feature_path = self.root / s_id / f"{s_id}_visual.npy"
                    a_feature_path = self.root / s_id / f"{s_id}_acoustic.npy"
                    v_feature = np.load(v_feature_path)
                    a_feature = np.load(a_feature_path)

                    # concat visual and acoustic features along the 2nd axis
                    T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                    # print(f"T_v: {T_v} & T_a: {T_a}")
                    if T_v == T_a:
                        feature = np.concatenate(
                            (v_feature, a_feature), axis=1
                        ).astype(np.float32)
                    else:
                        T = min(T_v, T_a)
                        feature = np.concatenate(
                            (v_feature[:T], a_feature[:T]), axis=1
                        ).astype(np.float32)
                    feature = normalize_data(feature)  # Normalize feature
                    self.features.append(feature)

    def is_sample(self, sample) -> bool:
        gender, fold = sample[3], sample[4]
        if self.gender == "both":
            return fold == self.fold
        return (fold == self.fold) and (gender == self.gender)

    def __getitem__(self, i: int):
        feature = self.features[i]
        label = self.labels[i]
        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # print("DVlog!!")
        # print(len(feature), len(label))
        return feature, label

    def __len__(self):
        return len(self.labels)

def _collate_fn(batch):
    # batch: [(feature, label), (feature, label), ...]
    # feature.shape = [T, F], F is fixed, but T varies from sample to sample
    features, labels = zip(*batch)
    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features], batch_first=True
    )
    labels = torch.tensor(labels)
    # print("DVlog!!")
    # print(padded_features.shape, labels.shape)
    return padded_features, labels

def get_dvlog_dataloader(
    root: Union[str, Path], fold: str="train", batch_size: int=8, 
    gender: str="both",
    transform=None, target_transform=None, 
):
    """
    Get dataloader for DVlog dataset.

    Args:
        root (Union[str, Path])         :   path to the dvlog dataset. Should be something
                                            like `*/dvlog-dataset`.
        fold (str, optional)            :   train / valid / test. Defaults to "train".
        batch_size (int, optional)      :   Defaults to 8.
        gender (str, optional)          :   m / f / both. Defaults to both.
        transform (optional)            :   Defaults to None.
        target_transform (optional)     :   Defaults to None.

    Returns:
        the dataloader.
    """
    dataset = DVlog(root, fold, gender, transform, target_transform)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, 
        collate_fn=_collate_fn,
        shuffle=(fold=="train"),
    )
    return dataloader

def kfold_get_dvlog_dataloader(
    root: Union[str, Path], indices: List[int], batch_size: int = 8, gender: str = "both", 
    transform=None, target_transform=None
):
    """
    Get dataloader for DVlog dataset based on specific indices for k-fold cross-validation.

    Args:
        root (Union[str, Path])         : path to the dvlog dataset. Should be something
                                          like `*/dvlog-dataset`.
        indices (list[int])             : List of indices to be used in the current fold.
        batch_size (int, optional)      : Defaults to 8.
        gender (str, optional)          : m / f / both. Defaults to both.
        transform (optional)            : Defaults to None.
        target_transform (optional)     : Defaults to None.

    Returns:
        the dataloader.
    """
    dataset = DVlog(root)
    subset = data.Subset(dataset, indices)  # Use the specified indices to create a subset
    dataloader = data.DataLoader(
        subset, batch_size=batch_size, 
        collate_fn=_collate_fn,
        shuffle=True
    )
    return dataloader

''' 
if __name__ == '__main__':
    train_loader = get_dvlog_dataloader(
        "./data/dvlog-dataset", "train"
    )
    print(f"train_loader: {len(train_loader.dataset)} samples")
    valid_loader = get_dvlog_dataloader(
        "./data/dvlog-dataset", "valid"
    )
    print(f"valid_loader: {len(valid_loader.dataset)} samples")
    test_loader = get_dvlog_dataloader(
        "./data/dvlog-dataset", "test"
    )
    print(f"test_loader: {len(test_loader.dataset)} samples")

    b1 = next(iter(train_loader))[0]
    print(f"A train_loader batch: shape={b1.shape}, dtype={b1.dtype}")
    b2 = next(iter(valid_loader))[0]
    print(f"A valid_loader batch: shape={b2.shape}, dtype={b2.dtype}")
    b3 = next(iter(test_loader))[0]
    print(f"A test_loader batch: shape={b3.shape}, dtype={b3.dtype}")

'''