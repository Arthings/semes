from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
    SequentialSampler,
    SubsetRandomSampler,
)
import torch
from typing import List
from omegaconf import DictConfig
import numpy as np
from bibcvmodels.segmentation.instance import maskrcnn_collate_fn
import pandas as pd


class ActiveLearningDataset(Dataset):
    def __init__(self, dataset: Dataset, config: DictConfig, names: List[str]):
        """
        Args:
            dataset (Dataset): The original dataset.
            indices (list, optional): Indices of the samples to be used. If None, all samples are used.
        """
        self.dataset = dataset
        self.config = config
        self.names = names
        assert len(self.names) == len(self.dataset)
        self.start_rep()

        # need this for the maskrcnn collate_func when constructing the dataloaders in instance seg
        assert "problem" in config, (
            f"problem key should appear in the config file, problem should be one of {['classif', 'odetection', 'iseg', 'sseg']}"
        )
        self.pb = self.config.problem
        assert self.pb in ["classif", "odetection", "iseg", "sseg"], (
            f"{self.pb} is not a valid problem type, pb should be in {['classif', 'odetection', 'iseg', 'sseg']}"
        )

    def start_rep(self):
        """Initializes the labeled and unlabeled indices."""
        self.labeled_indices = np.array([]).astype(int)
        self.unlabeled_indices = torch.randperm(len(self.dataset)).numpy().astype(int)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def label(self, indices_to_label: np.ndarray) -> None:
        """
        Args:
            indices (list): Indices of the samples to be labeled.
        """
        assert np.isin(indices_to_label, self.labeled_indices).sum() == 0, (
            f"{np.isin(indices_to_label, self.labeled_indices).sum()} labeled items have been selected"
        )
        self.labeled_indices = np.concatenate((self.labeled_indices, indices_to_label))
        self.unlabeled_indices = np.setdiff1d(self.unlabeled_indices, indices_to_label)
        # check if we selected items from the unlabeled dataset

    @property
    def unlabeled_subset(self) -> int:
        if (self.config.subset == -1) or (
            self.config.subset >= len(self.unlabeled_indices)
        ):
            subset = self.unlabeled_indices
        else:
            subset = np.random.choice(
                self.unlabeled_indices, self.config.subset, replace=False
            )

        return subset

    def labeled_dataloader(self, **kwargs):
        # the drop_last can remove the first cycle if the labeled items are less thant the batch_size
        if len(self.labeled_indices) < self.config.batch_size:
            bs = len(self.labeled_indices)
        else:
            bs = self.config.batch_size
        print("bs", bs)
        return DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(self.labeled_indices),
            drop_last=True,
            batch_size=bs,
            num_workers=self.config.workers,
            collate_fn=maskrcnn_collate_fn if self.pb == "iseg" else None,
        )

    def unlabeled_dataloader(self, **kwargs):
        return DataLoader(
            self.dataset,
            sampler=SequentialSampler(self.unlabeled_subset),
            batch_size=self.config.batch_size,
            drop_last=True,
            num_workers=self.config.workers,
            collate_fn=maskrcnn_collate_fn if self.pb == "iseg" else None,
        )

    def save(self, path):
        pd.DataFrame({}, columns=[]).to_csv(self.config.log_dir + "")
