import numpy as np
import torch

from pytorch_lightning import LightningDataModule
from torch import Generator
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Callable

from . import get_dataset
from src.dataset.types import Stage


def worker_init_fn(worker_id):
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


DatasetShim = Callable[[Dataset, Stage], Dataset]


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg,
        data_loader_cfg,
        global_rank,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.data_loader_cfg = data_loader_cfg
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_generator(self, loader_cfg):
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def test_dataloader(self):
        test_ds = get_dataset("test", self.data_cfg)
        test_ds = self.dataset_shim(test_ds, "test")
        return DataLoader(
            test_ds,
            batch_size=self.data_loader_cfg.test.batch_size,
            drop_last=False,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        )
