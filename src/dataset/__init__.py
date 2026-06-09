from pathlib import Path
from dataclasses import dataclass
from torch import nn
from torch.utils.data import Dataset

from .types import DataShim, Stage
from .waymo.waymo_dataset import WaymoDataset

DATASETS: dict[str, Dataset] = {
    "waymo": WaymoDataset
}
    
    
@dataclass
class WaymoDatasetCfg:
    name: str
    data_path: Path
    load_size: list[int]
    num_cams: int
    num_frames: int
    background_color: list[float]


DatasetCfg = WaymoDatasetCfg


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    seed: int


@dataclass
class DataLoaderCfg:
    test: DataLoaderStageCfg
    
    
def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


def get_dataset(
    mode: str,
    cfg: DatasetCfg,
) -> Dataset:
    return DATASETS[cfg.name](mode, cfg)