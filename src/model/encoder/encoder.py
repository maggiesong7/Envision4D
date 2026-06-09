from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians


@dataclass
class EncoderOutput:
    pred_gaussian: Gaussians
    extrinsics: Float[Tensor, "batch view 4 4"]
    intrinsics: Float[Tensor, "batch view 3 3"]
    extra_ext4sup: Float[Tensor, "batch v 9"]
    
    
T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> EncoderOutput:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x
