from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DataLoaderCfg, DatasetCfg
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import TestCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class RootCfg:
    wandb: dict
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    checkpointing: CheckpointingCfg
    test: TestCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(cfg, RootCfg)
