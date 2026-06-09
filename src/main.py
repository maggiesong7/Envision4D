import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import warnings

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from pytorch_lightning import Trainer

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.misc.LocalLogger import LocalLogger
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def test(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=LocalLogger(),
        devices="auto",
        enable_progress_bar=True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder, cfg.dataset)

    model_wrapper = ModelWrapper(
        cfg.test,
        encoder,
        encoder_visualizer,
        get_decoder("test", cfg.model.decoder, cfg.dataset),
    )

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        global_rank=trainer.global_rank,
    )

    checkpoint_path = Path(cfg.checkpointing.load) if cfg.checkpointing.load else None
    trainer.test(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')
    test()
