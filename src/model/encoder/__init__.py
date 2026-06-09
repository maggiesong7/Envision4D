from typing import Optional

from .encoder import Encoder
from .encoder_envision4d import EncoderEnvi4D, EncoderEnvi4DCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_envision4d import EncoderVisualizerEnvi4D

from ...dataset import DatasetCfg

ENCODERS = {
    "envision4d": (EncoderEnvi4D, EncoderVisualizerEnvi4D),
}

EncoderCfg = EncoderEnvi4DCfg


def get_encoder(cfg: EncoderCfg, dataset_cfg: DatasetCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg, dataset_cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
