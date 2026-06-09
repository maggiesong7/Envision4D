from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Optional

import torch
import json
import os
import numpy as np
from pytorch_lightning import LightningModule

from ..model.types import Gaussians
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg, get_data_shim
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import save_image
from ..visualization.vis_depth import viz_depth_tensor
from .decoder.decoder import Decoder
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    crop_range_h: list | None
    crop_range_w: list | None
    save_image: bool
    save_video: bool
    lane_shift: bool
    lane_shift_step: float
    eval_time_skip_steps: int


class ModelWrapper(LightningModule):
    encoder: torch.nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    test_cfg: TestCfg

    def __init__(
        self,
        test_cfg: TestCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
    ) -> None:
        super().__init__()
        self.test_cfg = test_cfg
        self.crop_h = test_cfg.crop_range_h
        self.crop_w = test_cfg.crop_range_w

        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)

        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        visualization_dump = {} if self.test_cfg.lane_shift else None

        with self.benchmarker.time("encoder"):
            encoder_output = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                encoder_output.pred_gaussian,
                encoder_output.extrinsics,
                encoder_output.intrinsics,
                (h, w),
                self.global_step,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        h_start, h_end = (self.crop_h[0], h - self.crop_h[1]) if self.crop_h is not None else (0, h)
        w_start, w_end = (self.crop_w[0], w - self.crop_w[1]) if self.crop_w is not None else (0, w)
        images_prob = output.color[0][:, :, h_start:h_end, w_start:w_end].clone()
        depths_prob = output.depth[0][:, h_start:h_end, w_start:w_end].clone()
        images_gt = batch["target"]["image"][0][:, :, h_start:h_end, w_start:w_end].clone()

        if self.test_cfg.save_image:
            for index, color, depth, gt in zip(batch["target"]["index"][0], images_prob, depths_prob, images_gt):
                save_image(color, path / scene / f"pred_rgb/{batch_idx}_{index:0>2}.png")
                save_image(gt, path / scene / f"target/{batch_idx}_{index:0>2}.png")

                save_path = path / scene / f"pred_depth/{batch_idx}_{index:0>2}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(viz_depth_tensor(1.0 / depth.cpu().detach(), return_numpy=True)).save(save_path)

        if self.test_cfg.lane_shift:
            shift_trans = torch.eye(4, dtype=torch.float32, device=batch["target"]["image"].device)
            shift_trans = shift_trans.broadcast_to((b, v, 4, 4)).clone()
            shift_trans[..., 0, 3] += self.test_cfg.lane_shift_step
            with self.benchmarker.time("decoder", num_calls=v):
                output_shift = self.decoder.forward(
                    encoder_output.pred_gaussian,
                    encoder_output.extrinsics @ shift_trans,
                    encoder_output.intrinsics,
                    (h, w),
                    self.global_step,
                )
            for index, shift in zip(batch["target"]["index"][0], output_shift.color[0]):
                save_image(shift, path / scene / f"pred_shift/{batch_idx}_{index:0>2}.png")

        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            for metric in ("psnr_future", "ssim_future", "lpips_future"):
                if metric not in self.test_step_outputs:
                    self.test_step_outputs[metric] = []

            future_gt, future_prob = images_gt[2:], images_prob[2:]
            self.test_step_outputs["psnr_future"].append(compute_psnr(future_gt, future_prob).mean().item())
            self.test_step_outputs["ssim_future"].append(compute_ssim(future_gt, future_prob).mean().item())
            self.test_step_outputs["lpips_future"].append(compute_lpips(future_gt, future_prob).mean().item())

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]):]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(self.test_cfg.output_path / name / "peak_memory.json")
            self.benchmarker.summarize()
