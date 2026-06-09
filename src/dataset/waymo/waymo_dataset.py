import glob
import os
import pickle
import time
import tqdm

import numpy as np
from PIL import Image
import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset
from torchvision import transforms


class WaymoDataset(Dataset):
    def __init__(
            self,
            mode,
            cfg,
    ):
        super().__init__()
        self.data_path = cfg.data_path
        self.load_size = cfg.load_size
        self.num_cams = cfg.num_cams
        self.num_frames = cfg.num_frames

        assert self.num_cams == 1

        # Single view dataset, only need the original size of the first camera
        self.ORIGINAL_SIZE = [1280, 1920]  # Only one camera, corresponding to the original cam_id 0

        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])

        self.data_dir = os.path.join(self.data_path, "validation")

        # Get all scene IDs
        scene_dirs = sorted(glob.glob(os.path.join(self.data_dir, "*") + "/"))
        self.scene_ids_list = [int(os.path.basename(os.path.dirname(scene_dir))) for scene_dir in scene_dirs]

        self.interval = 15
        self.all_datas = []
        for scene_id in self.scene_ids_list:
            img_dir = os.path.join(self.data_dir, str(scene_id).zfill(3), "images")

            img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

            timesteps = []
            for img_file in img_files:
                filename = os.path.basename(img_file)
                timestep = int(filename.split('_')[0])
                timesteps.append(timestep)

            start_timestep = min(timesteps)
            end_timestep = max(timesteps)

            img_filepaths = self.create_all_filelist(scene_id, start_timestep, end_timestep)

            for i in range(10):
                self.all_datas.append(
                    {
                        "scene_id": scene_id,
                        "img_filepaths": img_filepaths[i * self.interval:(i + 1) * self.interval],
                    }
                )

    def create_all_filelist(self, scene_id, start_timestep, end_timestep):
        img_filepaths = []

        scene_path = os.path.join(self.data_dir, str(scene_id).zfill(3))

        if end_timestep == -1:
            all_filepaths = os.path.join(scene_path, 'images', "*.png")
            image_filenames_all = glob.glob(all_filepaths)
            end_timestep = len(image_filenames_all) - 1

        for t in range(start_timestep, end_timestep):
            img_filepaths.append(
                os.path.join(scene_path, "images", f"{t:06d}_0.png")
            )

        return img_filepaths

    def load_rgb(self, img_filepath):
        rgb = Image.open(img_filepath).convert("RGB")
        rgb = rgb.resize(
            (self.load_size[1], self.load_size[0]), Image.BILINEAR
        )
        rgb = np.array(rgb, dtype=np.float32) / 255.0
        return rgb

    def __getitem__(self, index):
        scan = self.all_datas[index]

        complete_img_filepaths = scan["img_filepaths"]
        img_filepaths = complete_img_filepaths[:self.num_frames]

        images = []
        near, far = [], []
        for i in range(len(img_filepaths)):
            rgb = self.load_rgb(img_filepaths[i])
            images.append(self.to_tensor(rgb))
            near.append(torch.tensor(0.1, dtype=torch.float32))
            far.append(torch.tensor(8., dtype=torch.float32))
            
        context_ids, target_ids = list(range(2)), list(range(self.num_frames))

        data = {
            "context": {
                "image": torch.stack([images[i] for i in context_ids]),
                "near": torch.stack([near[i] for i in context_ids]),
                "far": torch.stack([far[i] for i in context_ids]),
                "index": torch.from_numpy(np.arange(len(context_ids))),
            },
            "target": {
                "image": torch.stack([images[i] for i in target_ids]),
                "near": torch.stack([near[i] for i in target_ids]),
                "far": torch.stack([far[i] for i in target_ids]),
                "index": torch.from_numpy(np.arange(len(target_ids))),
            },
            "scene": str(scan["scene_id"]),
        }

        return data

    def __len__(self):
        return len(self.all_datas)
