import os
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
from functools import lru_cache
from config import Config as Config

config = Config()

SFREQ = config.sfrq  # 采样率 500
WINDOW_SEC = config.time_window_length  # 窗口时长 10
WIN = SFREQ * WINDOW_SEC  # 一个窗口的数据点个数 5000
OVERLAP = config.time_window_overlap  # 重叠率 0.75
STRIDE = int(WIN * (1.0 - OVERLAP))  # 步长 1250

LABEL_MAP = {"HC": 0, "PD": 1}


class _BaseWindowDataset(Dataset):
    def __init__(self, root_dir: str, fixed_length: int):
        self.root_dir = root_dir
        self.fixed_length = fixed_length  # e.g., 90000 or 60000

        self.samples = []  # (fpath, label, start)

        n_win = int((self.fixed_length - WIN) // STRIDE) + 1

        for cls_name in ["HC", "PD"]:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue

            label = LABEL_MAP[cls_name]
            for fname in os.listdir(cls_dir):
                if not fname.endswith(".mat"):
                    continue
                fpath = os.path.join(cls_dir, fname)
                for w in range(n_win):
                    start = w * STRIDE
                    self.samples.append((fpath, label, start))

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=512)  # 你机器内存够就开大点
    def _load_file(self, fpath):
        mat = scio.loadmat(fpath, squeeze_me=True)
        x = mat["data"].astype(np.float32)  # [C, L]
        return x

    def __getitem__(self, idx):
        fpath, label, start = self.samples[idx]
        x = self._load_file(fpath)  # 不再反复读盘
        x_win = x[:, start:start + WIN]
        return torch.from_numpy(x_win), torch.tensor(label, dtype=torch.long)


class D2778S(_BaseWindowDataset):
    def __init__(self, root_dir: str):
        super().__init__(root_dir=root_dir, fixed_length=90000)


class D3940S(_BaseWindowDataset):
    def __init__(self, root_dir: str):
        super().__init__(root_dir=root_dir, fixed_length=90000)


class D4584S(_BaseWindowDataset):
    def __init__(self, root_dir: str):
        super().__init__(root_dir=root_dir, fixed_length=60000)
