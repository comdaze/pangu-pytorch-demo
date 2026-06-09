# SPDX-License-Identifier: Apache-2.0
"""
Custom CorrDiff dataset for the WRF downscaling data (train_data.tar).

NetCDF layout (groups):
  input      : low-res fields  (sample, y_lr, x_lr)  e.g. 14x14  (25 km)
  output     : high-res fields (sample, y_hr, x_hr)  e.g. 70x70  ( 5 km)
  invariant  : lon/lat grids for both resolutions (not used as channels here)
  top-level  : time (sample,), coord (sample, 2)

Variables (both input and output): U/V at 1000, 850, 10m, 100m -> 8 wind channels.

Pipeline per sample:
  1. upsample low-res input by integer factor (y_hr // y_lr) with bilinear
     extrapolation (reuses the CorrDiff-mini zoom kernel),
  2. center-crop input(upsampled) and output to `crop` x `crop` so the spatial
     size is divisible by the UNet downsampling factor (default 64, matching the
     CorrDiff-mini model which needs sizes divisible by 16),
  3. normalize with per-variable mean/std from stats.json.

Returns (output_hr, input_upsampled) like the other CorrDiff datasets.
"""
import datetime
import json
from typing import List, Tuple, Union

import numpy as np
import xarray as xr

from datasets.base import ChannelMetadata, DownscalingDataset
from datasets.hrrrmini import _zoom_extrapolate
from helpers.train_helpers import _convert_datetime_to_cftime


WIND_VARS = ["U1000", "U850", "U10m", "U100m", "V1000", "V850", "V10m", "V100m"]


def _load_group(data_path, group, variables):
    with xr.open_dataset(data_path, group=group) as ds:
        if variables is None:
            variables = list(ds.keys())
        data = np.stack([ds[v].values.astype(np.float32) for v in variables], axis=1)
    return data, variables  # (sample, C, H, W)


class WRFDataset(DownscalingDataset):
    """CorrDiff downscaling dataset for WRF paired (coarse, fine) wind fields."""

    def __init__(
        self,
        data_path: str,
        stats_path: str,
        input_variables: Union[List[str], None] = None,
        output_variables: Union[List[str], None] = None,
        invariant_variables: Union[List[str], None] = None,  # unused; kept for API
        crop: int = 64,
    ):
        self.input, self.input_variables = _load_group(data_path, "input",
                                                       input_variables or WIND_VARS)
        self.output, self.output_variables = _load_group(data_path, "output",
                                                         output_variables or WIND_VARS)
        with xr.open_dataset(data_path) as ds:
            self.times = np.array(ds["time"])

        self.upsample_factor = self.output.shape[-1] // self.input.shape[-1]
        self.crop = crop
        hr = self.output.shape[-1]
        self._c0 = (hr - crop) // 2  # center-crop start (same for H and W)
        self.img_shape = (crop, crop)

        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.input_mean, self.input_std = self._stats(stats, self.input_variables, "input")
        self.output_mean, self.output_std = self._stats(stats, self.output_variables, "output")

    @staticmethod
    def _stats(stats, variables, group):
        mean = np.array([stats[group][v]["mean"] for v in variables])[:, None, None].astype(np.float32)
        std = np.array([stats[group][v]["std"] for v in variables])[:, None, None].astype(np.float32)
        return mean, std

    def _upsample(self, x):
        f = self.upsample_factor
        y = np.empty((x.shape[0], x.shape[1] * f, x.shape[2] * f), dtype=np.float32)
        _zoom_extrapolate(x, y, f)
        return y

    def _crop(self, x):
        c0, c = self._c0, self.crop
        return x[:, c0:c0 + c, c0:c0 + c]

    def __getitem__(self, idx):
        x = self._crop(self._upsample(self.input[idx].copy()))
        y = self._crop(self.output[idx].copy())
        return self.normalize_output(y), self.normalize_input(x)

    def __len__(self):
        return self.input.shape[0]

    def longitude(self) -> np.ndarray:
        return np.full(self.img_shape, np.nan)

    def latitude(self) -> np.ndarray:
        return np.full(self.img_shape, np.nan)

    def input_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.input_variables]

    def output_channels(self) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v) for v in self.output_variables]

    def time(self) -> List:
        out = []
        for t in self.times:
            s = str(t)
            try:
                dt = datetime.datetime.strptime(s, "%Y%m%d%H")
            except ValueError:
                try:
                    dt = datetime.datetime.fromisoformat(s)
                except ValueError:
                    dt = datetime.datetime(2018, 1, 1)
            out.append(_convert_datetime_to_cftime(dt))
        return out

    def image_shape(self) -> Tuple[int, int]:
        return self.img_shape

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return (x - self.input_mean) / self.input_std

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        return x * self.input_std + self.input_mean

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return (x - self.output_mean) / self.output_std

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        return x * self.output_std + self.output_mean
