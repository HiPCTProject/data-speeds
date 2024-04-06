# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test how fast different JPEG2000 readers are

# %%
import os
from pathlib import Path
import tempfile
import time

# Has to go before cv2 import
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'

import cv2
import glymur
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

# %%
dtype = np.uint16


# %% [markdown]
# ## Function to time `glymur`

# %%
def time_glymur(size, cratio=10) -> tuple[float, float]:
    """
    Time saving a JPEG2000 image. Includes time to create the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data = np.random.randint(low=0, high=2**16, size=(size, size), dtype=dtype)

        start = time.time()
        jp2 = glymur.Jp2k(temp_path / f"{size}.jp2", cratios=[cratio])
        jp2[:] = data
        write_time = time.time() - start

        start = time.time()
        jp2 = glymur.Jp2k(temp_path / f"{size}.jp2")
        data = jp2[:]
        read_time = time.time() - start

        return write_time, read_time


# %% [markdown]
# ## Function to time `cv2`

# %%
def time_cv2(size, cratio=10) -> tuple[float, float]:
    """
    Time saving a JPEG2000 image. Includes time to create the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data = np.random.randint(low=0, high=2**16, size=(size, size), dtype=dtype)
        fpath = temp_path / f"{size}.jp2"

        start = time.time()
        cv2.imwrite(str(fpath), data, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
        write_time = time.time() - start

        start = time.time()
        image = np.array(cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED))
        read_time = time.time() - start

        return write_time, read_time


# %%
def time_sitk(size, cratio=10) -> tuple[float, float]:
    """
    Time saving a JPEG2000 image. Includes time to create the file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data = np.random.randint(low=0, high=2**16, size=(size, size), dtype=dtype)
        fpath = temp_path / f"{size}.jp2"

        start = time.time()
        cv2.imwrite(str(fpath), data, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100])
        # SimpleITK crashses when writing a file...
        write_time = np.nan

        start = time.time()
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(fpath)))
        read_time = time.time() - start

        return write_time, read_time


# %% [markdown]
# ## Run benchmarks

# %%
results = {}
for fmt, time_func in zip(['glymur', 'opencv', 'SimpleITK'][::-1], [time_glymur, time_cv2, time_sitk][::-1]):
    print(fmt)
    npixels = []
    write_times = []
    read_times = []

    for i in range(6, 13):
        size = 2**i
        npixels.append(size**2)
        write_time, read_time = time_func(size)
        write_times.append(write_time)
        read_times.append(read_time)

    df = pd.DataFrame({"npix": npixels, 'write_time': write_times, 'read_time': read_times})
    results[fmt] = df

# %%
fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=True, constrained_layout=True)

for fmt in ['glymur', 'opencv', 'SimpleITK']:
    df = results[fmt]

    for ax, res in zip(axs, ['read', 'write']):
        write_speed = df['npix'] * 2 / 1e6 / df[f'{res}_time']
        ax.plot(df['npix'], write_speed, marker='o', label=fmt)
        ax.set_title(f"{res.capitalize()}")


for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of pixels")
    ax.yaxis.grid(which="both", linewidth=0.7, alpha=1)

axs[0].set_ylabel("MB/s\n", rotation=0, size=12)
axs[0].set_ylim(1, 100)
axs[0].legend()
axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: str(int(x))))

plt.show()
