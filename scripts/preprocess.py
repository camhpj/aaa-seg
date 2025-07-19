import glob
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import click
import nrrd
import numpy as np
import pandas as pd
import tqdm

from src.data import AxialSlice, Volume, process_nrrd_metadata

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def read_nrrd(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    arr, meta = nrrd.read(path)
    meta = process_nrrd_metadata(meta)
    return (arr, meta)


def process_ct_scan(ct_dir: str) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
    path = list(glob.glob(f"{ct_dir}/*[!seg].nrrd"))[0]
    raw_image, img_meta = read_nrrd(path)
    raw_seg, seg_meta = read_nrrd(path.replace(".nrrd", ".seg.nrrd"))
    
    vol = Volume(raw_image, raw_seg, img_meta, seg_meta)
    # TODO: resample volume
    vol.window_volume(window=400, level=40, rescale=True)
    # TODO: denoise volume

    out = []
    for i, (img, mask) in enumerate(vol):
        img_path = f"{ct_dir}/img_{i}.npy"
        mask_path = f"{ct_dir}/mask_{i}.npy"
        axial_slice = AxialSlice(img, mask)
        axial_slice.process_slice()
        axial_slice.save(img_path, mask_path)

        props = axial_slice.props
        props.update({"img_path": img_path, "mask_path": mask_path})
        out.append(props)
    return out


def calculate_dataset_properties(df: pd.DataFrame) -> Tuple[float, float]:
    sum_of_pixels = df["X"].sum()
    sum_of_squares = df["X^2"].sum()
    total_pixels = df["N"].sum()
    mean = sum_of_pixels / total_pixels
    var = (sum_of_squares / total_pixels) - (mean ** 2)
    std = var ** 0.5
    return (mean, std)


@click.command
@click.argument("datadir", type=str)
def main(datadir: str) -> None:
    folders = os.listdir(datadir)

    logger.info("Processing dataset...")
    dataset_props = []
    for f in tqdm(folders, total=len(folders)):
        tmp = process_ct_scan(f)
        dataset_props.extend(tmp)

    path = "data/all.csv"
    df = pd.concat(dataset_props, ignore_index=True)
    df.to_csv(path, index=False)
    logger.info(f"Dataset manifest saved to {path}")

    props_path = "data/props.csv"
    mean, stdev = calculate_dataset_properties(df)
    props = pd.DataFrame({"mean": [mean], "stdev": [stdev]})
    props.to_csv(props_path, index=False)
    logger.info(f"Dataset properties saved to {props_path}")
    logger.info("Done")
