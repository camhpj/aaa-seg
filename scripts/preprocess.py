import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import click
import cv2
import nrrd
import numpy as np
import pandas as pd
import tqdm

from src.data import AxialSlice, Volume, process_nrrd_metadata, validate_ct_metadata

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
    is_equal = validate_ct_metadata(img_meta, seg_meta)

    if not is_equal:
        raise ValueError(f"Metadata for {ct_dir} is not equal")
    
    vol = Volume(raw_image, raw_seg, img_meta)
    # TODO: resample volume
    vol.window_volume(window=400, level=40, rescale=True)
    # TODO: denoise volume

    processed_data = []
    for i, (img, mask) in enumerate(vol):
        img_path = f"{ct_dir}/img_{i}.npy"
        mask_path = f"{ct_dir}/mask_{i}.npy"
        axial_slice = AxialSlice(img, mask)
        axial_slice.process_slice()
        axial_slice.save(img_path, mask_path)

        props = axial_slice.props
        props.update({"img_path": img_path, "mask_path": mask_path})
        processed_data.append(props)
    return processed_data


@click.command
@click.argument("datadir", type=str)
def main(datadir: str) -> None:
    folders = os.listdir(datadir)

    data = []
    for f in folders:
        tmp = process_ct_scan(f)
        data.extend(tmp)
