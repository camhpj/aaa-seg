import glob
import logging
import os
import sys
from typing import List, Tuple

import click
import cv2
import nrrd
import numpy as np
import pandas as pd

from src.preprocessing import (
    find_ct_bounding_rect,
    find_ct_outer_contour_mask,
    window_ct,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def find_folders() -> List[str]:
    tmp1 = glob.glob("data/Dongyang/*")
    tmp2 = glob.glob("data/KiTS/*")
    tmp3 = glob.glob("data/Rider/*")
    folders = tmp1 + tmp2 + tmp3
    return folders


def read_volume_and_segmentation(folder: str) -> Tuple[np.ndarray, np.ndarray]:
    pid = folder.split("/")[-1]
    vol, _ = nrrd.read(f"{folder}/{pid}.nrrd")
    seg, _ = nrrd.read(f"{folder}/{pid}.seg.nrrd")
    return (vol, seg)


def get_slices_with_mask(seg: np.ndarray) -> List[int]:
    idx = []
    for i in range(seg.shape[-1]):
        if seg[:, :, i].sum() != 0:
            idx.append(i)
    return idx


def save_slices_to_png(
    idx: List[int], folder: str, vol: np.ndarray, seg: np.ndarray
) -> List[Tuple[str, str, int, int, int, int, int]]:
    pid = folder.split("/")[-1]
    metadata = []
    for i in idx:
        img_path = f"data/images/{pid}_{i}.png"
        mask_path = f"data/masks/{pid}_{i}.png"

        img = vol[:, :, i].T
        mask = seg[:, :, i].T

        if folder.split("/")[-1][0] == "D":
            x, y, w, h = find_ct_bounding_rect(img)
            img_cropped = img[y : y + h + 1, x : x + w + 1]
            mask_cropped = mask[y : y + h + 1, x : x + w + 1]
        else:
            img_cropped = img
            mask_cropped = mask
        cv2.imwrite(img_path, img_cropped)
        cv2.imwrite(mask_path, mask_cropped)

        ct_contour_mask = find_ct_outer_contour_mask(img_cropped)

        pixel_sum = np.sum(img_cropped[ct_contour_mask > 0])
        pixel_squared_sum = np.sum(img_cropped[ct_contour_mask > 0].astype(np.int32) ** 2)
        pixel_count = np.sum(ct_contour_mask > 0)

        metadata.append((
            img_path,
            mask_path,
            img_cropped.shape[0],
            img_cropped.shape[1],
            pixel_sum,
            pixel_squared_sum,
            pixel_count,
        ))

    return metadata


def calculate_dataset_properties(df: pd.DataFrame) -> Tuple[float, float]:
    sum_of_pixels = df["X"].sum()
    sum_of_squares = df["X^2"].sum()
    total_pixels = df["N"].sum()
    mean = sum_of_pixels / total_pixels
    var = (sum_of_squares / total_pixels) - (mean ** 2)
    std = var ** 0.5
    return (mean, std)


@click.command()
def main() -> None:
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)

    frames = []
    folders = find_folders()
    logger.info(f"Found {len(folders)} folders")
    for f in folders[0:1]:
        if "(AD)" in f or "(AAA)" in f:
            continue

        logger.info(f"Reading nrrd and seg.nrrd files from {f}")
        vol, seg = read_volume_and_segmentation(f)
        vol_scaled = window_ct(vol, 350, 40, rescale=True)
        seg_scaled = (seg * 255).astype(np.uint8)
        idx = get_slices_with_mask(seg_scaled)
        logger.info(f"Saving slices from {f}")
        metadata = save_slices_to_png(
            idx, f, vol_scaled, seg_scaled
        )
        img_path, mask_path, height, width, pixel_sum, pixel_sqare_sum, pixel_count = list(zip(*metadata))
        frames.append(
            pd.DataFrame(
                {
                    "img": img_path,
                    "mask": mask_path,
                    "height": height,
                    "width": width,
                    "X": pixel_sum,
                    "X^2": pixel_sqare_sum,
                    "N": pixel_count,
                }
            )
        )

    path = "data/all.csv"
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    logger.info(f"Dataset manifest saved to {path}")

    props_path = "data/props.csv"
    mean, stdev = calculate_dataset_properties(df)
    props = pd.DataFrame({"mean": [mean], "stdev": [stdev]})
    props.to_csv(props_path, index=False)
    logger.info(f"Dataset properties saved to {props_path}")
    logger.info("Done")


if __name__ == "__main__":
    main()
