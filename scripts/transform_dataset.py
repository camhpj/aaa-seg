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

from src.preprocessing import find_ct_region, scale_img, scale_mask

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
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


def save_slices_to_png(idx: List[int], folder: str, vol: np.ndarray, seg: np.ndarray) -> Tuple[List[str], List[str]]:
    pid = folder.split("/")[-1]
    img_path_list, mask_path_list = [], []
    for i in idx:
        img_path = f"data/images/{pid}_{i}.png"
        img_scaled = scale_img(vol[:, :, i].T)
        if folder.split("/")[-1][0] == "D":
            x, y, w, h = find_ct_region(img_scaled)
            img_cropped = img_scaled[y:y+h+1, x:x+w+1]
        else:
            img_cropped = img_scaled
        cv2.imwrite(img_path, img_cropped)
        img_path_list.append(img_path)

        mask_path = f"data/masks/{pid}_{i}.png"
        mask_scaled = scale_mask(seg[:, :, i].T)
        if folder.split("/")[-1][0] == "D":
            x, y, w, h = find_ct_region(img_scaled)
            mask_cropped = mask_scaled[y:y+h+1, x:x+w+1]
        else:
            mask_cropped = mask_scaled
        cv2.imwrite(mask_path, mask_cropped)
        mask_path_list.append(mask_path)

    return (img_path_list, mask_path_list)


@click.command()
def main() -> None:
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)

    frames = []
    folders = find_folders()
    logger.info(f"Found {len(folders)} folders")
    for f in folders:
        if "(AD)" in f or "(AAA)" in f:
            continue

        logger.info(f"Reading nrrd and seg.nrrd files from {f}")
        vol, seg = read_volume_and_segmentation(f)
        idx = get_slices_with_mask(seg)
        logger.info(f"Saving slices from {f}")
        img_path_list, mask_path_list = save_slices_to_png(idx, f, vol, seg)
        frames.append(pd.DataFrame({
            "img": img_path_list,
            "mask": mask_path_list,
        }))

    df = pd.concat(frames, ignore_index=True)
    path = "data/all.csv"
    df.to_csv(path, index=False)
    logger.info(f"Dataset manifest saved to {path}")
    logger.info("Done")


if __name__ == "__main__":
    main()
