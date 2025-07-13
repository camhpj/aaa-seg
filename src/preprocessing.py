from typing import Tuple

import cv2
import numpy as np


def window_ct(
    img: np.ndarray, window: int, level: int, rescale: bool = True
) -> np.ndarray[np.uint8]:
    img_min = level - window // 2
    img_max = level + window // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    if rescale:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)


def normalize_ct_slice(
    img: np.ndarray, mean: float, stdev: float
) -> np.ndarray[np.float32]:
    img = img - mean / stdev
    return img.astype(np.float32)


def find_ct_bounding_rect(img: np.ndarray) -> Tuple[int, int, int, int]:
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)


def find_ct_outer_contour_mask(img: np.ndarray) -> None:
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    return mask
