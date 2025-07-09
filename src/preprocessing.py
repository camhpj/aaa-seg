from typing import Tuple

import cv2
import numpy as np


def scale_img(arr: np.ndarray) -> np.ndarray:
    arr -= arr.min()
    arr = np.divide(arr, np.clip(arr.max(), a_min=1e-8, a_max=None))
    arr *= 255
    return arr.astype(np.uint8)


def scale_mask(arr: np.ndarray) -> np.ndarray:
    return (arr * 255).astype(np.uint8)


def find_ct_region(img: np.ndarray) -> Tuple[int, int, int, int]:
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(i) for i in contours]
    x, y, w, h = cv2.boundingRect(contours[np.argmax(areas)])
    return (x, y, w, h)
