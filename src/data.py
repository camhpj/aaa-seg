import json
from functools import cached_property
from typing import (
    Any,
    Dict,
    Generator,
    Literal,
    Optional,
    OrderedDict,
    Self,
    Sequence,
    Tuple,
)

import cv2
import numpy as np
import SimpleITK as sitk


def process_nrrd_metadata(metadata: OrderedDict[str, Any]) -> Dict[str, Any]:
    size = metadata["sizes"][::-1].tolist()
    dimension = metadata["dimension"]
    origin: np.ndarray = metadata["space origin"]
    directions: np.ndarray = metadata["space directions"]
    spacing: np.ndarray = np.linalg.norm(directions, axis=0)
    directions_norm = directions / spacing
    return {
        "size": size,
        "spacing": spacing,
        "origin": origin,
        "direction": directions_norm,
        "dimension": dimension,
    }


def validate_ct_metadata(vol_meta: Dict[str, Any], seg_meta: Dict[str, Any]) -> bool:
    size = seg_meta["size"] == vol_meta["size"]
    spacing = seg_meta["spacing"] == vol_meta["spacing"]
    origin = seg_meta["origin"] == vol_meta["origin"]
    direction = seg_meta["direction"] == vol_meta["direction"]
    dimension = seg_meta["direction"] == vol_meta["direction"]
    return size and spacing and origin and direction and dimension


def create_itk_image_from_array(
        arr: np.ndarray,
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        direction: Tuple[float, float, float, float, float, float, float, float, float],
    ) -> sitk.Image:
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(origin)
    img.SetSpacing(spacing)
    img.SetDirection(direction)
    return img


def create_itk_resampler(
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float, float, float, float, float, float, float],
        new_spacing: Tuple[float, float, float],
        new_size: Tuple[int, int, int],
        default_pixel_value: int,
        interpolation: Literal["linear", "nn"],
    ) -> sitk.ResampleImageFilter:
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)

    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetDefaultPixelValue(default_pixel_value)
    match interpolation:
        case "linear":
            resampler.SetInterpolator(sitk.sitkLinear)
        case "nn":
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        case _:
            raise ValueError(f"Invalid interpolation method {interpolation}.")
    return resampler


def get_contours(
    img: np.ndarray,
    thresh_kwargs: Optional[Dict[str, Any]] = None,
    contour_kwargs: Optional[Dict[str, Any]] = None,
) -> Sequence[cv2.typing.MatLike]:
    if not thresh_kwargs:
        thresh_kwargs = {"thresh": 1, "maxval": 255, "type": cv2.THRESH_BINARY}
    if not contour_kwargs:
        contour_kwargs = {"mode": cv2.RETR_EXTERNAL, "method": cv2.CHAIN_APPROX_SIMPLE}
    _, thresh = cv2.threshold(img, **thresh_kwargs)
    contours, _ = cv2.findContours(thresh, **contour_kwargs)
    return contours


def get_largest_contour(contours: Sequence[cv2.typing.MatLike]) -> cv2.typing.MatLike:
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour


class AxialSlice:
    def __init__(self, img: np.ndarray, mask: np.ndarray) -> None:
        self.img = img
        self.mask = mask
        self.tissue_mask = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.img.shape

    @property
    def props(self) -> Dict[str, Any]:
        if self.tissue_mask:
            pixel_sum = np.sum(self.img[self.tissue_mask > 0])
            pixel_square_sum = np.sum(self.img[self.tissue_mask > 0].astype(np.int32) ** 2)
            pixel_count = np.sum(self.tissue_mask > 0)
        else:
            pixel_sum = np.sum(self.img)
            pixel_square_sum = np.sum(self.img.astype(np.int32) ** 2)
            pixel_count = self.img.shape[0] * self.img.shape[1]
        return {
            "height": self.img.shape[0],
            "width": self.img.shape[1],
            "pixel_sum": pixel_sum,
            "pixel_square_sum": pixel_square_sum,
            "pixel_count": pixel_count,
        }

    def _get_largest_contour(self) -> np.ndarray:
        contours = get_contours(self.img)
        largest_contour = get_largest_contour(contours)
        return largest_contour

    def process_slice(self) -> None:
        largest_contour = self._get_largest_contour()

        # remove background
        mask = np.zeros(self.img.shape, np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        self.img[mask == 0] = 0

        # crop to tissue region
        x, y, w, h = cv2.boundingRect(largest_contour)
        self.img = self.img[y : y + h, x : x + w]
        self.mask = self.mask[y : y + h, x : x + w]

        # save cropped mask
        self.tissue_mask = mask[y: y + h, x: x + w]

    def save_slice(self, img_path: str, mask_path: str) -> None:
        np.save(img_path, self.img)
        np.save(mask_path, self.mask)


class Volume:
    def __init__(self, img: np.ndarray, mask: np.ndarray, metadata: Dict[str, Any]) -> None:
        self.img = img
        self.mask = mask
        self.metadata = metadata

    def __iter__(self) -> Generator[Tuple[np.ndarray], None, None]:
        for i in range(self.img.shape[-1]):
            img = self.img[:, :, i]
            mask = self.mask[:, :, i]
            yield (img, mask)

    def window_volume(self, window: int, level: int, rescale: bool = False) -> None:
        min_ = level - window // 2
        max_ = level + window // 2
        self.img[self.img < min_] = min_
        self.img[self.img > max_] = max_
        if rescale:
            img = (self.img - min_) / (max_ - min_) * 255.0
            return img.astype(np.uint8)
        return img

    def resample(self, new_spacing: Tuple[float, float, float]) -> None:
        # get physical properties
        size = self.metadata["size"]
        origin = self.metadata["origin"]
        spacing = self.metadata["spacing"]
        direction = self.metadata["direction"]

        new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(size, spacing, new_spacing)]

        img = create_itk_image_from_array(self.img.arr, origin, spacing, direction)
        mask = create_itk_image_from_array(self.mask, origin, spacing, direction)

        img_resampler = create_itk_resampler(origin, spacing, direction, new_spacing, new_size, -1000, "linear")
        mask_resampler = create_itk_resampler(origin, spacing, direction, new_spacing, new_size, 0, "nn")

        resampled_image: sitk.Image = img_resampler.Execute(img)
        resampled_mask: sitk.Image = mask_resampler.Execute(mask)

        arr = sitk.GetArrayFromImage(resampled_image)
        metadata = {
            "size": resampled_image.GetSize(),
            "spacing": resampled_image.GetSpacing(),
            "origin": resampled_image.GetOrigin(),
            "direction": resampled_image.GetDirection(),
            "dimension": resampled_image.GetDimension(),
        }
        self.img = Volume(arr, metadata)

        mask = sitk.GetArrayFromImage(resampled_mask)
        metadata = {
            "size": resampled_mask.GetSize(),
            "spacing": resampled_mask.GetSpacing(),
            "origin": resampled_mask.GetOrigin(),
            "direction": resampled_mask.GetDirection(),
            "dimension": resampled_mask.GetDimension(),
        }
        self.mask = mask
