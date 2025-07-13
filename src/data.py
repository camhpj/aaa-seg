from functools import cached_property
from typing import Any, Dict, Optional,  Sequence, Tuple, OrderedDict, Literal

import cv2
import nrrd
import numpy as np
import SimpleITK as sitk


# TODO: validate volume and mask have the same physical properties
def load_volume_and_segmentation(
        path: str
    ) -> Tuple[np.ndarray, OrderedDict, np.ndarray | None, OrderedDict | None]:
    vol, vol_head = nrrd.read(path)
    try:
        seg, seg_head = nrrd.read(path.replace(".nrrd", ".seg.nrrd"))
    except Exception as e:
        print(f"Error loading segmentation: {e}")
        seg, seg_head = None, None
    return (vol, vol_head, seg, seg_head)


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


class Volume:
    def __init__(self, arr: np.ndarray, metadata: Dict[str, Any]) -> None:
        self.arr = arr
        self.metadata = metadata

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.arr.shape

    @cached_property
    def contours(
        self,
        thresh_kwargs: Optional[Dict[str, Any]] = None,
        contour_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Sequence[cv2.typing.MatLike]:
        if not thresh_kwargs:
            thresh_kwargs = {"thresh": 1, "maxval": 255, "type": cv2.THRESH_BINARY}
        if not contour_kwargs:
            contour_kwargs = {"mode": cv2.RETR_EXTERNAL, "method": cv2.CHAIN_APPROX_SIMPLE}
        _, thresh = cv2.threshold(self.arr, **thresh_kwargs)
        contours, _ = cv2.findContours(thresh, **contour_kwargs)
        return contours

    @cached_property
    def largest_contour(self) -> cv2.typing.MatLike:
        largest_contour = max(self.contours, key=cv2.contourArea)
        return largest_contour

    @cached_property
    def tissue_mask(self) -> np.ndarray:
        mask = np.zeros(self.arr.shape, np.uint8)
        cv2.drawContours(mask, [self.largest_contour], -1, 255, thickness=cv2.FILLED)
        return mask


class AxialSlice:
    def __init__(self, img: Volume, mask: np.ndarray) -> None:
        self.img = img
        self.mask = mask
        self.tissue_mask = None

    def crop_to_tissue_region(self) -> None:
        x, y, w, h = cv2.boundingRect(self.img.largest_contour)
        img_cropped = self.img[y : y + h, x : x + w]
        mask_cropped = self.mask[y : y + h, x : x + w]
        self.img = img_cropped
        self.mask = mask_cropped

    def remove_background(self) -> None:
        self.tissue_mask = self.img.tissue_mask
        self.img[self.tissue_mask == 0] = 0

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
        size = self.img.metadata["size"]
        origin = self.img.metadata["origin"]
        spacing = self.img.metadata["spacing"]
        direction = self.img.metadata["direction"]

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
