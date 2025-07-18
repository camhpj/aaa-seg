from typing import (
    Any,
    Dict,
    Generator,
    Tuple,
)

import cv2
import numpy as np
import SimpleITK as sitk

from src.preprocessing import (
    create_itk_image_from_array,
    create_itk_resampler,
    get_contours,
    get_largest_contour,
)


class AxialSlice:
    def __init__(self, img: np.ndarray, mask: np.ndarray) -> None:
        """Data processing class for a single CT slice (img and mask).

        Args:
            img (np.ndarray): CT slice (2D).
            mask (np.ndarray): Segmentation mask for slice.
        """
        if len(img.shape) != 2:
            raise ValueError(f"img should have two dimensions, but instead has {len(img.shape)} dimensions")
        if img.shape != mask.shape:
            raise ValueError(f"img and mask must be the same shape but are {img.shape} and {mask.shape}")
        self.img = img
        self.mask = mask
        self.tissue_mask = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        """img (and mask) shape"""
        return self.img.shape

    @property
    def props(self) -> Dict[str, Any]:
        """Returns various img properties as a dict. If self.process_slice()
        has been called then properties will be calculated only on non-background pixels.

        pixel_sum = sum of all pixel values
        pixel_square_sum = sum of all pixel values squared
        pixel_count = number of pixels in the image
        """
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
        """Get the countours from img array and return the largest contour.

        Returns:
            np.ndarray: cv2 contour.
        """
        contours = get_contours(self.img)
        largest_contour = get_largest_contour(contours)
        return largest_contour

    def process_slice(self) -> None:
        """Process CT slice. Removes background and crops image to tissue region."""
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
        """Save img and mask array as numpy array.

        Args:
            img_path (str): img file path.
            mask_path (str): mask file path.
        """
        np.save(img_path, self.img)
        np.save(mask_path, self.mask)


class Volume:
    def __init__(self, img: np.ndarray, mask: np.ndarray, img_metadata: Dict[str, Any], mask_metadata: Dict[str, Any]) -> None:
        """Data processing class for 3D ct volume. Supports various methods for processing
        the 3D volume. 

        Args:
            img (np.ndarray): CT volume image.
            mask (np.ndarray): Segmentation mask for volume.
            metadata (Dict[str, Any]): _description_
        """
        if len(img.shape) != 3:
            raise ValueError(f"img should have three dimensions, but instead has {len(img.shape)} dimensions")
        if img.shape != mask.shape:
            raise ValueError(f"img and mask must be the same shape but are {img.shape} and {mask.shape}")
        self.img = img.astype(np.float32)
        self.mask = mask.astype(np.float32)
        if not Volume._validate_ct_metadata(img_metadata, mask_metadata):
            raise ValueError("img and mask metadata is not equal")
        self.metadata = img_metadata

    def __iter__(self) -> Generator[Tuple[np.ndarray], None, None]:
        """Yield slice from img and mask arrays.

        Yields:
            Generator[Tuple[np.ndarray], None, None]: Slice from img and mask.
        """
        for i in range(self.img.shape[-1]):
            img = self.img[:, :, i]
            mask = self.mask[:, :, i]
            yield (img, mask)

    @classmethod
    def _validate_ct_metadata(cls, vol_meta: Dict[str, Any], seg_meta: Dict[str, Any]) -> bool:
        """Confirm that volume and segmentation metadata are equal."""
        size = seg_meta["size"] == vol_meta["size"]
        spacing = seg_meta["spacing"] == vol_meta["spacing"]
        origin = seg_meta["origin"] == vol_meta["origin"]
        direction = seg_meta["direction"] == vol_meta["direction"]
        dimension = seg_meta["direction"] == vol_meta["direction"]
        return size and spacing and origin and direction and dimension

    def window_volume(self, window: int, level: int, rescale: bool = False) -> None:
        """Perform CT windowing on the img array inplace.

        Args:
            window (int): Width of window.
            level (int): Level of window.
            rescale (bool, optional): Rescale data uint8 (instead of 0-1). Defaults to False.
        """
        img = self.img
        min_ = level - window // 2
        max_ = level + window // 2
        img[img < min_] = min_
        img[img > max_] = max_
        if rescale:
            img = (img - min_) / (max_ - min_) * 255.0
            self.img = img.astype(np.uint8)
        self.img = img

    def resample(self, new_spacing: Tuple[float, float, float]) -> None:
        """Perform 3D resampling on img array and then apply the same resampling to
        the mask volume.

        Args:
            new_spacing (Tuple[float, float, float]): Spacing in mm for resampled volume.
        """
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
