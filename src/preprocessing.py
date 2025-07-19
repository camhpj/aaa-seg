from typing import Any, Dict, Literal, Optional, OrderedDict, Sequence, Tuple

import cv2
import numpy as np
import SimpleITK as sitk


def process_nrrd_metadata(metadata: OrderedDict[str, Any]) -> Dict[str, Any]:
    """Standardize nrrd metadata. Specific for AAA dataset."""
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
    # convert ndarray into itk Image() class instance
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
    # itk resampler
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
    """Get contours in image.

    Args:
        img (np.ndarray): 2D image.
        thresh_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for cv2.threshold() . Defaults to None.
        contour_kwargs (Optional[Dict[str, Any]], optional): Keyword arguments for cv2.findcontour(). Defaults to None.

    Returns:
        Sequence[cv2.typing.MatLike]: Sequence of contours.
    """
    if not thresh_kwargs:
        thresh_kwargs = {"thresh": 1, "maxval": 255, "type": cv2.THRESH_BINARY}
    if not contour_kwargs:
        contour_kwargs = {"mode": cv2.RETR_EXTERNAL, "method": cv2.CHAIN_APPROX_SIMPLE}
    _, thresh = cv2.threshold(img, **thresh_kwargs)
    contours, _ = cv2.findContours(thresh, **contour_kwargs)
    return contours


def get_largest_contour(contours: Sequence[cv2.typing.MatLike]) -> cv2.typing.MatLike:
    """Return the contours with the largest area."""
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour
