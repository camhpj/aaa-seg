


# Ripped from MONAI

##HAUSDORFF
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction, get_edge_surface_distance, ignore_background, prepare_spacing
from monai.utils import MetricReduction, convert_data_type

from .metric import CumulativeIterationMetric

__all__ = ["HausdorffDistanceMetric", "compute_hausdorff_distance"]


class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: float | None = None,
        directed: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.
            kwargs: additional parameters, e.g. ``spacing`` should be passed to correctly compute the metric.
                ``spacing``: spacing of pixel (or voxel). This parameter is relevant only
                if ``distance_metric`` is set to ``"euclidean"``.
                If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
                the length of the sequence must be equal to the image dimensions.
                This spacing will be used for all images in the batch.
                If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
                If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
                else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
                for all images in batch. Defaults to ``None``.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")

        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
            spacing=kwargs.get("spacing"),
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_hausdorff_distance(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    directed: bool = False,
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
) -> torch.Tensor:
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
            the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
            If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
            If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
            else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
            for all images in batch. Defaults to ``None``.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    hd = torch.empty((batch_size, n_class), dtype=torch.float, device=y_pred.device)

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        _, distances, _ = get_edge_surface_distance(
            y_pred[b, c],
            y[b, c],
            distance_metric=distance_metric,
            spacing=spacing_list[b],
            symmetric=not directed,
            class_index=c,
        )
        percentile_distances = [_compute_percentile_hausdorff_distance(d, percentile) for d in distances]
        max_distance = torch.max(torch.stack(percentile_distances))
        hd[b, c] = max_distance
    return hd


def _compute_percentile_hausdorff_distance(
    surface_distance: torch.Tensor, percentile: float | None = None
) -> torch.Tensor:
    """
    This function is used to compute the Hausdorff distance.
    """

    # for both pred and gt do not have foreground
    if surface_distance.shape == (0,):
        return torch.tensor(np.nan, dtype=torch.float, device=surface_distance.device)

    if not percentile:
        return surface_distance.max()

    if 0 <= percentile <= 100:
        return torch.quantile(surface_distance, percentile / 100)
    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")


###Surface DIstance = basic DICE for Boundary

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction, get_edge_surface_distance, ignore_background, prepare_spacing
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric


class SurfaceDiceMetric(CumulativeIterationMetric):
    """
    Computes the Normalized Surface Dice (NSD) for each batch sample and class of
    predicted segmentations `y_pred` and corresponding reference segmentations `y` according to equation :eq:`nsd`.
    This implementation is based on https://arxiv.org/abs/2111.05408 and supports 2D and 3D images.
    Be aware that by default (`use_subvoxels=False`), the computation of boundaries is different from DeepMind's
    implementation https://github.com/deepmind/surface-distance.
    In this implementation, the length/area of a segmentation boundary is
    interpreted as the number of its edge pixels. In DeepMind's implementation, the length of a segmentation boundary
    depends on the local neighborhood (cf. https://arxiv.org/abs/1809.04430).
    This issue is discussed here: https://github.com/Project-MONAI/MONAI/issues/4103.

    The class- and batch sample-wise NSD values can be aggregated with the function `aggregate`.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        class_thresholds: List of class-specific thresholds.
            The thresholds relate to the acceptable amount of deviation in the segmentation boundary in pixels.
            Each threshold needs to be a finite, non-negative number.
        include_background: Whether to include NSD computation on the first channel of the predicted output.
            Defaults to ``False``.
        distance_metric: The metric used to compute surface distances.
            One of [``"euclidean"``, ``"chessboard"``, ``"taxicab"``].
            Defaults to ``"euclidean"``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count.
            Defaults to ``False``.
            `not_nans` is the number of batch samples for which not all class-specific NSD values were nan values.
            If set to ``True``, the function `aggregate` will return both the aggregated NSD and the `not_nans` count.
            If set to ``False``, `aggregate` will only return the aggregated NSD.
        use_subvoxels: Whether to use subvoxel distances. Defaults to ``False``.
    """

    def __init__(
        self,
        class_thresholds: list[float],
        include_background: bool = False,
        distance_metric: str = "euclidean",
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        use_subvoxels: bool = False,
    ) -> None:
        super().__init__()
        self.class_thresholds = class_thresholds
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.use_subvoxels = use_subvoxels

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        r"""
        Args:
            y_pred: Predicted segmentation, typically segmentation model output.
                It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference segmentation.
                It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            kwargs: additional parameters: ``spacing`` should be passed to correctly compute the metric.
                ``spacing``: spacing of pixel (or voxel). This parameter is relevant only
                if ``distance_metric`` is set to ``"euclidean"``.
                If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
                the length of the sequence must be equal to the image dimensions.
                This spacing will be used for all images in the batch.
                If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
                If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
                else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
                for all images in batch. Defaults to ``None``.
                use_subvoxels: Whether to use subvoxel distances. Defaults to ``False``.


        Returns:
            Pytorch Tensor of shape [B,C], containing the NSD values :math:`\operatorname {NSD}_{b,c}` for each batch
            index :math:`b` and class :math:`c`.
        """
        return compute_surface_dice(
            y_pred=y_pred,
            y=y,
            class_thresholds=self.class_thresholds,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            spacing=kwargs.get("spacing"),
            use_subvoxels=self.use_subvoxels,
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        r"""
        Aggregates the output of `_compute_tensor`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        Returns:
            If `get_not_nans` is set to ``True``, this function returns the aggregated NSD and the `not_nans` count.
            If `get_not_nans` is set to ``False``, this function returns only the aggregated NSD.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_surface_dice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    class_thresholds: list[float],
    include_background: bool = False,
    distance_metric: str = "euclidean",
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
    use_subvoxels: bool = False,
) -> torch.Tensor:
    r"""
    This function computes the (Normalized) Surface Dice (NSD) between the two tensors `y_pred` (referred to as
    :math:`\hat{Y}`) and `y` (referred to as :math:`Y`). This metric determines which fraction of a segmentation
    boundary is correctly predicted. A boundary element is considered correctly predicted if the closest distance to the
    reference boundary is smaller than or equal to the specified threshold related to the acceptable amount of deviation
    in pixels. The NSD is bounded between 0 and 1.

    This implementation supports multi-class tasks with an individual threshold :math:`\tau_c` for each class :math:`c`.
    The class-specific NSD for batch index :math:`b`, :math:`\operatorname {NSD}_{b,c}`, is computed using the function:

    .. math::
        \operatorname {NSD}_{b,c} \left(Y_{b,c}, \hat{Y}_{b,c}\right) = \frac{\left|\mathcal{D}_{Y_{b,c}}^{'}\right| +
        \left| \mathcal{D}_{\hat{Y}_{b,c}}^{'} \right|}{\left|\mathcal{D}_{Y_{b,c}}\right| +
        \left|\mathcal{D}_{\hat{Y}_{b,c}}\right|}
        :label: nsd

    with :math:`\mathcal{D}_{Y_{b,c}}` and :math:`\mathcal{D}_{\hat{Y}_{b,c}}` being two sets of nearest-neighbor
    distances. :math:`\mathcal{D}_{Y_{b,c}}` is computed from the predicted segmentation boundary towards the reference
    segmentation boundary and vice-versa for :math:`\mathcal{D}_{\hat{Y}_{b,c}}`. :math:`\mathcal{D}_{Y_{b,c}}^{'}` and
    :math:`\mathcal{D}_{\hat{Y}_{b,c}}^{'}` refer to the subsets of distances that are smaller or equal to the
    acceptable distance :math:`\tau_c`:

    .. math::
        \mathcal{D}_{Y_{b,c}}^{'} = \{ d \in \mathcal{D}_{Y_{b,c}} \, | \, d \leq \tau_c \}.


    In the case of a class neither being present in the predicted segmentation, nor in the reference segmentation,
    a nan value will be returned for this class. In the case of a class being present in only one of predicted
    segmentation or reference segmentation, the class NSD will be 0.

    This implementation is based on https://arxiv.org/abs/2111.05408 and supports 2D and 3D images.
    The computation of boundaries follows DeepMind's implementation
    https://github.com/deepmind/surface-distance when `use_subvoxels=True`; Otherwise the length of a segmentation
    boundary is interpreted as the number of its edge pixels.

    Args:
        y_pred: Predicted segmentation, typically segmentation model output.
            It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
        y: Reference segmentation.
            It must be a one-hot encoded, batch-first tensor [B,C,H,W] or [B,C,H,W,D].
        class_thresholds: List of class-specific thresholds.
            The thresholds relate to the acceptable amount of deviation in the segmentation boundary in pixels.
            Each threshold needs to be a finite, non-negative number.
        include_background: Whether to include the surface dice computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: The metric used to compute surface distances.
            One of [``"euclidean"``, ``"chessboard"``, ``"taxicab"``].
            Defaults to ``"euclidean"``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
            the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
            If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
            If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
            else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
            for all images in batch. Defaults to ``None``.
        use_subvoxels: Whether to use subvoxel distances. Defaults to ``False``.

    Raises:
        ValueError: If `y_pred` and/or `y` are not PyTorch tensors.
        ValueError: If `y_pred` and/or `y` do not have four dimensions.
        ValueError: If `y_pred` and/or `y` have different shapes.
        ValueError: If `y_pred` and/or `y` are not one-hot encoded
        ValueError: If the number of channels of `y_pred` and/or `y` is different from the number of class thresholds.
        ValueError: If any class threshold is not finite.
        ValueError: If any class threshold is negative.

    Returns:
        Pytorch Tensor of shape [B,C], containing the NSD values :math:`\operatorname {NSD}_{b,c}` for each batch index
        :math:`b` and class :math:`c`.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y_pred.ndimension() not in (4, 5) or y.ndimension() not in (4, 5):
        raise ValueError("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].")

    if y_pred.shape != y.shape:
        raise ValueError(
            f"y_pred and y should have same shape, but instead, shapes are {y_pred.shape} (y_pred) and {y.shape} (y)."
        )

    batch_size, n_class = y_pred.shape[:2]

    if n_class != len(class_thresholds):
        raise ValueError(
            f"number of classes ({n_class}) does not match number of class thresholds ({len(class_thresholds)})."
        )

    if any(~np.isfinite(class_thresholds)):
        raise ValueError("All class thresholds need to be finite.")

    if any(np.array(class_thresholds) < 0):
        raise ValueError("All class thresholds need to be >= 0.")

    nsd = torch.empty((batch_size, n_class), device=y_pred.device, dtype=torch.float)

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas = get_edge_surface_distance(  # type: ignore
            y_pred[b, c],
            y[b, c],
            distance_metric=distance_metric,
            spacing=spacing_list[b],
            use_subvoxels=use_subvoxels,
            symmetric=True,
            class_index=c,
        )
        boundary_correct: int | torch.Tensor | float
        boundary_complete: int | torch.Tensor | float
        if not use_subvoxels:
            boundary_complete = len(distances_pred_gt) + len(distances_gt_pred)
            boundary_correct = torch.sum(distances_pred_gt <= class_thresholds[c]) + torch.sum(
                distances_gt_pred <= class_thresholds[c]
            )
        else:
            areas_pred, areas_gt = areas  # type: ignore
            areas_gt, areas_pred = areas_gt[edges_gt], areas_pred[edges_pred]
            boundary_complete = areas_gt.sum() + areas_pred.sum()
            gt_true = areas_gt[distances_gt_pred <= class_thresholds[c]].sum() if len(areas_gt) > 0 else 0.0
            pred_true = areas_pred[distances_pred_gt <= class_thresholds[c]].sum() if len(areas_pred) > 0 else 0.0
            boundary_correct = gt_true + pred_true
        if boundary_complete == 0:
            # the class is neither present in the prediction, nor in the reference segmentation
            nsd[b, c] = torch.tensor(np.nan)
        else:
            nsd[b, c] = boundary_correct / boundary_complete

    return nsd
