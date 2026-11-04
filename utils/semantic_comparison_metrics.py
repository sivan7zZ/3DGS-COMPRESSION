"""Dataset-aggregated City semantic, boundary and connected-region metrics.

All metric values are fractions, not percentages. This module compares semantic
label maps; connected regions are not annotated physical object instances.
Only NumPy and OpenCV are required, and no model or GPU is used here.
"""
from collections import deque
import math

import cv2
import numpy as np


def _ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def _mean_present(values):
    present = [float(v) for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _validate_map(value, name, num_classes):
    value = np.asarray(value)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError('{} must be a 2D integer label map'.format(name))
    if value.size == 0:
        raise ValueError('{} is empty'.format(name))
    if np.any(value < 0) or np.any(value >= num_classes):
        raise ValueError('{} contains labels outside [0, {})'.format(name, num_classes))
    return value


def _erode(mask, distance, outside_value):
    return cv2.erode(np.ascontiguousarray(mask, dtype=np.uint8),
                     np.ones((3, 3), dtype=np.uint8), iterations=int(distance),
                     borderType=cv2.BORDER_CONSTANT, borderValue=int(outside_value)).astype(bool)


def inner_boundary(mask, distance):
    """Official Boundary IoU inner band: foreground minus zero-padded erosion."""
    mask = np.asarray(mask, dtype=bool)
    if int(distance) < 1:
        raise ValueError('boundary distance must be positive')
    return mask & ~_erode(mask, int(distance), outside_value=0)


def _maximum_cardinality_matching(adjacency):
    """One-to-one sparse bipartite matching, without a dense IoU matrix.

    An iterative augmenting-path search avoids recursion depth dependence.
    Adjacency order may favor high IoU, but cardinality, not greedy IoU, decides
    the number of recalled regions.
    """
    matched_gt, matched_pred = {}, {}
    for start in sorted(adjacency):
        if start in matched_gt:
            continue
        queue = deque([start])
        seen_gt, seen_pred, parent_pred = {start}, set(), {}
        free_pred = None
        while queue and free_pred is None:
            gt_id = queue.popleft()
            for pred_id in adjacency[gt_id]:
                if pred_id in seen_pred:
                    continue
                seen_pred.add(pred_id)
                parent_pred[pred_id] = gt_id
                if pred_id not in matched_pred:
                    free_pred = pred_id
                    break
                next_gt = matched_pred[pred_id]
                if next_gt not in seen_gt:
                    seen_gt.add(next_gt)
                    queue.append(next_gt)
        if free_pred is not None:
            pred_id = free_pred
            while True:
                gt_id = parent_pred[pred_id]
                old_pred = matched_gt.get(gt_id)
                matched_gt[gt_id] = pred_id
                matched_pred[pred_id] = gt_id
                if old_pred is None:
                    break
                pred_id = old_pred
    return matched_gt


def _connected_regions(pred_mask, gt_mask, min_gt_area, small_max_area, iou_threshold):
    n_gt, labels_gt, stats_gt, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(gt_mask, dtype=np.uint8), connectivity=8)
    n_pred, labels_pred, stats_pred, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(pred_mask, dtype=np.uint8), connectivity=8)
    areas_gt = stats_gt[:, cv2.CC_STAT_AREA].astype(np.int64)
    areas_pred = stats_pred[:, cv2.CC_STAT_AREA].astype(np.int64)
    eligible = np.zeros(n_gt, dtype=bool)
    eligible[1:] = areas_gt[1:] >= min_gt_area
    small = eligible & (areas_gt <= small_max_area)
    small[0] = False
    adjacency_scores = {}
    overlap = (labels_gt > 0) & (labels_pred > 0)
    if np.any(overlap):
        # A full G x P matrix is unnecessary: disconnected maps have sparse overlap.
        pair_codes = labels_gt[overlap].astype(np.int64) * n_pred + labels_pred[overlap]
        codes, intersections = np.unique(pair_codes, return_counts=True)
        for code, intersection in zip(codes, intersections):
            gt_id, pred_id = int(code // n_pred), int(code % n_pred)
            if not eligible[gt_id]:
                continue
            union = int(areas_gt[gt_id] + areas_pred[pred_id] - intersection)
            iou = float(intersection) / union
            if iou + 1e-12 >= iou_threshold:
                adjacency_scores.setdefault(gt_id, []).append((iou, pred_id))
    adjacency = {g: [p for _, p in sorted(edges, key=lambda x: (-x[0], x[1]))]
                 for g, edges in adjacency_scores.items()}
    matches = _maximum_cardinality_matching(adjacency)
    matched_ids = list(matches)
    return {'gt': int(eligible.sum()), 'tp': len(matches),
            'small_gt': int(small.sum()),
            'small_tp': int(small[matched_ids].sum()) if matched_ids else 0,
            'pred_components': int(n_pred - 1)}


def evaluate_semantics(predictions, ground_truths, num_classes=7, ignore_index=0,
                       boundary_dilation_ratio=0.02, region_min_area=16,
                       region_small_fraction=0.01, region_iou_threshold=0.5):
    """Evaluate aligned prediction/GT maps with a frozen, common protocol.

    Confusion rows are GT and columns prediction. GT=ignore_index is excluded;
    predicting ignore on a valid GT pixel still creates a false negative.
    mIoU averages non-void classes with positive dataset union, including false
    positives for absent GT classes. mAcc averages classes with GT support.

    Boundary IoU uses d=max(1, round(ratio*hypot(H,W))) and 3x3 inner erosion.
    Bands are computed on full class masks then restricted to valid GT pixels
    eroded by d; image exterior is valid for that restriction, so void edges are
    excluded but ordinary image-border bands remain. Intersection/union are
    aggregated across the dataset before macro averaging positive-union classes.

    RegionRecall@IoU>=0.5 uses same-class 8-connected semantic components,
    minimum GT area 16 pixels, and maximum-cardinality one-to-one matching.
    Small means GT region area <=1% of its full image. Predictions are evaluated
    only within valid GT; no minimum prediction area removes missed targets.
    """
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
        predictions = [predictions]
    if isinstance(ground_truths, np.ndarray) and ground_truths.ndim == 2:
        ground_truths = [ground_truths]
    predictions, ground_truths = list(predictions), list(ground_truths)
    if len(predictions) != len(ground_truths) or not predictions:
        raise ValueError('predictions and ground_truths must have equal nonzero length')
    if num_classes < 2 or not 0 <= ignore_index < num_classes:
        raise ValueError('invalid class count or ignore index')
    if not math.isfinite(boundary_dilation_ratio) or boundary_dilation_ratio < 0:
        raise ValueError('boundary_dilation_ratio must be finite and nonnegative')
    if region_min_area < 1 or not 0 < region_small_fraction <= 1:
        raise ValueError('invalid region area thresholds')
    if not 0 < region_iou_threshold <= 1:
        raise ValueError('region_iou_threshold must be in (0, 1]')
    classes = [c for c in range(num_classes) if c != ignore_index]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    boundary_intersection = np.zeros(num_classes, dtype=np.int64)
    boundary_union = np.zeros(num_classes, dtype=np.int64)
    region_counts = {c: {'gt': 0, 'tp': 0, 'small_gt': 0, 'small_tp': 0,
                         'pred_components': 0} for c in classes}
    images = []
    for index, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred = _validate_map(pred, 'prediction[{}]'.format(index), num_classes)
        gt = _validate_map(gt, 'ground_truth[{}]'.format(index), num_classes)
        if pred.shape != gt.shape:
            raise ValueError('prediction and GT shapes differ at image {}'.format(index))
        valid = gt != ignore_index
        pair_codes = gt[valid].astype(np.int64) * num_classes + pred[valid]
        confusion += np.bincount(pair_codes, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        height, width = gt.shape
        distance = max(1, int(round(boundary_dilation_ratio * math.hypot(height, width))))
        safe_valid = _erode(valid, distance, outside_value=1)
        image_regions = {}
        for cls in classes:
            gt_mask, pred_mask = gt == cls, pred == cls
            gt_band = inner_boundary(gt_mask, distance) & safe_valid
            pred_band = inner_boundary(pred_mask, distance) & safe_valid
            boundary_intersection[cls] += np.count_nonzero(gt_band & pred_band)
            boundary_union[cls] += np.count_nonzero(gt_band | pred_band)
            counts = _connected_regions(pred_mask & valid, gt_mask & valid,
                                        region_min_area, region_small_fraction * height * width,
                                        region_iou_threshold)
            image_regions[str(cls)] = counts
            for key in counts:
                region_counts[cls][key] += counts[key]
        images.append({'index': index, 'height': height, 'width': width,
                       'valid_pixels': int(valid.sum()), 'boundary_dilation_px': distance,
                       'boundary_valid_pixels': int(safe_valid.sum()),
                       'region_counts': image_regions})
    gt_support = confusion.sum(axis=1)
    pred_support = confusion.sum(axis=0)
    intersection = np.diag(confusion)
    union = gt_support + pred_support - intersection
    per_class = {}
    for cls in classes:
        counts = region_counts[cls]
        per_class[str(cls)] = {'gt_pixels': int(gt_support[cls]),
                              'pred_pixels_on_valid_gt': int(pred_support[cls]),
                              'intersection': int(intersection[cls]), 'union': int(union[cls]),
                              'iou': _ratio(intersection[cls], union[cls]),
                              'pixel_recall': _ratio(intersection[cls], gt_support[cls]),
                              'boundary_intersection': int(boundary_intersection[cls]),
                              'boundary_union': int(boundary_union[cls]),
                              'boundary_iou': _ratio(boundary_intersection[cls], boundary_union[cls]),
                              'regions': dict(counts, recall=_ratio(counts['tp'], counts['gt']),
                                              small_recall=_ratio(counts['small_tp'], counts['small_gt']))}
    gt_regions = sum(x['gt'] for x in region_counts.values())
    true_positive_regions = sum(x['tp'] for x in region_counts.values())
    small_gt = sum(x['small_gt'] for x in region_counts.values())
    small_tp = sum(x['small_tp'] for x in region_counts.values())
    vehicle = region_counts.get(4)
    return {'miou': _mean_present(x['iou'] for x in per_class.values()),
            'macc': _mean_present(x['pixel_recall'] for x in per_class.values()),
            'b_miou': _mean_present(x['boundary_iou'] for x in per_class.values()),
            'region_recall': {'name': 'Semantic RegionRecall@IoU>={:g}'.format(region_iou_threshold),
                              'tp': true_positive_regions, 'gt': gt_regions,
                              'recall': _ratio(true_positive_regions, gt_regions),
                              'macro_class_recall': _mean_present(x['regions']['recall'] for x in per_class.values()),
                              'small_tp': small_tp, 'small_gt': small_gt,
                              'small_recall': _ratio(small_tp, small_gt),
                              'vehicle_tp': vehicle['tp'] if vehicle else None,
                              'vehicle_gt': vehicle['gt'] if vehicle else None,
                              'vehicle_recall': _ratio(vehicle['tp'], vehicle['gt']) if vehicle else None},
            'confusion': confusion.tolist(), 'per_class': per_class, 'per_image': images,
            'protocol': {'num_classes': num_classes, 'ignore_gt': ignore_index,
                         'value_scale': 'fraction_0_to_1', 'aggregation': 'dataset_counts_before_class_macro',
                         'miou_classes': 'nonvoid_classes_with_union>0',
                         'macc_classes': 'nonvoid_classes_with_gt_support>0',
                         'boundary_classes': 'nonvoid_classes_with_boundary_union>0',
                         'boundary_dilation_ratio': float(boundary_dilation_ratio),
                         'boundary_dilation_formula': 'max(1, round(ratio * hypot(height, width)))',
                         'boundary_band': 'mask minus zero-padded repeated 3x3 erosion',
                         'boundary_valid_domain': 'GT nonvoid eroded by d; image exterior valid for this restriction',
                         'region_definition': 'same-class 8-connected semantic regions, not physical object instances',
                         'region_matching': 'maximum-cardinality one-to-one sparse IoU matching',
                         'region_iou_threshold': float(region_iou_threshold),
                         'region_min_gt_area_pixels': int(region_min_area),
                         'region_small_max_image_area_fraction': float(region_small_fraction)}}


def semantic_metrics(predictions, ground_truths, **kwargs):
    """Alias retained for callers requesting semantic_metrics by name."""
    return evaluate_semantics(predictions, ground_truths, **kwargs)
