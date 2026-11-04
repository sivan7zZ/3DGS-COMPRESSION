"""Fixed City preservation metrics on registered semantic masks and native RGB.

Semantic connected components are evaluation proxies, not physical instances.
Skeletonization is deliberately lazy so the image-quality GPU environment does
not require scikit-image. CPU mask evaluation uses its hard Zhang skeleton.
"""
import math

import cv2
import numpy as np

from utils.semantic_comparison_metrics import _connected_regions, _validate_map


def mean_present(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def erode(mask, iterations, outside=1):
    return cv2.erode(np.asarray(mask, np.uint8), np.ones((3, 3), np.uint8),
                     iterations=iterations, borderType=cv2.BORDER_CONSTANT,
                     borderValue=outside).astype(bool)


def boundary(mask):
    """One-pixel inner contour, excluding artificial image-exterior contours."""
    mask = np.asarray(mask, bool)
    return mask & ~erode(mask, 1, outside=1)


def boundary_counts(pred, gt, valid, tolerance=2):
    if int(tolerance) != tolerance or tolerance < 0:
        raise ValueError('Boundary tolerance must be a nonnegative pixel integer')
    tolerance = int(tolerance)
    # Keep a complete tolerance neighborhood away from unannotated pixels.
    safe = erode(valid, tolerance + 1, outside=1)
    pb, gb = boundary(pred) & safe, boundary(gt) & safe
    y, x = np.ogrid[-tolerance:tolerance+1, -tolerance:tolerance+1]
    disk = (x*x + y*y <= tolerance*tolerance).astype(np.uint8)
    pd = cv2.dilate(pb.astype(np.uint8), disk).astype(bool)
    gd = cv2.dilate(gb.astype(np.uint8), disk).astype(bool)
    return dict(predicted=int(pb.sum()), target=int(gb.sum()),
                matched_predicted=int((pb & gd).sum()), matched_target=int((gb & pd).sum()))


def boundary_score(counts):
    p, g = counts['predicted'], counts['target']
    if not p and not g:
        return dict(precision=None, recall=None, f1=None)
    precision = counts['matched_predicted']/p if p else 1.
    recall = counts['matched_target']/g if g else 1.
    score = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(precision=float(precision), recall=float(recall), f1=float(score))


def hard_skeleton(mask):
    from skimage.morphology import skeletonize
    # Padding makes the treatment of objects touching the image border explicit.
    value = np.pad(np.asarray(mask, bool), 1)
    return skeletonize(value, method='zhang')[1:-1, 1:-1]


def cldice_counts(pred, gt, valid, gt_skeleton=None, pred_skeleton=None):
    pm, gm = np.asarray(pred, bool) & valid, np.asarray(gt, bool) & valid
    ps = hard_skeleton(pm) if pred_skeleton is None else np.asarray(pred_skeleton, bool)
    gs = hard_skeleton(gm) if gt_skeleton is None else np.asarray(gt_skeleton, bool)
    safe = erode(valid, 1, outside=1)
    ps, gs = ps & safe, gs & safe
    return dict(predicted_skeleton=int(ps.sum()), target_skeleton=int(gs.sum()),
                predicted_in_target=int((ps & gm).sum()), target_in_predicted=int((gs & pm).sum()))


def cldice_score(counts):
    p, g = counts['predicted_skeleton'], counts['target_skeleton']
    if not p and not g:
        return dict(topology_precision=None, topology_recall=None, cldice=None)
    precision = counts['predicted_in_target']/p if p else 0.
    recall = counts['target_in_predicted']/g if g else 0.
    value = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(topology_precision=float(precision), topology_recall=float(recall), cldice=float(value))


def evaluate_masks(predictions, ground_truths, tolerance=2, gt_skeletons=None,
                   min_area=16, max_fraction=.01, iou_threshold=.5):
    predictions, ground_truths = list(predictions), list(ground_truths)
    if not predictions or len(predictions) != len(ground_truths):
        raise ValueError('Expected aligned nonempty mask lists')
    totals = {str(c): dict(boundary=dict(predicted=0,target=0,matched_predicted=0,matched_target=0),
                           skeleton=dict(predicted_skeleton=0,target_skeleton=0,predicted_in_target=0,target_in_predicted=0),
                           regions=dict(gt=0,tp=0,small_gt=0,small_tp=0,pred_components=0)) for c in range(1,7)}
    per_view = []
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred, gt = _validate_map(pred,'prediction',7), _validate_map(gt,'GT',7)
        if pred.shape != gt.shape:
            raise ValueError('Prediction and GT geometry differ')
        valid = gt != 0
        rows = {}
        for c in range(1,7):
            pm, gm = pred == c, gt == c
            b = boundary_counts(pm,gm,valid,tolerance)
            sk = cldice_counts(pm,gm,valid,None if gt_skeletons is None else gt_skeletons[i][c])
            regions = _connected_regions(pm & valid,gm,min_area,max_fraction*gt.size,iou_threshold)
            rows[str(c)] = dict(boundary=b,skeleton=sk,regions=regions)
            for group in ('boundary','skeleton','regions'):
                for key, value in rows[str(c)][group].items():
                    totals[str(c)][group][key] += value
        per_view.append(rows)
    for value in totals.values():
        value['boundary'].update(boundary_score(value['boundary']))
        value['skeleton'].update(cldice_score(value['skeleton']))
    small_tp = sum(x['regions']['small_tp'] for x in totals.values())
    small_gt = sum(x['regions']['small_gt'] for x in totals.values())
    vehicle = totals['4']['regions']
    return dict(small_object_recall=float(small_tp/small_gt) if small_gt else None,
                small_tp=small_tp,small_gt=small_gt,
                vehicle_small_recall=float(vehicle['small_tp']/vehicle['small_gt']) if vehicle['small_gt'] else None,
                vehicle_small_tp=vehicle['small_tp'],vehicle_small_gt=vehicle['small_gt'],
                boundary_f1=mean_present(x['boundary']['f1'] for x in totals.values()),
                cldice=mean_present(x['skeleton']['cldice'] for x in totals.values()),
                per_class=totals,per_view=per_view)


def roi_box(box, width, height, padding=8, minimum=32):
    """Same native context crop for every method; never resize tiny targets."""
    x0,y0,x1,y1 = map(int,box)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Invalid target bounding box')
    if padding < 0 or minimum < 1:
        raise ValueError('Invalid crop parameters')
    def interval(lo,hi,limit):
        desired = min(limit,max(minimum,hi-lo+2*padding))
        left = max(0,min((lo+hi-desired)//2,limit-desired))
        return left,left+desired
    left,right = interval(x0,x1,width)
    top,bottom = interval(y0,y1,height)
    return [left,top,right,bottom]


def small_regions(labels,min_area=16,max_fraction=.01):
    labels = _validate_map(labels,'GT',7)
    mapping = np.zeros(labels.shape,np.int32)
    rows = []
    h,w = labels.shape
    for c in range(1,7):
        n,cc,stats,_ = cv2.connectedComponentsWithStats((labels==c).astype(np.uint8),connectivity=8)
        for k in range(1,n):
            x,y,ww,hh,area = map(int,stats[k])
            if not min_area <= area <= max_fraction*labels.size:
                continue
            ident = len(rows)+1
            mapping[cc==k] = ident
            rows.append(dict(region_id=ident,class_id=c,component_id=k,area=area,
                             bbox_xyxy=[x,y,x+ww,y+hh],crop_xyxy=roi_box([x,y,x+ww,y+hh],w,h)))
    return rows,mapping


def roi_psnr(reference,predicted,box,mask=None):
    if reference.shape != predicted.shape or reference.dtype != np.uint8 or predicted.dtype != np.uint8:
        raise ValueError('Expected matching native uint8 RGB')
    x0,y0,x1,y1 = box
    difference=(reference[y0:y1,x0:x1].astype(np.float64)-predicted[y0:y1,x0:x1].astype(np.float64))/255.
    if mask is not None:
        selected=np.asarray(mask,bool)[y0:y1,x0:x1]
        if not selected.any():raise ValueError('Empty ROI mask')
        difference=difference[selected]
    mse=float(np.mean(difference*difference))
    return dict(mse=mse,psnr=float(-10*math.log10(max(mse,1e-12))))
