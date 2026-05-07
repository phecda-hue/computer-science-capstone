"""
utils/metrics.py
YOLO 검출 / DAv2 깊이 추정 평가 지표
"""

import numpy as np
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# YOLO 평가 지표
# ══════════════════════════════════════════════════════════════════════════════

def iou(box_a: list, box_b: list) -> float:
    """[x1,y1,x2,y2] 형식의 두 박스 IoU"""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / (union + 1e-6)


def compute_detection_metrics(
    preds: List[Dict],
    gts:   List[Dict],
    iou_threshold: float = 0.5,
) -> Dict:
    """
    Parameters
    ----------
    preds : [{"box":[x1,y1,x2,y2], "score":float, "class":int}, ...]
    gts   : [{"box":[x1,y1,x2,y2], "class":int}, ...]

    Returns
    -------
    {"precision": float, "recall": float, "f1": float, "num_pred": int, "num_gt": int}
    """
    if not gts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "num_pred": len(preds), "num_gt": 0}

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    matched_gt   = [False] * len(gts)
    tp = 0

    for pred in preds_sorted:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            if pred["class"] != gt["class"]:
                continue
            v = iou(pred["box"], gt["box"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt[best_j] = True

    precision = tp / (len(preds) + 1e-6)
    recall    = tp / (len(gts)  + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "num_pred":  len(preds),
        "num_gt":    len(gts),
    }


def aggregate_detection_metrics(per_image_metrics: List[Dict]) -> Dict:
    keys = ["precision", "recall", "f1"]
    agg = {k: np.mean([m[k] for m in per_image_metrics]) for k in keys}
    agg["total_pred"] = sum(m["num_pred"] for m in per_image_metrics)
    agg["total_gt"]   = sum(m["num_gt"]   for m in per_image_metrics)
    return {k: round(float(v), 4) for k, v in agg.items()}


# ══════════════════════════════════════════════════════════════════════════════
# DAv2 깊이 추정 평가 지표  (GT depth 있을 때 사용)
# ══════════════════════════════════════════════════════════════════════════════

def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth:   np.ndarray,
    min_depth:  float = 0.1,
    max_depth:  float = 80.0,
) -> Dict:
    """
    표준 depth estimation 지표:
      AbsRel, SqRel, RMSE, RMSElog, δ<1.25, δ<1.25², δ<1.25³
    GT depth가 없으면 consistency 지표(Spearman 상관계수)만 사용
    """
    mask = (gt_depth > min_depth) & (gt_depth < max_depth)
    if mask.sum() == 0:
        return {}

    pred = pred_depth[mask]
    gt   = gt_depth[mask]

    thresh = np.maximum(gt / pred, pred / gt)
    d1 = (thresh < 1.25  ).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    abs_rel  = (np.abs(gt - pred) / gt).mean()
    sq_rel   = (((gt - pred) ** 2) / gt).mean()
    rmse     = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())

    return {
        "AbsRel":  round(float(abs_rel),  4),
        "SqRel":   round(float(sq_rel),   4),
        "RMSE":    round(float(rmse),      4),
        "RMSElog": round(float(rmse_log),  4),
        "d1":      round(float(d1),        4),
        "d2":      round(float(d2),        4),
        "d3":      round(float(d3),        4),
    }


def compute_depth_consistency(pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    """
    GT가 없을 때 두 모델 출력 간 일관성 측정
    - Spearman 순위 상관계수
    - 평균 절대 차이 (정규화)
    """
    from scipy.stats import spearmanr

    a_flat = pred_a.flatten().astype(np.float64)
    b_flat = pred_b.flatten().astype(np.float64)

    corr, _ = spearmanr(a_flat, b_flat)

    norm_a = (a_flat - a_flat.min()) / (a_flat.max() - a_flat.min() + 1e-6)
    norm_b = (b_flat - b_flat.min()) / (b_flat.max() - b_flat.min() + 1e-6)
    mad = np.abs(norm_a - norm_b).mean()

    return {
        "spearman_corr":  round(float(corr), 4),
        "norm_mean_diff": round(float(mad),  4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 레이턴시 통계
# ══════════════════════════════════════════════════════════════════════════════

def compute_latency_stats(latencies_ms: List[float]) -> Dict:
    arr = np.array(latencies_ms)
    return {
        "mean_ms":   round(float(arr.mean()), 2),
        "std_ms":    round(float(arr.std()),  2),
        "min_ms":    round(float(arr.min()),  2),
        "max_ms":    round(float(arr.max()),  2),
        "p50_ms":    round(float(np.percentile(arr, 50)), 2),
        "p95_ms":    round(float(np.percentile(arr, 95)), 2),
        "p99_ms":    round(float(np.percentile(arr, 99)), 2),
        "fps":       round(1000 / float(arr.mean()), 1),
    }
