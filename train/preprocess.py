"""
utils/preprocess.py
공통 전처리 함수 - YOLO / DAv2 입력 정규화
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


# ── 공통 상수 ──────────────────────────────────────────────────────────────────
YOLO_SIZE   = 640          # YOLO 입력 해상도 (정방형)
DAV2_SIZE   = 518          # DAv2 Small 입력 해상도 (정방형)
YOLO_MEAN   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
YOLO_STD    = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # ultralytics는 /255 only
DAV2_MEAN   = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DAV2_STD    = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── YOLO 전처리 ────────────────────────────────────────────────────────────────
def preprocess_yolo(
    image_path: str | Path,
    size: int = YOLO_SIZE,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
    """
    Returns
    -------
    blob        : (1, 3, H, W) float32  – 모델 입력
    orig_shape  : (orig_h, orig_w)
    scale       : (scale_w, scale_h)   – 후처리 박스 복원에 사용
    """
    img = cv2.imread(str(image_path))
    orig_h, orig_w = img.shape[:2]

    # letterbox resize
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # padding
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # BGR → RGB, HWC → CHW, /255
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = blob[np.newaxis, ...]  # (1,3,H,W)

    return blob, (orig_h, orig_w), (scale, scale), (pad_x, pad_y)


# ── DAv2 전처리 ────────────────────────────────────────────────────────────────
def preprocess_dav2(
    image_path: str | Path,
    size: int = DAV2_SIZE,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Returns
    -------
    blob       : (1, 3, H, W) float32  – ImageNet 정규화 적용
    orig_shape : (orig_h, orig_w)
    """
    img = cv2.imread(str(image_path))
    orig_h, orig_w = img.shape[:2]

    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0

    normalized = (rgb - DAV2_MEAN) / DAV2_STD               # (H,W,3)
    blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]    # (1,3,H,W)

    return blob.astype(np.float32), (orig_h, orig_w)


# ── 테스트 이미지 수집 ─────────────────────────────────────────────────────────
def collect_test_images(test_dir: str | Path, exts=(".jpg", ".jpeg", ".png")) -> list:
    test_dir = Path(test_dir)
    images = []
    for ext in exts:
        images.extend(sorted(test_dir.rglob(f"*{ext}")))
    if not images:
        raise FileNotFoundError(f"테스트 이미지를 찾을 수 없습니다: {test_dir}")
    return images

# ── YOLO 출력 후처리 (NMS 포함) ───────────────────────────────────────────────
def postprocess_yolo(
    output: np.ndarray,
    orig_shape,
    scale,
    pad,
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
) -> list:
    """
    두 가지 YOLO 출력 형식 자동 감지:
      A) (1, 300, 6)          → end-to-end NMS 내장 [x1,y1,x2,y2,score,cls]
      B) (1, 4+classes, anch) → 표준 앵커 출력, NMS 수동 적용
    """

    # ── 형식 A: (1, 300, 6) end-to-end ───────────────────────────────────────
    if output.ndim == 3 and output.shape[2] == 6:
        pred = output[0]   # (300, 6)
        results = []
        for det in pred:
            x1, y1, x2, y2, score, cls_id = det
            if score < conf_thres:
                continue
            # letterbox 역변환
            pad_x, pad_y   = pad
            scale_w, scale_h = scale
            oh, ow = orig_shape
            x1 = float(np.clip((x1 - pad_x) / scale_w, 0, ow))
            y1 = float(np.clip((y1 - pad_y) / scale_h, 0, oh))
            x2 = float(np.clip((x2 - pad_x) / scale_w, 0, ow))
            y2 = float(np.clip((y2 - pad_y) / scale_h, 0, oh))
            results.append({
                "box":   [x1, y1, x2, y2],
                "score": float(score),
                "class": int(cls_id),
            })
        return results

    # ── 형식 B: (1, 4+classes, anchors) 표준 ─────────────────────────────────
    pred = output[0].T                           # (num_anchors, 4+num_classes)
    boxes  = pred[:, :4]
    scores = pred[:, 4:]

    class_ids   = scores.argmax(axis=1)
    confidences = scores.max(axis=1)

    mask = confidences > conf_thres
    boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    pad_x, pad_y   = pad
    scale_w, scale_h = scale
    oh, ow = orig_shape
    x1 = np.clip((x1 - pad_x) / scale_w, 0, ow)
    y1 = np.clip((y1 - pad_y) / scale_h, 0, oh)
    x2 = np.clip((x2 - pad_x) / scale_w, 0, ow)
    y2 = np.clip((y2 - pad_y) / scale_h, 0, oh)

    rects       = np.stack([x1, y1, x2-x1, y2-y1], axis=1).tolist()
    scores_list = confidences.tolist()

    indices = cv2.dnn.NMSBoxes(rects, scores_list, conf_thres, iou_thres)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append({
                "box":   [x1[i], y1[i], x2[i], y2[i]],
                "score": float(confidences[i]),
                "class": int(class_ids[i]),
            })
    return results
