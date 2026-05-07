import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


def load_kitti_depth(depth_path: Path) -> np.ndarray:
    """
    KITTI depth png는 보통 uint16이고, 실제 meter 단위 depth는 raw / 256.0 입니다.
    depth가 없는 픽셀은 0입니다.
    """
    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if raw is None:
        raise FileNotFoundError(f"GT depth를 읽을 수 없습니다: {depth_path}")

    depth = raw.astype(np.float32) / 256.0
    return depth


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    pred, gt: meter 단위 depth map
    """
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)

    if valid.sum() == 0:
        return {
            "valid_pixels": 0,
            "abs_rel": np.nan,
            "sq_rel": np.nan,
            "rmse": np.nan,
            "rmse_log": np.nan,
            "mae": np.nan,
            "delta1": np.nan,
            "delta2": np.nan,
            "delta3": np.nan,
            "mae_0_5m": np.nan,
            "mae_5_15m": np.nan,
        }

    p = pred[valid]
    g = gt[valid]

    eps = 1e-6

    abs_rel = np.mean(np.abs(g - p) / (g + eps))
    sq_rel = np.mean(((g - p) ** 2) / (g + eps))
    rmse = np.sqrt(np.mean((g - p) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(g + eps) - np.log(p + eps)) ** 2))
    mae = np.mean(np.abs(g - p))

    ratio = np.maximum(g / (p + eps), p / (g + eps))
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25 ** 2)
    delta3 = np.mean(ratio < 1.25 ** 3)

    near_0_5 = valid & (gt <= 5.0)
    near_5_15 = valid & (gt > 5.0) & (gt <= 15.0)

    mae_0_5m = np.mean(np.abs(pred[near_0_5] - gt[near_0_5])) if near_0_5.sum() > 0 else np.nan
    mae_5_15m = np.mean(np.abs(pred[near_5_15] - gt[near_5_15])) if near_5_15.sum() > 0 else np.nan

    return {
        "valid_pixels": int(valid.sum()),
        "abs_rel": float(abs_rel),
        "sq_rel": float(sq_rel),
        "rmse": float(rmse),
        "rmse_log": float(rmse_log),
        "mae": float(mae),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3),
        "mae_0_5m": float(mae_0_5m),
        "mae_5_15m": float(mae_5_15m),
    }


def save_depth_visualization(depth: np.ndarray, save_path: Path):
    d = depth.copy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    d_min, d_max = np.percentile(d[d > 0], [1, 99]) if np.any(d > 0) else (0, 1)
    vis = np.clip((d - d_min) / (d_max - d_min + 1e-6), 0, 1)
    vis = (vis * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

    cv2.imwrite(str(save_path), vis)


@torch.no_grad()
def predict_depth(image_path: Path, processor, model, device) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    inputs = processor(images=image, return_tensors="pt").to(device)

    outputs = model(**inputs)
    pred = outputs.predicted_depth

    pred = F.interpolate(
        pred.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    return pred.cpu().numpy().astype(np.float32)


def find_gt_path(image_path: Path, gt_dir: Path) -> Path | None:
    candidates = [
        gt_dir / image_path.name,
        gt_dir / image_path.with_suffix(".png").name,
        gt_dir / image_path.with_suffix(".jpg").name,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="dav2_kitti_results")
    parser.add_argument("--model_size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    out_dir = Path(args.out_dir)

    depth_out = out_dir / "pred_depth_npy"
    vis_out = out_dir / "pred_depth_vis"

    depth_out.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = MODEL_IDS[args.model_size]
    print(f"사용 모델: {model_id}")
    print(f"사용 장치: {device}")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    model.eval()

    image_paths = sorted(
        list(image_dir.glob("*.png")) +
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg"))
    )

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    results = []

    for image_path in tqdm(image_paths):
        pred_depth = predict_depth(image_path, processor, model, device)

        npy_path = depth_out / f"{image_path.stem}_pred_depth.npy"
        vis_path = vis_out / f"{image_path.stem}_pred_depth.png"

        np.save(npy_path, pred_depth)
        save_depth_visualization(pred_depth, vis_path)

        row = {
            "image": image_path.name,
            "pred_min_m": float(np.nanmin(pred_depth)),
            "pred_max_m": float(np.nanmax(pred_depth)),
            "pred_mean_m": float(np.nanmean(pred_depth)),
        }

        if gt_dir is not None:
            gt_path = find_gt_path(image_path, gt_dir)

            if gt_path is not None:
                gt_depth = load_kitti_depth(gt_path)
                metrics = compute_metrics(pred_depth, gt_depth)
                row.update(metrics)
                row["gt_found"] = True
            else:
                row["gt_found"] = False

        results.append(row)

    df = pd.DataFrame(results)
    csv_path = out_dir / "metrics_per_image.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n저장 완료:")
    print(f"- 예측 depth npy: {depth_out}")
    print(f"- depth 시각화: {vis_out}")
    print(f"- 이미지별 결과 CSV: {csv_path}")

    if gt_dir is not None and "gt_found" in df.columns:
        eval_df = df[df["gt_found"] == True]

        if len(eval_df) > 0:
            metric_cols = [
                "abs_rel", "sq_rel", "rmse", "rmse_log",
                "mae", "delta1", "delta2", "delta3",
                "mae_0_5m", "mae_5_15m"
            ]

            print("\n=== 전체 평균 성능 ===")
            print(eval_df[metric_cols].mean(numeric_only=True))

            summary_path = out_dir / "metrics_summary.csv"
            eval_df[metric_cols].mean(numeric_only=True).to_csv(
                summary_path,
                header=["mean"],
                encoding="utf-8-sig"
            )
            print(f"\n요약 결과 저장: {summary_path}")
        else:
            print("\nGT depth를 찾지 못해 성능 평가는 수행하지 않았습니다.")


if __name__ == "__main__":
    main()