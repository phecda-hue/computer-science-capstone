"""
validate.py
PT / ONNX / TFLite 형식별 성능 검증

사용법:
    python validate.py \
        --yolo-pt   weights/yolo_best.pt \
        --yolo-onnx converted/yolo_sim.onnx \
        --yolo-tflite converted/yolo_fp32.tflite \
        --dav2-pt   weights/dav2_small.pt \
        --dav2-onnx converted/dav2_small_sim.onnx \
        --dav2-tflite converted/dav2_small_fp32.tflite \
        --test-dir  data/test \
        [--gt-labels data/test/labels]   # YOLO format gt (선택)
        [--gt-depth  data/test/depth]    # depth GT .npy (선택)
        [--warmup 3] \
        [--runs   20] \
        [--output results/validation_report.json]
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 로컬 유틸리티
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import (
    preprocess_yolo, preprocess_dav2,
    collect_test_images, postprocess_yolo,
    YOLO_SIZE, DAV2_SIZE,
)
from metrics import (
    compute_detection_metrics, aggregate_detection_metrics,
    compute_depth_metrics, compute_depth_consistency,
    compute_latency_stats,
)


# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    tag = {"INFO": "✅", "WARN": "⚠️ ", "ERR": "❌", "HEAD": "🔷"}.get(level, "  ")
    print(f"[{tag}] {msg}", flush=True)


def load_yolo_gt(label_dir: Optional[Path], image_path: Path, img_shape) -> list:
    """YOLO format txt GT 로드 → [{"box":[x1,y1,x2,y2], "class":int}]"""
    if label_dir is None:
        return []
    label_file = label_dir / (image_path.stem + ".txt")
    if not label_file.exists():
        return []
    h, w = img_shape
    gts = []
    for line in label_file.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        gts.append({"box": [x1, y1, x2, y2], "class": cls})
    return gts


def load_depth_gt(depth_dir: Optional[Path], image_path: Path) -> Optional[np.ndarray]:
    """GT depth 로드 (.npy 또는 16bit PNG)"""
    if depth_dir is None:
        return None
    npy = depth_dir / (image_path.stem + ".npy")
    if npy.exists():
        return np.load(str(npy)).astype(np.float32)
    png = depth_dir / (image_path.stem + ".png")
    if png.exists():
        return cv2.imread(str(png), cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
    return None


# ══════════════════════════════════════════════════════════════════════════════
# YOLO 추론 인터페이스 (PT / ONNX / TFLite 통일)
# ══════════════════════════════════════════════════════════════════════════════

class YoloRunner:
    def __init__(self, model_path: Path, fmt: str, size: int = YOLO_SIZE):
        self.path  = model_path
        self.fmt   = fmt          # "pt" | "onnx" | "tflite"
        self.size  = size
        self._init()

    def _init(self):
        if self.fmt == "pt":
            from ultralytics import YOLO
            self._model = YOLO(str(self.path))

        elif self.fmt == "onnx":
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._sess = ort.InferenceSession(
                str(self.path),
                sess_opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._sess.get_inputs()[0].name

        elif self.fmt == "tflite":
            import tensorflow as tf
            self._interp = tf.lite.Interpreter(model_path=str(self.path))
            self._interp.allocate_tensors()
            self._inp  = self._interp.get_input_details()[0]
            self._outp = self._interp.get_output_details()

    def infer(self, image_path: Path):
        """Returns (latency_ms, detections_list)"""
        blob, orig_shape, scale, pad = preprocess_yolo(image_path, self.size)

        t0 = time.perf_counter()

        if self.fmt == "pt":
            from ultralytics import YOLO
            import torch
            results = self._model.predict(str(image_path), verbose=False)
            raw_out = results[0].boxes
            t1 = time.perf_counter()
            dets = []
            if raw_out is not None:
                for box in raw_out:
                    dets.append({
                        "box":   box.xyxy[0].tolist(),
                        "score": float(box.conf),
                        "class": int(box.cls),
                    })
            return (t1 - t0) * 1000, dets

        elif self.fmt == "onnx":
            raw = self._sess.run(None, {self._input_name: blob})
            t1  = time.perf_counter()
            dets = postprocess_yolo(raw[0], orig_shape, scale, pad)
            return (t1 - t0) * 1000, dets

        elif self.fmt == "tflite":
            expected = tuple(self._inp["shape"])
            feed = blob.copy()

            # NCHW -> NHWC
            if len(expected) == 4 and expected[-1] == 3:
                feed = feed.transpose(0, 2, 3, 1)

            # 입력 양자화 처리
            if self._inp["dtype"] in [np.int8, np.uint8]:
                sc, zp = self._inp["quantization"]
                feed = (feed / sc + zp).astype(self._inp["dtype"])
            else:
                feed = feed.astype(self._inp["dtype"])

            self._interp.set_tensor(self._inp["index"], feed)
            self._interp.invoke()

        raw = self._interp.get_tensor(self._outp[0]["index"])

        if self._outp[0]["dtype"] in [np.int8, np.uint8]:
            sc, zp = self._outp[0]["quantization"]
            if sc != 0:
                raw = (raw.astype(np.float32) - zp) * sc

        t1 = time.perf_counter()

        # TFLite YOLO26 end2end output: (1, 6, 300)
        if raw.ndim == 3 and raw.shape[1] == 6:
            # raw: (1, 6, 300) → (300, 6)
            pred = raw[0].T

            dets = []
            h, w = orig_shape

            for p in pred:
                x1, y1, x2, y2, cls, score = p.tolist()

                if score < 0.25:
                    continue

                # class id 보정
                cls = int(round(cls))

                # bbox가 0~1 정규화 좌표인 경우
                if max(x1, y1, x2, y2) <= 1.5:
                    x1 *= w
                    x2 *= w
                    y1 *= h
                    y2 *= h

                dets.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score),
                    "class": cls,
                })

            return (t1 - t0) * 1000, dets

        # 일반 YOLO output인 경우만 postprocess 사용
        if raw.ndim == 3 and raw.shape[1] > raw.shape[2]:
            raw = raw.transpose(0, 2, 1)

        dets = postprocess_yolo(raw, orig_shape, scale, pad)
        pred = raw[0].T
        print(pred[:10])
        print("MODEL:", self.path)
        print("OUTPUT DETAILS:", [o["shape"] for o in self._outp])
        return (t1 - t0) * 1000, dets


# ══════════════════════════════════════════════════════════════════════════════
# DAv2 추론 인터페이스
# ══════════════════════════════════════════════════════════════════════════════

class DAv2Runner:
    def __init__(self, model_path: Path, fmt: str, size: int = DAV2_SIZE):
        self.path = model_path
        self.fmt  = fmt
        self.size = size
        self._init()

    def _init(self):
        # 변경 후
        if self.fmt == "pt":
            import torch
            dav2_metric = Path.cwd().resolve() / "Depth_Anything_V2" / "metric_depth"
            if str(dav2_metric) not in sys.path:
                sys.path.insert(0, str(dav2_metric))
            from depth_anything_v2.dpt import DepthAnythingV2
            model_cfg = dict(encoder="vits", features=64,
                            out_channels=[48, 96, 192, 384],
                            max_depth=80)
            self._model = DepthAnythingV2(**model_cfg)
            state = torch.load(str(self.path), map_location="cpu", weights_only=False)
            self._model.load_state_dict(
                state if "model" not in state else state["model"]
            )
            self._model.eval()

        elif self.fmt == "onnx":
            import onnxruntime as ort
            self._sess = ort.InferenceSession(
                str(self.path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._sess.get_inputs()[0].name

        elif self.fmt == "tflite":
            import tensorflow as tf
            self._interp = tf.lite.Interpreter(model_path=str(self.path))
            self._interp.allocate_tensors()
            self._inp  = self._interp.get_input_details()[0]
            self._outp = self._interp.get_output_details()

    def infer(self, image_path: Path):
        """Returns (latency_ms, depth_map: np.ndarray HxW)"""
        blob, orig_shape = preprocess_dav2(image_path, self.size)

        t0 = time.perf_counter()

        if self.fmt == "pt":
            import torch
            with torch.no_grad():
                out = self._model(torch.from_numpy(blob))
            if isinstance(out, dict):
                depth = out["depth"].squeeze().numpy()
            else:
                depth = out.squeeze().numpy()
            t1 = time.perf_counter()

        elif self.fmt == "onnx":
            raw = self._sess.run(None, {self._input_name: blob})
            t1  = time.perf_counter()
            depth = raw[0].squeeze()

        elif self.fmt == "tflite":
            expected = tuple(self._inp["shape"])

            feed = blob.copy()

            # NHWC 자동 감지: (1,3,518,518) → (1,518,518,3)
            if len(expected) == 4 and expected[3] == blob.shape[1]:
                feed = feed.transpose(0, 2, 3, 1)

            if self._inp["dtype"] == np.int8:
                sc, zp = self._inp["quantization"]
                feed = (feed / sc + zp).astype(np.int8)
            else:
                feed = feed.astype(self._inp["dtype"])

            self._interp.set_tensor(self._inp["index"], feed)
            self._interp.invoke()
            raw = self._interp.get_tensor(self._outp[0]["index"])

            if self._outp[0]["dtype"] == np.int8:
                sc, zp = self._outp[0]["quantization"]
                raw = (raw.astype(np.float32) - zp) * sc

            t1    = time.perf_counter()
            depth = raw.squeeze()

        # 원본 해상도로 복원
        depth_resized = cv2.resize(
            depth.astype(np.float32),
            (orig_shape[1], orig_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return (t1 - t0) * 1000, depth_resized


# ══════════════════════════════════════════════════════════════════════════════
# 단일 모델 검증 루프
# ══════════════════════════════════════════════════════════════════════════════

def validate_yolo_runner(
    runner: YoloRunner,
    images: list,
    gt_label_dir: Optional[Path],
    warmup: int = 3,
    runs:   int = 20,
) -> dict:
    log(f"  YOLO [{runner.fmt.upper()}] 워밍업 {warmup}회 ...", "INFO")
    for img in images[:warmup]:
        runner.infer(img)

    latencies, det_metrics = [], []
    sample_images = images[:runs]

    log(f"  YOLO [{runner.fmt.upper()}] {len(sample_images)}장 검증 중 ...")
    for img_path in sample_images:
        ms, dets = runner.infer(img_path)
        latencies.append(ms)

        img = cv2.imread(str(img_path))
        gts = load_yolo_gt(gt_label_dir, img_path, img.shape[:2])
        det_metrics.append(compute_detection_metrics(dets, gts))

    return {
        "latency":   compute_latency_stats(latencies),
        "detection": aggregate_detection_metrics(det_metrics),
        "num_images": len(sample_images),
    }


def validate_dav2_runner(
    runner: DAv2Runner,
    images: list,
    gt_depth_dir: Optional[Path],
    reference_depths: Optional[dict],   # {img_path_str: depth_array}  for consistency
    warmup: int = 3,
    runs:   int = 20,
) -> dict:
    log(f"  DAv2 [{runner.fmt.upper()}] 워밍업 {warmup}회 ...", "INFO")
    for img in images[:warmup]:
        runner.infer(img)

    latencies, depth_metrics, consistency_metrics = [], [], []
    sample_images = images[:runs]
    current_depths = {}

    log(f"  DAv2 [{runner.fmt.upper()}] {len(sample_images)}장 검증 중 ...")
    for img_path in sample_images:
        ms, depth = runner.infer(img_path)
        latencies.append(ms)
        current_depths[str(img_path)] = depth

        gt = load_depth_gt(gt_depth_dir, img_path)
        if gt is not None:
            depth_resized_gt = cv2.resize(depth, (gt.shape[1], gt.shape[0]))
            depth_metrics.append(compute_depth_metrics(depth_resized_gt, gt))

        if reference_depths and str(img_path) in reference_depths:
            ref = reference_depths[str(img_path)]
            consistency_metrics.append(
                compute_depth_consistency(depth, ref)
            )

    result = {
        "latency":    compute_latency_stats(latencies),
        "num_images": len(sample_images),
        "depths":     current_depths,   # consistency 계산용 (JSON 저장 전 제거)
    }
    if depth_metrics:
        agg = {k: round(float(np.mean([m[k] for m in depth_metrics])), 4)
               for k in depth_metrics[0]}
        result["depth_metrics"] = agg

    if consistency_metrics:
        agg = {k: round(float(np.mean([m[k] for m in consistency_metrics])), 4)
               for k in consistency_metrics[0]}
        result["consistency_vs_reference"] = agg

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 리포트 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(report: dict):
    print("\n" + "═" * 70)
    print("  📊 검증 결과 비교표")
    print("═" * 70)

    for model_key in ("yolo", "dav2"):
        model_data = report.get(model_key, {})
        if not model_data:
            continue
        label = "YOLO" if model_key == "yolo" else "DAv2 Small"
        print(f"\n  ▶ {label}")
        print(f"  {'형식':<10} {'평균(ms)':>9} {'P95(ms)':>9} {'FPS':>7}", end="")

        if model_key == "yolo":
            print(f"  {'Precision':>10} {'Recall':>8} {'F1':>8}")
        else:
            print(f"  {'AbsRel':>8} {'δ<1.25':>8} {'Spearman':>10}")

        print("  " + "-" * 68)
        for fmt in ("pt", "onnx", "tflite"):
            d = model_data.get(fmt)
            if not d:
                continue
            lat = d.get("latency", {})
            row = (f"  {fmt:<10} {lat.get('mean_ms','N/A'):>9} "
                   f"{lat.get('p95_ms','N/A'):>9} {lat.get('fps','N/A'):>7}")

            if model_key == "yolo":
                det = d.get("detection", {})
                row += (f"  {det.get('precision','N/A'):>10} "
                        f"{det.get('recall','N/A'):>8} "
                        f"{det.get('f1','N/A'):>8}")
            else:
                dm  = d.get("depth_metrics", {})
                con = d.get("consistency_vs_reference", {})
                row += (f"  {dm.get('AbsRel','N/A'):>8} "
                        f"{dm.get('d1','N/A'):>8} "
                        f"{con.get('spearman_corr','N/A'):>10}")
            print(row)

    print("\n" + "═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PT/ONNX/TFLite 성능 검증")

    # YOLO 모델 경로
    p.add_argument("--yolo-pt",     type=Path, default=None)
    p.add_argument("--yolo-onnx",   type=Path, default=None)
    p.add_argument("--yolo-tflite", type=Path, default=None)
    p.add_argument("--yolo-size",   type=int,  default=YOLO_SIZE)

    # DAv2 모델 경로
    p.add_argument("--dav2-pt",     type=Path, default=None)
    p.add_argument("--dav2-onnx",   type=Path, default=None)
    p.add_argument("--dav2-tflite", type=Path, default=None)
    p.add_argument("--dav2-size",   type=int,  default=DAV2_SIZE)

    # 데이터 경로
    p.add_argument("--test-dir",   type=Path, required=True, help="data/test 이미지 폴더")
    p.add_argument("--gt-labels",  type=Path, default=None,  help="YOLO format GT 레이블 폴더")
    p.add_argument("--gt-depth",   type=Path, default=None,  help="GT depth 폴더 (.npy / 16bit png)")

    # 검증 설정
    p.add_argument("--warmup", type=int, default=3,  help="워밍업 반복 횟수")
    p.add_argument("--runs",   type=int, default=20, help="검증 이미지 최대 수")
    p.add_argument("--output", type=Path, default=Path("results/validation_report.json"))

    return p.parse_args()


def main():
    args = parse_args()

    images = collect_test_images(args.test_dir)
    log(f"테스트 이미지 {len(images)}장 발견 (최대 {args.runs}장 사용)")

    report = {"yolo": {}, "dav2": {}}
    ref_depths_pt = {}   # PT 출력을 ONNX/TFLite consistency 기준으로 사용

    # ══ YOLO 검증 ════════════════════════════════════════════════════════════
    log("\n YOLO 검증 시작", "HEAD")
    yolo_configs = [
        ("pt",     args.yolo_pt),
        ("onnx",   args.yolo_onnx),
        ("tflite", args.yolo_tflite),
    ]
    for fmt, path in yolo_configs:
        if path is None or not path.exists():
            continue
        log(f"\n  YOLO [{fmt.upper()}]: {path.name}")
        try:
            runner = YoloRunner(path, fmt, args.yolo_size)
            result = validate_yolo_runner(
                runner, images,
                gt_label_dir=args.gt_labels,
                warmup=args.warmup,
                runs=args.runs,
            )
            report["yolo"][fmt] = result
            log(f"  → 평균 레이턴시: {result['latency']['mean_ms']} ms  "
                f"FPS: {result['latency']['fps']}")
        except Exception as e:
            log(f"  YOLO [{fmt}] 검증 실패: {e}", "ERR")
            import traceback; traceback.print_exc()

    # ══ DAv2 검증 ════════════════════════════════════════════════════════════
    log("\n DAv2 검증 시작", "HEAD")
    dav2_configs = [
        ("pt",     args.dav2_pt),
        ("onnx",   args.dav2_onnx),
        ("tflite", args.dav2_tflite),
    ]
    for fmt, path in dav2_configs:
        if path is None or not path.exists():
            continue
        log(f"\n  DAv2 [{fmt.upper()}]: {path.name}")
        try:
            runner = DAv2Runner(path, fmt, args.dav2_size)
            result = validate_dav2_runner(
                runner, images,
                gt_depth_dir=args.gt_depth,
                reference_depths=ref_depths_pt if fmt != "pt" else None,
                warmup=args.warmup,
                runs=args.runs,
            )
            # PT 결과를 기준 depth로 저장
            if fmt == "pt" and "depths" in result:
                ref_depths_pt = result["depths"]

            result.pop("depths", None)   # JSON 직렬화 전 np.ndarray 제거
            report["dav2"][fmt] = result
            log(f"  → 평균 레이턴시: {result['latency']['mean_ms']} ms  "
                f"FPS: {result['latency']['fps']}")
        except Exception as e:
            log(f"  DAv2 [{fmt}] 검증 실패: {e}", "ERR")
            import traceback; traceback.print_exc()

    # ══ 리포트 출력 및 저장 ═══════════════════════════════════════════════════
    print_comparison_table(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # np 타입 → Python 기본 타입으로 변환
    def to_serializable(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=to_serializable)
    log(f"검증 리포트 저장 완료: {args.output}")


if __name__ == "__main__":
    main()
