"""
validate_all.py
사이즈별 PT / ONNX / TFLite 일괄 성능 검증

폴더 구조 가정 (convert_all.py 실행 후):
    converted/
    ├── yolo_n/  yolo_sim.onnx, yolo_fp32.tflite
    ├── yolo_s/  ...
    ├── yolo_m/  ...
    ├── yolo_l/  ...
    ├── yolo_x/  ...
    └── dav2/    dav2_small_sim.onnx, dav2_small_fp32.tflite

사용법:
    python validate_all.py \
        --detect-dir  runs/detect \
        --converted   converted/ \
        --dav2-pt     checkpoints/depth_anything_v2_metric_vkitti_vits.pth \
        --test-dir    data/test \
        [--gt-labels  data/test/labels] \
        [--gt-depth   data/test/depth] \
        [--sizes n s m l x] \
        [--formats pt onnx tflite] \
        [--runs 30]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import collect_test_images, YOLO_SIZE, DAV2_SIZE
from validate import (
    YoloRunner, DAv2Runner,
    validate_yolo_runner, validate_dav2_runner,
    load_yolo_gt, load_depth_gt,
    log,
)

SIZE_PATTERNS = {
    'n': ['yolo26n', 'yolov8n', 'yolo_n'],
    's': ['yolo26s', 'yolov8s', 'yolo_s'],
    'm': ['yolo26m', 'yolov8m', 'yolo_m'],
    'l': ['yolo26l', 'yolov8l', 'yolo_l'],
    'x': ['yolo26x', 'yolov8x', 'yolo_x'],
}


# ══════════════════════════════════════════════════════════════════════════════
# 경로 탐색 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def find_pt(detect_dir: Path, size: str) -> Path | None:
    """runs/detect/ 에서 사이즈별 best.pt 탐색"""
    for folder in detect_dir.iterdir():
        if not folder.is_dir():
            continue
        if any(p in folder.name.lower() for p in SIZE_PATTERNS[size]):
            for name in [f"best_{size}.pt", "best.pt"]:
                p = folder / "weights" / name
                if p.exists():
                    return p
    return None


def find_converted(converted_dir: Path, size: str, fmt: str) -> Path | None:
    """converted/yolo_{size}/ 에서 ONNX 또는 TFLite 탐색"""
    folder = converted_dir / f"yolo_{size}"
    if not folder.exists():
        return None

    if fmt == "onnx":
        candidates = [
            "best.onnx",
            f"best_{size}.onnx",
            "yolo.onnx",
            "yolo_sim.onnx",
        ]

        for name in candidates:
            p = folder / name
            if p.exists():
                return p

        # 그래도 없으면 폴더 안의 onnx 아무거나 탐색
        files = sorted(folder.glob("*.onnx"))
        if files:
            return files[0]

    elif fmt == "tflite":
        candidates = [
            "best_fp32.tflite",
            "best_int8.tflite",
            f"best_{size}_fp32.tflite",
            f"best_{size}_int8.tflite",
            "yolo_fp32.tflite",
            "yolo_int8.tflite",
        ]

        for name in candidates:
            p = folder / name
            if p.exists():
                return p

        # 그래도 없으면 폴더 안의 tflite 아무거나 탐색
        files = sorted(folder.glob("*.tflite"))
        if files:
            return files[0]

    return None


def find_dav2_converted(converted_dir: Path, fmt: str) -> Path | None:
    folder = converted_dir / "dav2"
    if not folder.exists():
        return None
    if fmt == "onnx":
        for name in ["dav2_small_sim.onnx", "dav2_small.onnx"]:
            p = folder / name
            if p.exists():
                return p
    elif fmt == "tflite":
        for name in ["dav2_small_int8.tflite", "dav2_small_fp32.tflite"]:
            p = folder / name
            if p.exists():
                return p
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 비교표 출력
# ══════════════════════════════════════════════════════════════════════════════

def print_full_table(all_results: dict):
    print("\n" + "═" * 80)
    print("  📊 전체 검증 결과 비교표")
    print("═" * 80)

    # ── YOLO 표 ───────────────────────────────────────────────────────────────
    yolo_data = {k: v for k, v in all_results.items() if k != 'dav2'}
    if yolo_data:
        print(f"\n  ▶ YOLO — 레이턴시 / 검출 성능")
        print(f"  {'모델':<12} {'형식':<8} {'mean(ms)':>9} {'P95(ms)':>9} {'FPS':>7}  {'Prec':>7} {'Recall':>7} {'F1':>7}")
        print("  " + "-" * 74)
        for size in ['n', 's', 'm', 'l', 'x']:
            if size not in yolo_data:
                continue
            for fmt in ['pt', 'onnx', 'tflite']:
                d = yolo_data[size].get(fmt)
                if not d:
                    continue
                lat = d.get('latency', {})
                det = d.get('detection', {})
                label = f"YOLO-{size.upper()}"
                print(
                    f"  {label:<12} {fmt:<8}"
                    f" {lat.get('mean_ms','N/A'):>9}"
                    f" {lat.get('p95_ms','N/A'):>9}"
                    f" {lat.get('fps','N/A'):>7}"
                    f"  {det.get('precision','N/A'):>7}"
                    f" {det.get('recall','N/A'):>7}"
                    f" {det.get('f1','N/A'):>7}"
                )

    # ── DAv2 표 ───────────────────────────────────────────────────────────────
    dav2_data = all_results.get('dav2', {})
    if dav2_data:
        print(f"\n  ▶ DAv2 Small — 레이턴시 / 깊이 추정")
        print(f"  {'형식':<8} {'mean(ms)':>9} {'P95(ms)':>9} {'FPS':>7}  {'AbsRel':>8} {'δ<1.25':>8} {'Spearman':>10}")
        print("  " + "-" * 68)
        for fmt in ['pt', 'onnx', 'tflite']:
            d = dav2_data.get(fmt)
            if not d:
                continue
            lat = d.get('latency', {})
            dm  = d.get('depth_metrics', {})
            con = d.get('consistency_vs_reference', {})
            print(
                f"  {fmt:<8}"
                f" {lat.get('mean_ms','N/A'):>9}"
                f" {lat.get('p95_ms','N/A'):>9}"
                f" {lat.get('fps','N/A'):>7}"
                f"  {dm.get('AbsRel','N/A'):>8}"
                f" {dm.get('d1','N/A'):>8}"
                f" {con.get('spearman_corr','N/A'):>10}"
            )

    # ── 모바일 추천 ───────────────────────────────────────────────────────────
    print(f"\n  ▶ 모바일 배포 추천 (ONNX 기준 FPS 상위)")
    candidates = []
    for size in ['n','s','m','l','x']:
        d = yolo_data.get(size, {}).get('onnx', {})
        if d and d.get('latency'):
            candidates.append((size, d['latency'].get('fps', 0), d.get('detection',{}).get('f1', 0)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    for size, fps, f1 in candidates[:3]:
        print(f"    YOLO-{size.upper()}: {fps:.1f} FPS  F1={f1:.4f}")

    print("═" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="YOLO 전 사이즈 + DAv2 일괄 검증")
    p.add_argument("--detect-dir",  type=Path, default=Path("runs/detect"))
    p.add_argument("--converted",   type=Path, default=Path("converted"))
    p.add_argument("--dav2-pt",     type=Path, default=None)
    p.add_argument("--test-dir",    type=Path, required=True)
    p.add_argument("--gt-labels",   type=Path, default=None)
    p.add_argument("--gt-depth",    type=Path, default=None)
    p.add_argument("--sizes",       nargs="+", default=["n","s","m","l","x"],
                   choices=["n","s","m","l","x"])
    p.add_argument("--formats",     nargs="+", default=["pt","onnx","tflite"],
                   choices=["pt","onnx","tflite"])
    p.add_argument("--yolo-size",   type=int, default=YOLO_SIZE)
    p.add_argument("--dav2-size",   type=int, default=DAV2_SIZE)
    p.add_argument("--warmup",      type=int, default=3)
    p.add_argument("--runs",        type=int, default=30)
    p.add_argument("--skip-dav2",   action="store_true")
    p.add_argument("--skip-yolo",   action="store_true")
    p.add_argument("--output",      type=Path, default=Path("results/validation_all.json"))
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    images = collect_test_images(args.test_dir)

    if args.gt_labels:
        label_files = list(args.gt_labels.glob("*.txt"))
        log(f"GT 라벨 폴더: {args.gt_labels}")
        log(f"GT 라벨 파일 수: {len(label_files)}")

        if len(label_files) == 0:
            log("GT 라벨 txt 파일이 없습니다. --gt-labels 경로를 확인하세요.", "ERR")

        missing = []
        for img in images[:10]:
            label_path = args.gt_labels / f"{img.stem}.txt"
            if not label_path.exists():
                missing.append(img.name)

        if missing:
            log(f"앞 10개 이미지 중 라벨이 없는 이미지: {missing}", "WARN")
    else:
        log("--gt-labels가 지정되지 않았습니다. F1 계산이 0으로 나올 수 있습니다.", "WARN")

    log(f"테스트 이미지 {len(images)}장 (최대 {args.runs}장 사용)")

    all_results = {}
    ref_depths  = {}   # DAv2 PT 출력 → consistency 기준

    # ══ YOLO 사이즈별 검증 ════════════════════════════════════════════════════
    if not args.skip_yolo:
        total = len(args.sizes) * len(args.formats)
        done  = 0

        for size in args.sizes:
            all_results[size] = {}

            for fmt in args.formats:
                done += 1
                label = f"YOLO-{size.upper()} [{fmt}]"
                print(f"\n[{done:2d}/{total}] {label}")

                # 경로 탐색
                if fmt == "pt":
                    path = find_pt(args.detect_dir, size)
                elif fmt == "onnx":
                    path = find_converted(args.converted, size, "onnx")
                else:
                    path = find_converted(args.converted, size, "tflite")

                if not path or not path.exists():
                    log(f"  {label} 파일 없음 — 건너뜀", "WARN")
                    continue

                log(f"  파일: {path.name}")
                try:
                    runner = YoloRunner(path, fmt, args.yolo_size)
                    result = validate_yolo_runner(
                        runner, images,
                        gt_label_dir=args.gt_labels,
                        warmup=args.warmup,
                        runs=args.runs,
                    )
                    all_results[size][fmt] = result
                    log(f"  → {result['latency']['mean_ms']}ms  FPS:{result['latency']['fps']}"
                        f"  F1:{result['detection'].get('f1','N/A')}")
                except Exception as e:
                    log(f"  {label} 검증 실패: {e}", "ERR")
                    import traceback; traceback.print_exc()

    # ══ DAv2 검증 ════════════════════════════════════════════════════════════
    if not args.skip_dav2:
        all_results['dav2'] = {}
        dav2_configs = []

        if 'pt' in args.formats and args.dav2_pt and args.dav2_pt.exists():
            dav2_configs.append(('pt', args.dav2_pt))
        if 'onnx' in args.formats:
            p = find_dav2_converted(args.converted, 'onnx')
            if p:
                dav2_configs.append(('onnx', p))
        if 'tflite' in args.formats:
            p = find_dav2_converted(args.converted, 'tflite')
            if p:
                dav2_configs.append(('tflite', p))

        for fmt, path in dav2_configs:
            log(f"\nDAv2 [{fmt.upper()}]: {path.name}")
            try:
                runner = DAv2Runner(path, fmt, args.dav2_size)
                result = validate_dav2_runner(
                    runner, images,
                    gt_depth_dir=args.gt_depth,
                    reference_depths=ref_depths if fmt != 'pt' else None,
                    warmup=args.warmup,
                    runs=args.runs,
                )
                if fmt == 'pt' and 'depths' in result:
                    ref_depths = result['depths']
                result.pop('depths', None)
                all_results['dav2'][fmt] = result
                log(f"  → {result['latency']['mean_ms']}ms  FPS:{result['latency']['fps']}")
            except Exception as e:
                log(f"  DAv2 [{fmt}] 실패: {e}", "ERR")
                import traceback; traceback.print_exc()

    # ══ 결과 출력 및 저장 ═════════════════════════════════════════════════════
    print_full_table(all_results)

    elapsed = time.time() - t_start
    log(f"전체 소요 시간: {elapsed/60:.1f}분")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32,   np.int64)):   return int(obj)
        raise TypeError(type(obj))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=to_serializable)
    log(f"검증 결과 저장: {args.output}")


if __name__ == "__main__":
    main()