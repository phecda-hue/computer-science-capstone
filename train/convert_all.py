"""
convert_all.py
사이즈별 YOLO 모델 전체 + DAv2 일괄 변환

폴더 구조 가정:
    runs/detect/
    ├── yolo26l_6cls_ep200_bs16/weights/best.pt
    ├── yolo26m_6cls_ep200_bs32/weights/best.pt
    ├── yolo26n_6cls_ep200_bs64/weights/best.pt
    ├── yolo26s_6cls_ep200_bs64/weights/best.pt
    └── yolo26x_6cls_ep200_bs8/ weights/best.pt

사용법:
    # 전체 변환 (YOLO 5종 + DAv2)
    python convert_all.py --detect-dir runs/detect --dav2 checkpoints/depth_anything_v2_metric_vkitti_vits.pth

    # YOLO만 변환
    python convert_all.py --detect-dir runs/detect --skip-dav2

    # INT8 양자화 포함
    python convert_all.py --detect-dir runs/detect --dav2 checkpoints/dav2.pth --quantize --calib-images data/test
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 기존 convert.py의 함수들을 그대로 재사용
sys.path.insert(0, str(Path(__file__).parent))
from convert import (
    log, convert_yolo, convert_dav2, print_summary
)


# ══════════════════════════════════════════════════════════════════════════════
# 폴더 구조 자동 탐색
# ══════════════════════════════════════════════════════════════════════════════

SIZE_PATTERNS = {
    'n': ['yolo26n', 'yolov8n', 'yolo_n'],
    's': ['yolo26s', 'yolov8s', 'yolo_s'],
    'm': ['yolo26m', 'yolov8m', 'yolo_m'],
    'l': ['yolo26l', 'yolov8l', 'yolo_l'],
    'x': ['yolo26x', 'yolov8x', 'yolo_x'],
}

def find_best_pt(weights_dir: Path, size: str) -> Path | None:
    """weights/ 폴더에서 best checkpoint 탐색 (우선순위 순)"""
    candidates = [
        f"best_{size}.pt",   # best_l.pt 형식
        "best.pt",           # 기본 best.pt
        f"best_{size.upper()}.pt",
    ]
    for name in candidates:
        p = weights_dir / name
        if p.exists():
            return p
    return None


def discover_models(detect_dir: Path) -> dict:
    """
    runs/detect/ 아래 폴더를 스캔해서
    {'n': Path, 's': Path, ...} 형태로 반환
    """
    found = {}
    if not detect_dir.exists():
        log(f"detect 폴더를 찾을 수 없습니다: {detect_dir}", "ERR")
        return found

    for folder in sorted(detect_dir.iterdir()):
        if not folder.is_dir():
            continue
        folder_lower = folder.name.lower()
        for size, patterns in SIZE_PATTERNS.items():
            if any(p in folder_lower for p in patterns):
                weights_dir = folder / "weights"
                pt = find_best_pt(weights_dir, size)
                if pt:
                    found[size] = pt
                    log(f"  [{size.upper()}] 발견: {pt.relative_to(detect_dir.parent)}")
                else:
                    log(f"  [{size.upper()}] weights/ 폴더에 best.pt 없음: {folder.name}", "WARN")
                break

    return found


# ══════════════════════════════════════════════════════════════════════════════
# 진행률 표시
# ══════════════════════════════════════════════════════════════════════════════

def progress_bar(current: int, total: int, label: str = "") -> str:
    filled = int(20 * current / total)
    bar = "█" * filled + "░" * (20 - filled)
    return f"[{bar}] {current}/{total}  {label}"


# ══════════════════════════════════════════════════════════════════════════════
# 전체 변환 결과 요약
# ══════════════════════════════════════════════════════════════════════════════

def print_all_summary(all_results: dict):
    print("\n" + "═" * 70)
    print("  전체 변환 결과 요약")
    print("═" * 70)

    # YOLO 사이즈별
    yolo_results = {k: v for k, v in all_results.items() if k != 'dav2'}
    if yolo_results:
        print(f"\n  {'모델':<10} {'PT':^6} {'ONNX':^6} {'TFLite':^8} {'ONNX 크기':>10} {'TFLite 크기':>12}")
        print("  " + "-" * 56)
        for size in ['n', 's', 'm', 'l', 'x']:
            res = yolo_results.get(size, {})
            if not res:
                continue
            pt_ok     = "✅" if res.get("pt")     else "❌"
            onnx_ok   = "✅" if res.get("onnx")   else "❌"
            tflite_ok = "✅" if res.get("tflite") else "❌"

            onnx_size   = f"{Path(res['onnx']).stat().st_size/1e6:.1f}MB"   if res.get("onnx")   and Path(res["onnx"]).exists()   else "N/A"
            tflite_size = f"{Path(res['tflite']).stat().st_size/1e6:.1f}MB" if res.get("tflite") and Path(res["tflite"]).exists() else "N/A"

            print(f"  YOLO-{size.upper():<5} {pt_ok:^6} {onnx_ok:^6} {tflite_ok:^8} {onnx_size:>10} {tflite_size:>12}")

    # DAv2
    dav2 = all_results.get('dav2', {})
    if dav2:
        print(f"\n  {'모델':<10} {'PT':^6} {'ONNX':^6} {'TFLite':^8} {'ONNX 크기':>10} {'TFLite 크기':>12}")
        print("  " + "-" * 56)
        onnx_size   = f"{Path(dav2['onnx']).stat().st_size/1e6:.1f}MB"   if dav2.get("onnx")   and Path(dav2["onnx"]).exists()   else "N/A"
        tflite_size = f"{Path(dav2['tflite']).stat().st_size/1e6:.1f}MB" if dav2.get("tflite") and Path(dav2["tflite"]).exists() else "N/A"
        print(f"  {'DAv2-S':<10} {'✅':^6} {('✅' if dav2.get('onnx') else '❌'):^6} {('✅' if dav2.get('tflite') else '❌'):^8} {onnx_size:>10} {tflite_size:>12}")

    print("═" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="YOLO 전 사이즈 + DAv2 일괄 변환")
    p.add_argument("--detect-dir",   type=Path, default=Path("runs/detect"),
                   help="YOLO 학습 결과 루트 (runs/detect)")
    p.add_argument("--dav2",         type=Path, default=None,
                   help="DAv2 .pth 체크포인트 경로")
    p.add_argument("--outdir",       type=Path, default=Path("converted"),
                   help="변환 결과 저장 루트 폴더")
    p.add_argument("--yolo-size",    type=int,  default=640)
    p.add_argument("--dav2-size",    type=int,  default=518)
    p.add_argument("--quantize",     action="store_true", help="TFLite INT8 양자화")
    p.add_argument("--calib-images", type=Path, default=None)
    p.add_argument("--sizes",        nargs="+", default=["n","s","m","l","x"],
                   choices=["n","s","m","l","x"],
                   help="변환할 사이즈 선택 (기본: 전체)")
    p.add_argument("--skip-dav2",    action="store_true")
    p.add_argument("--skip-yolo",    action="store_true")
    p.add_argument("--save-json",    type=Path, default=Path("converted/conversion_summary.json"),
                   help="변환 결과 JSON 저장 경로")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    print("\n" + "═" * 70)
    print("  YOLO 전 사이즈 + DAv2 일괄 변환")
    print("═" * 70)

    all_results = {}

    # ── YOLO 탐색 ─────────────────────────────────────────────────────────────
    if not args.skip_yolo:
        log("\n학습 폴더 탐색 중...")
        model_map = discover_models(args.detect_dir)

        # --sizes 필터 적용
        target_sizes = [s for s in args.sizes if s in model_map]
        missing      = [s for s in args.sizes if s not in model_map]
        if missing:
            log(f"탐색되지 않은 사이즈: {missing} — 건너뜀", "WARN")

        if not target_sizes:
            log("변환할 YOLO 모델이 없습니다.", "ERR")
        else:
            log(f"\n변환 대상: {[s.upper() for s in target_sizes]}  ({len(target_sizes)}개)\n")

            for i, size in enumerate(target_sizes, 1):
                pt_path = model_map[size]
                print(f"\n{progress_bar(i, len(target_sizes), f'YOLO-{size.upper()}')}")
                log(f"YOLO-{size.upper()} 변환 시작: {pt_path.name}")

                # 사이즈별 출력 폴더: converted/yolo_n/, converted/yolo_s/, ...
                outdir = args.outdir / f"yolo_{size}"

                result = convert_yolo(
                    pt_path=pt_path,
                    outdir=outdir,
                    size=args.yolo_size,
                    quantize=args.quantize,
                    calib_dir=args.calib_images,
                )
                all_results[size] = result

    # ── DAv2 변환 ─────────────────────────────────────────────────────────────
    if not args.skip_dav2:
        if args.dav2 is None:
            log("--dav2 경로가 지정되지 않았습니다. DAv2 변환 건너뜀.", "WARN")
        elif not args.dav2.exists():
            log(f"DAv2 체크포인트를 찾을 수 없습니다: {args.dav2}", "ERR")
        else:
            print(f"\n{'─'*70}")
            log("DAv2 Small (Metric) 변환 시작")
            outdir = args.outdir / "dav2"
            dav2_result = convert_dav2(
                pt_path=args.dav2,
                outdir=outdir,
                size=args.dav2_size,
                quantize=args.quantize,
                calib_dir=args.calib_images,
            )
            all_results['dav2'] = dav2_result

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    print_all_summary(all_results)

    elapsed = time.time() - t_start
    log(f"전체 소요 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")

    # JSON 저장
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    log(f"변환 결과 저장: {args.save_json}")


if __name__ == "__main__":
    main()