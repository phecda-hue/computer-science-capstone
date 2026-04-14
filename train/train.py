# ================================================
# SmartView - 시각장애인 보행 보조 시스템
# YOLO26 전체 사이즈 학습 스크립트 (n/s/m/l/x)
# CLI 기반 멀티 GPU 분리 실행 버전
# 실행 예시:
#   python train.py --gpu 0 --sizes n,m,x
#   python train.py --gpu 1 --sizes s,l
# ================================================

from pathlib import Path
import argparse
import os

# ────────────────────────────────────────────────
# 경로/상수 설정
# ────────────────────────────────────────────────
ROOT = Path(r"C:\Users\SBL-336server2\Desktop\컴공 캡스톤")
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_YAML = BASE_DIR / "data" / "data.yaml"
PROJECT_DIR = BASE_DIR / "runs" / "detect"

MODEL_PATHS = {
    "n": ROOT / "models" / "yolo26n.pt",
    "s": ROOT / "models" / "yolo26s.pt",
    "m": ROOT / "models" / "yolo26m.pt",
    "l": ROOT / "models" / "yolo26l.pt",
    "x": ROOT / "models" / "yolo26x.pt",
}

CLASS_NAMES = ["stairs", "curb", "manhole", "person", "hole", "vehicle"]


# ────────────────────────────────────────────────
# 설정 함수
# ────────────────────────────────────────────────
def get_config():
    import torch

    if torch.cuda.is_available():
        epochs = 200
        img_size = 640
        batch_size = 64
        sizes = ["n", "s", "m", "l", "x"]
    else:
        epochs = 5
        img_size = 640
        batch_size = 4
        sizes = ["n"]

    return epochs, img_size, batch_size, sizes


def get_batch_by_size():
    import torch

    if torch.cuda.is_available():
        # GPU 학습 시 모델 크기에 맞게 보수적으로 설정
        return {
            "n": 64,
            "s": 64,
            "m": 32,
            "l": 16,
            "x": 8,
        }
    else:
        _, _, batch_size, _ = get_config()
        return {
            "n": batch_size,
            "s": batch_size,
            "m": batch_size,
            "l": batch_size,
            "x": batch_size,
        }


# ────────────────────────────────────────────────
# 유틸 함수
# ────────────────────────────────────────────────
def check_model_files(target_sizes):
    """학습 전 필요한 모델 파일 존재 여부 확인"""
    print("\n📂 모델 파일 확인 중...")
    all_exist = True

    for size in target_sizes:
        path = MODEL_PATHS[size]
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} yolo26{size}.pt → {path}")
        if not exists:
            all_exist = False

    if not all_exist:
        raise FileNotFoundError(
            "\n일부 모델 파일이 없습니다. "
            "models 폴더 안의 필요한 yolo26*.pt 파일을 확인해주세요."
        )

    print("모든 모델 파일 확인 완료!\n")


def parse_sizes_arg(sizes_arg, available_sizes):
    """--sizes n,m,x 형태를 리스트로 변환하고 유효성 검사"""
    if sizes_arg is None:
        return available_sizes

    selected = [s.strip().lower() for s in sizes_arg.split(",") if s.strip()]
    valid_set = set(MODEL_PATHS.keys())

    for s in selected:
        if s not in valid_set:
            raise ValueError(
                f"잘못된 모델 크기 '{s}'입니다. 사용 가능: {sorted(valid_set)}"
            )

    return selected


# ────────────────────────────────────────────────
# 학습 함수
# ────────────────────────────────────────────────
def train_model(
    size: str,
    device,
    data_yaml,
    project_dir,
    model_paths,
    class_names,
    epochs,
    img_size,
    batch_by_size,
):
    import torch
    from ultralytics import YOLO

    batch = batch_by_size[size]
    model_path = model_paths[size]

    if size in ["l", "x"] and batch >= 32:
        print(f"⚠️ YOLO26{size.upper()} + batch={batch} → VRAM 부족 가능성 있음")
        print("   VRAM 부족 시 BATCH_BY_SIZE 값을 낮춰주세요.\n")

    print(f"\n{'=' * 60}")
    print(f"🚀 YOLO26{size.upper()} 학습 시작")
    print(f"   모델 경로 : {model_path}")
    print(f"   Batch size : {batch}")
    print(f"   Epochs     : {epochs}")
    print(f"   Image size : {img_size}")
    print(f"   Device     : {device}")
    print(f"{'=' * 60}\n")

    try:
        model = YOLO(str(model_path))

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            device=device,
            project=str(project_dir),
            name=f"yolo26{size}_6cls_ep{epochs}_bs{batch}",
            exist_ok=True,

            # 학습 설정
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            patience=0,
            amp=True,
            workers=2,
            seed=42,

            # 저장 설정
            save=True,
            save_period=-1,
            verbose=True,

            # 증강 설정
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            fliplr=0.0,
            mosaic=1.0,
            mixup=0.1,
        )

        print(f"\n✅ YOLO26{size.upper()} 학습 완료!")
        print(f"   저장 경로 : {results.save_dir}")
        print(f"   best.pt   : {results.save_dir}\\weights\\best.pt")
        print(f"   last.pt   : {results.save_dir}\\weights\\last.pt\n")

        print(f"📊 YOLO26{size.upper()} 테스트셋 평가 중...\n")

        best_model = YOLO(str(results.save_dir / "weights" / "best.pt"))
        metrics = best_model.val(
            data=str(data_yaml),
            split="test",
            imgsz=img_size,
            device=device,
        )

        print(f"\n{'─' * 40}")
        print(f"📊 YOLO26{size.upper()} 평가 결과")
        print(f"  mAP@50    : {metrics.box.map50:.4f}")
        print(f"  mAP@50-95 : {metrics.box.map:.4f}")
        print(f"  Precision : {metrics.box.mp:.4f}")
        print(f"  Recall    : {metrics.box.mr:.4f}")
        print(f"{'─' * 40}")
        print("  클래스별 mAP@50")
        for name, ap in zip(class_names, metrics.box.ap50):
            print(f"    {name:10s}: {ap:.4f}")
        print(f"{'─' * 40}\n")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA OOM 발생! → YOLO26{size.upper()} batch={batch}")
            print(f"   BATCH_BY_SIZE['{size}'] 값을 더 낮춰서 다시 시도하세요.\n")
        else:
            print(f"\n❌ 학습 중 에러 발생 (YOLO26{size.upper()}): {e}\n")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ────────────────────────────────────────────────
# 메인 함수
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, help="사용할 물리 GPU 번호")
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="학습할 모델 크기 목록. 예: n,m,x",
    )
    args = parser.parse_args()

    # GPU 지정은 torch import 전에 해야 함
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    epochs, img_size, _, available_sizes = get_config()
    batch_by_size = get_batch_by_size()
    sizes = parse_sizes_arg(args.sizes, available_sizes)

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml 파일이 없습니다: {DATA_YAML}")

    if not torch.cuda.is_available():
        device = "cpu"
    else:
        # CUDA_VISIBLE_DEVICES로 하나만 보이게 제한했으므로 내부에서는 0번 사용
        device = 0

    print("\n" + "=" * 60)
    print("SmartView YOLO26 CLI 학습")
    print("=" * 60)
    print(f"입력 GPU       : {args.gpu}")
    print(f"실행 Device    : {device}")
    print(f"학습 모델      : {sizes}")
    print(f"Epochs         : {epochs}")
    print(f"Image size     : {img_size}")
    print(f"데이터 경로    : {DATA_YAML}")
    print(f"결과 저장      : {PROJECT_DIR}")

    if torch.cuda.is_available():
        print(f"논리 GPU 개수   : {torch.cuda.device_count()}")
        print(f"현재 GPU 이름   : {torch.cuda.get_device_name(0)}")
        print(
            f"현재 GPU 메모리 : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("현재 장치       : CPU")
    print("=" * 60)

    check_model_files(sizes)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    for size in sizes:
        train_model(
            size=size,
            device=device,
            data_yaml=DATA_YAML,
            project_dir=PROJECT_DIR,
            model_paths=MODEL_PATHS,
            class_names=CLASS_NAMES,
            epochs=epochs,
            img_size=img_size,
            batch_by_size=batch_by_size,
        )

    print("\n" + "=" * 60)
    print("🎉 지정한 모델 학습 완료!")
    print(f"결과 폴더: {PROJECT_DIR}")
    for size in sizes:
        batch = batch_by_size[size]
        run_name = f"yolo26{size}_6cls_ep{epochs}_bs{batch}"
        print(f"  YOLO26{size.upper()} → {PROJECT_DIR}\\{run_name}\\weights\\best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()