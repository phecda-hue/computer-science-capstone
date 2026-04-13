# ================================================
# SmartView - 시각장애인 보행 보조 시스템
# YOLO26 전체 사이즈 학습 스크립트 (n/s/m/l/x)
# ================================================

from ultralytics import YOLO
from pathlib import Path
import torch
import multiprocessing as mp

# ────────────────────────────────────────────────
# 설정값 (필요 시 여기만 수정)
# ────────────────────────────────────────────────
EPOCHS     = 200
IMG_SIZE   = 640
BATCH_SIZE = 64
SIZES      = ["n", "s", "m", "l", "x"]   # YOLO26 전체 사이즈

ROOT        = Path(r"C:\Users\USER\Desktop\1")
DATA_YAML   = ROOT / "data.yaml"
PROJECT_DIR = ROOT / "runs" / "detect"

# 로컬 모델 경로 (ROOT 폴더에 직접 넣은 .pt 파일)
MODEL_PATHS = {
    "n": ROOT / "yolo26n.pt",
    "s": ROOT / "yolo26s.pt",
    "m": ROOT / "yolo26m.pt",
    "l": ROOT / "yolo26l.pt",
    "x": ROOT / "yolo26x.pt",
}

# 사이즈별 배치 크기 (VRAM 부족 시 개별 조정)
# 권장 가이드
# n, s → 64 OK
# m    → 32~64
# l    → 16~32
# x    → 8~16
BATCH_BY_SIZE = {
    "n": BATCH_SIZE,
    "s": BATCH_SIZE,
    "m": BATCH_SIZE,
    "l": BATCH_SIZE,
    "x": BATCH_SIZE,
}

# 클래스 이름 (data.yaml과 동일한 순서)
CLASS_NAMES = ["stairs", "curb", "manhole", "person", "hole", "vehicle"]
# ────────────────────────────────────────────────


def check_model_files():
    """학습 전 모델 파일 존재 여부 확인"""
    print("\n📂 모델 파일 확인 중...")
    all_exist = True
    for size, path in MODEL_PATHS.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} yolo26{size}.pt → {path}")
        if not exists:
            all_exist = False
    if not all_exist:
        raise FileNotFoundError(
            "\n일부 모델 파일이 없습니다. "
            f"C:\\Users\\USER\\Desktop\\1 폴더에 .pt 파일을 확인해주세요."
        )
    print("모든 모델 파일 확인 완료!\n")


def train_model(size: str, device: int):
    """단일 YOLO26 사이즈 학습"""
    batch      = BATCH_BY_SIZE[size]
    model_path = MODEL_PATHS[size]

    # l, x 사이즈 VRAM 경고
    if size in ["l", "x"] and batch >= 32:
        print(f"⚠️  YOLO26{size.upper()} + batch={batch} → VRAM 부족 가능성 있음")
        print("   VRAM 부족 시 BATCH_BY_SIZE 값을 낮춰주세요\n")

    print(f"\n{'='*60}")
    print(f"🚀 YOLO26{size.upper()} 학습 시작")
    print(f"   모델 경로 : {model_path}")
    print(f"   Batch size : {batch}")
    print(f"   Epochs     : {EPOCHS}")
    print(f"   Image size : {IMG_SIZE}")
    print(f"   Device     : CUDA:{device}")
    print(f"{'='*60}\n")

    try:
        model = YOLO(str(model_path))

        results = model.train(
            data         = str(DATA_YAML),
            epochs       = EPOCHS,
            imgsz        = IMG_SIZE,
            batch        = batch,
            device       = device,
            project      = str(PROJECT_DIR),
            name         = f"yolo26{size}_6cls_ep{EPOCHS}_bs{batch}",
            exist_ok     = True,

            # 학습 설정
            lr0          = 0.001,
            lrf          = 0.01,
            momentum     = 0.937,
            weight_decay = 0.0005,
            patience     = 0,          # 조기종료 비활성화 (200 epoch 완주)
            amp          = True,       # 혼합 정밀도 (학습 속도 향상)
            workers      = 4,
            seed         = 42,

            # 저장 설정
            save         = True,
            save_period  = -1,         # best.pt + last.pt만 저장
            verbose      = True,

            # 증강 설정
            hsv_h        = 0.0,
            hsv_s        = 0.0,
            hsv_v        = 0.0,
            degrees      = 0.0,
            fliplr       = 0.0,
            mosaic       = 1.0,
            mixup        = 0.1,
        )

        print(f"\n✅ YOLO26{size.upper()} 학습 완료!")
        print(f"   저장 경로 : {results.save_dir}")
        print(f"   best.pt   : {results.save_dir}\\weights\\best.pt")
        print(f"   last.pt   : {results.save_dir}\\weights\\last.pt\n")

        # ── 테스트셋 평가 ──────────────────────────
        print(f"📊 YOLO26{size.upper()} 테스트셋 평가 중...\n")

        best_model = YOLO(str(results.save_dir / "weights" / "best.pt"))
        metrics = best_model.val(
            data   = str(DATA_YAML),
            split  = "test",
            imgsz  = IMG_SIZE,
            device = device,
        )

        print(f"\n{'─'*40}")
        print(f"📊 YOLO26{size.upper()} 평가 결과")
        print(f"  mAP@50    : {metrics.box.map50:.4f}")
        print(f"  mAP@50-95 : {metrics.box.map:.4f}")
        print(f"  Precision : {metrics.box.mp:.4f}")
        print(f"  Recall    : {metrics.box.mr:.4f}")
        print(f"{'─'*40}")
        print(f"  클래스별 mAP@50")
        for name, ap in zip(CLASS_NAMES, metrics.box.ap50):
            print(f"    {name:10s}: {ap:.4f}")
        print(f"{'─'*40}\n")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ CUDA OOM 발생! → YOLO26{size.upper()} batch={batch}")
            print(f"   BATCH_BY_SIZE[\"{size}\"] 값을 절반으로 줄인 후 재시도하세요\n")
        else:
            print(f"\n❌ 학습 중 에러 발생 (YOLO26{size.upper()}): {e}\n")
        # 에러 발생해도 다음 사이즈로 계속 진행
        return

    torch.cuda.empty_cache()


def main():
    # ── 사전 체크 ──────────────────────────────
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml 파일이 없습니다: {DATA_YAML}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다. GPU 환경을 확인해주세요.")

    # 모델 파일 존재 여부 확인
    check_model_files()

    DEVICE = 0
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 환경 정보 출력 ─────────────────────────
    print("="*60)
    print("SmartView YOLO26 전체 사이즈 학습")
    print("="*60)
    print(f"PyTorch 버전  : {torch.__version__}")
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"학습 사이즈   : {SIZES}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Image size    : {IMG_SIZE}")
    print(f"데이터 경로   : {DATA_YAML}")
    print(f"결과 저장     : {PROJECT_DIR}")
    print("="*60)

    # ── 전체 사이즈 순차 학습 ──────────────────
    for size in SIZES:
        train_model(size, DEVICE)

    # ── 최종 결과 요약 ─────────────────────────
    print("\n" + "="*60)
    print("🎉 전체 학습 완료!")
    print(f"결과 폴더: {PROJECT_DIR}")
    for size in SIZES:
        batch    = BATCH_BY_SIZE[size]
        run_name = f"yolo26{size}_6cls_ep{EPOCHS}_bs{batch}"
        print(f"  YOLO26{size.upper()} → {PROJECT_DIR}\\{run_name}\\weights\\best.pt")
    print("="*60)


if __name__ == "__main__":
    mp.freeze_support()    # Windows 필수
    main()