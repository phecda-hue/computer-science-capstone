"""
convert.py
YOLO + DAv2 모델을 ONNX / TFLite로 동시에 변환

사용법:
    python convert.py \
        --yolo   weights/yolo_best.pt \
        --dav2   weights/dav2_small.pt \
        --outdir converted/ \
        [--yolo-size 640] \
        [--dav2-size 518] \
        [--quantize]          # TFLite INT8 양자화 활성화
        [--calib-images data/test]  # INT8 캘리브레이션용 이미지 경로
"""

import argparse
from pyexpat import model
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort


# ══════════════════════════════════════════════════════════════════════════════
# 공통 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    tag = {"INFO": "✅", "WARN": "⚠️ ", "ERR": "❌"}.get(level, "  ")
    print(f"[{tag}] {msg}", flush=True)


def save_path(outdir: Path, name: str, ext: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{name}{ext}"


# ══════════════════════════════════════════════════════════════════════════════
# ONNX 공통 변환
# ══════════════════════════════════════════════════════════════════════════════

def simplify_onnx(onnx_path: Path) -> Path:
    """onnxsim으로 그래프 단순화. 설치되어 있으면 사용하고, 없으면 원본 유지."""
    try:
        from onnxsim import simplify

        model = onnx.load(str(onnx_path))
        simplified, ok = simplify(model)

        if ok:
            sim_path = onnx_path.with_stem(onnx_path.stem + "_sim")
            onnx.save(simplified, str(sim_path))
            log(f"  ONNX 단순화 완료 → {sim_path.name}")
            return sim_path
        else:
            log("  ONNX 단순화 실패. 원본 ONNX를 유지합니다.", "WARN")

    except ImportError:
        log("  onnxsim 미설치. 단순화를 생략합니다. pip install onnxsim", "WARN")
    except Exception as e:
        log(f"  ONNX 단순화 중 오류 발생. 원본 ONNX를 유지합니다: {e}", "WARN")

    return onnx_path


def verify_onnx(onnx_path: Path, dummy_input: np.ndarray):
    """ONNX Runtime으로 추론 가능한지 확인."""
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy_input})
        log(f"  ONNX 검증 OK  입력={input_name}, 출력shape={[o.shape for o in out]}")
    except Exception as e:
        log(f"  ONNX 검증 실패: {e}", "WARN")


# ══════════════════════════════════════════════════════════════════════════════
# TFLite 변환 공통 파이프라인: ONNX → TF SavedModel → TFLite
# ══════════════════════════════════════════════════════════════════════════════

def onnx_to_tflite(
    onnx_path: Path,
    tflite_path: Path,
    input_shape: tuple,
    input_name: str,
    quantize: bool = False,
    calib_images_dir: Path = None,
    input_range: tuple = (0.0, 1.0),
):
    """
    ONNX를 onnx2tf를 통해 TFLite로 변환.

    input_shape:
        보통 NCHW 형식. 예: (1, 3, 518, 518)

    input_name:
        ONNX 입력 노드 이름.
        - YOLO: 보통 "images"
        - DAv2: 여기서는 "image"
    """
    try:
        import onnx2tf
    except ImportError:
        log("  onnx2tf 미설치 — pip install onnx2tf", "WARN")
        return False

    import shutil

    onnx_path = onnx_path.resolve()
    tflite_path = tflite_path.resolve()

    out_dir = tflite_path.parent / (tflite_path.stem + "_onnx2tf")
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n, c, h, w = input_shape
    overwrite_shape = f"{input_name}:{n},{c},{h},{w}"

    log(f"  ONNX → TFLite 변환 시작: {onnx_path.name}")
    log(f"  input_name={input_name}, overwrite_input_shape={overwrite_shape}")

    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(out_dir),
            output_signaturedefs=True,
            non_verbose=True,
            batch_size=n,
            overwrite_input_shape=[overwrite_shape],
        )

    except Exception as e:
        log(f"  onnx2tf 변환 실패: {e}", "ERR")
        return False

    tflite_files = list(out_dir.glob("*.tflite"))

    if not tflite_files:
        # 하위 폴더에 생성되는 경우까지 탐색
        tflite_files = list(out_dir.rglob("*.tflite"))

    if not tflite_files:
        log("  .tflite 파일이 생성되지 않았습니다.", "ERR")
        return False

    # 보통 가장 큰 파일이 실제 모델일 가능성이 높음
    tflite_files = sorted(tflite_files, key=lambda p: p.stat().st_size, reverse=True)

    shutil.copy(tflite_files[0], tflite_path)

    size_mb = tflite_path.stat().st_size / 1e6
    log(f"  TFLite 저장 완료: {tflite_path.name}  ({size_mb:.1f} MB)")

    return True


def verify_tflite(tflite_path: Path, dummy_input: np.ndarray):
    """TFLite 검증. 모델이 NHWC를 기대하면 NCHW 입력을 자동 transpose."""
    try:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()

        inp = interp.get_input_details()[0]
        outp = interp.get_output_details()

        expected = tuple(inp["shape"])
        provided = tuple(dummy_input.shape)

        feed = dummy_input.copy()

        # 예:
        # expected=(1, 518, 518, 3)
        # provided=(1, 3, 518, 518)
        # 이 경우 NCHW → NHWC 변환
        if len(expected) == 4 and len(provided) == 4:
            if expected[1] == provided[2] and expected[2] == provided[3] and expected[3] == provided[1]:
                feed = feed.transpose(0, 2, 3, 1)
                log(f"  입력 NHWC 변환 적용: {provided} → {feed.shape}")

        if inp["dtype"] == np.int8:
            scale, zp = inp["quantization"]
            feed = (feed / scale + zp).astype(np.int8)
        elif inp["dtype"] == np.uint8:
            scale, zp = inp["quantization"]
            feed = (feed / scale + zp).astype(np.uint8)
        else:
            feed = feed.astype(inp["dtype"])

        interp.set_tensor(inp["index"], feed)
        interp.invoke()

        shapes = [interp.get_tensor(o["index"]).shape for o in outp]

        input_format = "NHWC" if feed.shape != dummy_input.shape else "NCHW"
        log(f"  TFLite 검증 OK  입력format={input_format}, 출력shape={shapes}")

    except Exception as e:
        log(f"  TFLite 검증 실패: {e}", "WARN")


# ══════════════════════════════════════════════════════════════════════════════
# YOLO 변환
# ══════════════════════════════════════════════════════════════════════════════

def convert_yolo(
    pt_path: Path,
    outdir: Path,
    size: int = 640,
    quantize: bool = False,
    calib_dir: Path = None,
) -> dict:

    from ultralytics import YOLO
    import shutil

    outdir.mkdir(parents=True, exist_ok=True)

    results = {"pt": str(pt_path)}

    model = YOLO(str(pt_path))

    # ---------------- ONNX ----------------
    onnx_file = model.export(
        format="onnx",
        imgsz=size,
        simplify=True,
        opset=17,
        nms=False,
    )

    onnx_file = Path(onnx_file)

    final_onnx = outdir / f"{pt_path.stem}.onnx"
    shutil.copy(onnx_file, final_onnx)

    results["onnx"] = str(final_onnx)

    log(f"ONNX 저장 완료: {final_onnx}")

    # ---------------- TFLite ----------------
    tflite_exported = model.export(
        format="tflite",
        imgsz=size,
        int8=quantize,
        nms=False,
    )

    tflite_exported = Path(tflite_exported)

    if tflite_exported.is_dir():
        tflite_candidates = list(tflite_exported.rglob("*.tflite"))
        if not tflite_candidates:
            raise FileNotFoundError(f"TFLite 파일을 찾지 못했습니다: {tflite_exported}")
        tflite_file = tflite_candidates[0]
    else:
        tflite_file = tflite_exported

    final_tflite = outdir / f"{pt_path.stem}_{'int8' if quantize else 'fp32'}.tflite"
    shutil.copy(tflite_file, final_tflite)

    results["tflite"] = str(final_tflite)
    log(f"TFLite 저장 완료: {final_tflite}")

    results["tflite"] = str(final_tflite)

    log(f"TFLite 저장 완료: {final_tflite}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# DAv2 변환
# ══════════════════════════════════════════════════════════════════════════════

class DAv2Wrapper(torch.nn.Module):
    """
    DAv2 모델을 ONNX export 가능하도록 감싸는 wrapper.
    모델 출력이 dict이면 depth 관련 key를 찾아 반환하고,
    아니면 그대로 반환한다.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        if isinstance(out, dict):
            # Depth Anything V2 계열에서 자주 쓰일 수 있는 key들
            for key in ["depth", "metric_depth", "out", "pred"]:
                if key in out:
                    return out[key]

            # key를 못 찾으면 첫 번째 Tensor 값을 반환
            for value in out.values():
                if torch.is_tensor(value):
                    return value

            raise RuntimeError(f"DAv2 출력 dict에서 Tensor를 찾지 못했습니다. keys={list(out.keys())}")

        return out


def load_dav2_model(pt_path: Path):
    """
    Depth Anything V2 metric depth 모델 로드.

    예상 폴더 구조:
        현재 실행 위치/
        ├── convert.py
        ├── convert_all.py
        └── Depth_Anything_V2/
            └── metric_depth/
                └── depth_anything_v2/
                    └── dpt.py
    """
    project_root = Path.cwd().resolve()
    dav2_metric = project_root / "Depth_Anything_V2" / "metric_depth"

    log(f"  project_root : {project_root}")
    log(f"  dav2_metric  : {dav2_metric}")
    log(f"  경로 존재 여부 : {dav2_metric.exists()}")

    if not dav2_metric.exists():
        raise FileNotFoundError(
            f"DAv2 metric_depth 폴더를 찾지 못했습니다: {dav2_metric}\n"
            f"현재 작업 디렉터리에서 Depth_Anything_V2/metric_depth 구조가 맞는지 확인하세요."
        )

    if str(dav2_metric) not in sys.path:
        sys.path.insert(0, str(dav2_metric))

    # 중요:
    # sys.path에 Depth_Anything_V2/metric_depth를 넣었으므로
    # import는 depth_anything_v2.dpt 기준으로 해야 함.
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
        max_depth=80,
    )

    state = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]

    # DataParallel로 저장된 경우 module. prefix 제거
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        state = new_state

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def convert_dav2(
    pt_path: Path,
    outdir: Path,
    size: int = 640,
    quantize: bool = False,
    calib_dir: Path = None,
) -> dict:
    log(f"\n{'=' * 60}")
    log(f"DAv2 변환 시작: {pt_path.name}")
    log(f"{'=' * 60}")

    results = {"pt": str(pt_path)}

    try:
        model = load_dav2_model(pt_path)
    except Exception as e:
        log(f"  DAv2 모델 로드 실패: {e}", "ERR")
        return results

    wrapper = DAv2Wrapper(model)
    wrapper.eval()

    dummy = torch.zeros(1, 3, size, size)

    outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = outdir / "dav2_small.onnx"

    try:
        with torch.no_grad():
            try:
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(onnx_path),
                    opset_version=16,
                    input_names=["image"],
                    output_names=["depth"],
                    do_constant_folding=True,
                    dynamic_axes=None,
                    external_data=False,
                )
            except TypeError:
                # PyTorch 버전에 따라 external_data 인자를 지원하지 않을 수 있음
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(onnx_path),
                    opset_version=16,
                    input_names=["image"],
                    output_names=["depth"],
                    do_constant_folding=True,
                    dynamic_axes=None,
                )

        log(f"  DAv2 ONNX 저장: {onnx_path}")
        results["onnx"] = str(onnx_path)

    except Exception as e:
        log(f"  DAv2 ONNX 변환 실패: {e}", "ERR")
        return results

    # DAv2는 Resize/Interpolate 때문에 onnxsim에서 오류가 자주 나므로 생략
    dummy_np = np.random.rand(1, 3, size, size).astype(np.float32)
    verify_onnx(onnx_path, dummy_np)

    # TFLite 변환 시도
    suffix = "_int8" if quantize else "_fp32"
    tflite_path = outdir / f"dav2_small{suffix}.tflite"
    '''
    ok = dav2_onnx_to_tflite(
        onnx_path=onnx_path,
        tflite_path=tflite_path,
        size=size,
        quantize=quantize,
    )
    
    if ok:
        results["tflite"] = str(tflite_path)
    '''
    return results

def dav2_onnx_to_tflite(
    onnx_path: Path,
    tflite_path: Path,
    size: int = 640,
    quantize: bool = False,
) -> bool:
    """
    DAv2 ONNX → TensorFlow SavedModel → TFLite 변환 시도.
    .onnx.data 파일이 있는 경우에도 onnx_path와 같은 폴더에 있으면 onnx2tf가 함께 읽을 수 있음.
    """
    try:
        import onnx2tf
        import tensorflow as tf
        import shutil
    except ImportError as e:
        log(f"  DAv2 TFLite 변환에 필요한 패키지가 없습니다: {e}", "ERR")
        return False

    onnx_path = onnx_path.resolve()
    tflite_path = tflite_path.resolve()

    out_dir = tflite_path.parent / "dav2_saved_model"

    # 이전 실패 결과가 파일/폴더로 남아 있으면 삭제
    if out_dir.exists():
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        else:
            out_dir.unlink()

    log("  DAv2 ONNX → TensorFlow SavedModel 변환 시작")

    try:
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(out_dir),
            output_signaturedefs=True,
            non_verbose=False,
            batch_size=1,
            overwrite_input_shape=[f"image:1,3,{size},{size}"],
        )
    except Exception as e:
        log(f"  DAv2 ONNX → TF 변환 실패: {e}", "ERR")
        return False

    tflite_candidates = list(out_dir.rglob("*.tflite"))

    if tflite_candidates:
        tflite_candidates = sorted(
            tflite_candidates,
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        shutil.copy(tflite_candidates[0], tflite_path)
        log(f"  DAv2 TFLite 저장 완료: {tflite_path}")
        return True

    log("  onnx2tf에서 TFLite가 바로 생성되지 않아 TensorFlow Lite Converter로 재시도")

    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(str(out_dir))

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        log(f"  DAv2 TFLite 저장 완료: {tflite_path}")
        return True

    except Exception as e:
        log(f"  TensorFlow SavedModel → TFLite 변환 실패: {e}", "ERR")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 변환 결과 요약
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(yolo_results: dict, dav2_results: dict):
    print("\n" + "═" * 60)
    print("  변환 결과 요약")
    print("═" * 60)

    for model_name, res in [("YOLO", yolo_results), ("DAv2", dav2_results)]:
        print(f"\n  [{model_name}]")

        for fmt, path in res.items():
            p = Path(path)
            size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "N/A"
            print(f"    {fmt:8s}: {p.name}  ({size})")

    print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="YOLO + DAv2 → ONNX + TFLite 변환")

    p.add_argument("--yolo", type=Path, required=True, help="YOLO .pt 경로")
    p.add_argument("--dav2", type=Path, required=True, help="DAv2 .pt 또는 .pth 경로")
    p.add_argument("--outdir", type=Path, default=Path("converted"), help="출력 폴더")

    p.add_argument("--yolo-size", type=int, default=640, help="YOLO 입력 해상도")
    p.add_argument("--dav2-size", type=int, default=518, help="DAv2 입력 해상도")

    p.add_argument("--quantize", action="store_true", help="TFLite INT8 양자화")
    p.add_argument("--calib-images", type=Path, default=None, help="INT8 캘리브레이션 이미지 폴더")

    p.add_argument("--skip-yolo", action="store_true")
    p.add_argument("--skip-dav2", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    yolo_results = {}
    dav2_results = {}

    if not args.skip_yolo:
        yolo_results = convert_yolo(
            pt_path=args.yolo,
            outdir=args.outdir,
            size=args.yolo_size,
            quantize=args.quantize,
            calib_dir=args.calib_images,
        )

    if not args.skip_dav2:
        dav2_results = convert_dav2(
            pt_path=args.dav2,
            outdir=args.outdir,
            size=args.dav2_size,
            quantize=args.quantize,
            calib_dir=args.calib_images,
        )

    print_summary(yolo_results, dav2_results)


if __name__ == "__main__":
    main()