# dav2_onnx_to_tflite_only.py

from pathlib import Path
import shutil
import tensorflow as tf
import onnx2tf


def convert_dav2_onnx_to_tflite(
    onnx_path: str,
    output_dir: str = "converted/dav2_tflite",
    tflite_name: str = "dav2_small_fp32.tflite",
    size: int = 518,
):
    onnx_path = Path(onnx_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = output_dir / "saved_model"
    tflite_path = output_dir / tflite_name

    if saved_model_dir.exists():
        if saved_model_dir.is_dir():
            shutil.rmtree(saved_model_dir)
        else:
            saved_model_dir.unlink()

    print(f"ONNX 입력: {onnx_path}")
    print("ONNX → TensorFlow SavedModel 변환 시작")

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        output_signaturedefs=True,
        non_verbose=False,
        batch_size=1,
        overwrite_input_shape=[f"image:1,3,{size},{size}"],
    )

    # onnx2tf가 tflite를 이미 만든 경우
    candidates = list(saved_model_dir.rglob("*.tflite"))
    if candidates:
        candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
        shutil.copy(candidates[0], tflite_path)
        print(f"TFLite 저장 완료: {tflite_path}")
        return

    print("SavedModel → TFLite 변환 시작")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite 저장 완료: {tflite_path}")


if __name__ == "__main__":
    convert_dav2_onnx_to_tflite(
        onnx_path="converted/dav2/dav2_small.onnx",
        output_dir="converted/dav2_tflite",
        tflite_name="dav2_small_fp32.tflite",
        size=518,
    )