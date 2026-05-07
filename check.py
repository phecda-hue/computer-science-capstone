# check_inference.py
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train.preprocess import preprocess_yolo, postprocess_yolo

img_path = next(Path("data/test/images").rglob("*.jpg"))
blob, orig_shape, scale, pad = preprocess_yolo(img_path)

print(f"이미지: {img_path.name}")
print(f"원본 shape: {orig_shape}")

# ── PT 모델 출력 확인 ──────────────────────────────────────
from ultralytics import YOLO
model = YOLO("runs/detect/yolo26s_6cls_ep200_bs64/weights/best.pt")
results = model.predict(str(img_path), verbose=False)
boxes = results[0].boxes
print(f"\n[PT] 탐지 수: {len(boxes)}")
if len(boxes) > 0:
    print(f"  박스 예시: {boxes[0].xyxy}, conf={float(boxes[0].conf):.3f}, cls={int(boxes[0].cls)}")

# ── ONNX 출력 확인 ────────────────────────────────────────
import onnxruntime as ort
sess = ort.InferenceSession("converted/yolo_s/yolo_sim.onnx",
                            providers=["CPUExecutionProvider"])
raw = sess.run(None, {sess.get_inputs()[0].name: blob})
print(f"\n[ONNX] 출력 shape: {raw[0].shape}")
print(f"  최대 confidence: {raw[0][0, 4:, :].max():.4f}")

dets = postprocess_yolo(raw[0], orig_shape, scale, pad)
print(f"  postprocess 후 탐지 수: {len(dets)}")
if dets:
    print(f"  탐지 예시: {dets[0]}")

# ── GT 레이블 확인 ────────────────────────────────────────
label = Path("data/test/labels") / (img_path.stem + ".txt")
print(f"\n[GT] 레이블 내용:")
print(label.read_text().strip())

# check_inference.py에 아래 추가
import numpy as np
import onnxruntime as ort
from train.preprocess import preprocess_yolo

img_path = next(Path("data/test/images").rglob("*.jpg"))
blob, orig_shape, scale, pad = preprocess_yolo(img_path)

sess = ort.InferenceSession("converted/yolo_s/yolo_sim.onnx",
                            providers=["CPUExecutionProvider"])
raw = sess.run(None, {sess.get_inputs()[0].name: blob})

pred = raw[0][0]   # (300, 6)

# 값이 0이 아닌 첫 5개 행 출력
print("ONNX 출력 raw 값 (처음 10행):")
print(pred[:10])

print("\n전체 최대/최솟값:")
print(f"  col0: {pred[:,0].min():.2f} ~ {pred[:,0].max():.2f}")
print(f"  col1: {pred[:,1].min():.2f} ~ {pred[:,1].max():.2f}")
print(f"  col2: {pred[:,2].min():.2f} ~ {pred[:,2].max():.2f}")
print(f"  col3: {pred[:,3].min():.2f} ~ {pred[:,3].max():.2f}")
print(f"  col4: {pred[:,4].min():.2f} ~ {pred[:,4].max():.2f}")
print(f"  col5: {pred[:,5].min():.2f} ~ {pred[:,5].max():.2f}")