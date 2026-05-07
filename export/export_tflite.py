from ultralytics import YOLO
from numpy.core import umath

models = [
    "models/best_n.pt",
    "models/best_s.pt",
    "models/best_m.pt",
    "models/best_l.pt",
    "models/best_x.pt"
]

for model_name in models:

    model = YOLO(model_name)

    # INT8 양자화 + TFLite 변환 (한 번에)
    model.export(
        format="tflite",
        imgsz=640,          # 스마트글래스 해상도에 맞게 조정
        int8=False,          # 엣지 기기 최적화 필수
        data="train/dataset.yaml",  # 양자화용 캘리브레이션 데이터
    )