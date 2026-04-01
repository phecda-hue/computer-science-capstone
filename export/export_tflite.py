from ultralytics import YOLO

model = YOLO("models/best.pt")

# INT8 양자화 + TFLite 변환 (한 번에)
model.export(
    format="tflite",
    imgsz=320,          # 스마트글래스 해상도에 맞게 조정
    int8=True,          # 엣지 기기 최적화 필수
    data="train/dataset.yaml",  # 양자화용 캘리브레이션 데이터
)