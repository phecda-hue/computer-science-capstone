from ultralytics import YOLO

model = YOLO("yolo11n.pt")   # OIV7 pretrained 가중치
model.train(
    data="train/dataset.yaml",
    epochs=10,
    imgsz=320,
    batch=16,
    project="runs",
    name="baseline",
)