from ultralytics import YOLO

model = YOLO("runs/detect/yolo26s_6cls_ep200_bs64/weights/best.pt")
print(f"클래스 수: {len(model.names)}")
print(f"클래스 목록: {model.names}")