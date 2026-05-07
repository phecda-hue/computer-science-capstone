# check_model.py에 추가
from ultralytics import YOLO

model = YOLO("runs/detect/yolo26s_6cls_ep200_bs64/weights/best.pt")

# 마지막 레이어 (detection head) 확인
last_layer = model.model.model[-1]
print(f"Detection head 타입: {type(last_layer).__name__}")