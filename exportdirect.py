# export_no_nms.py
import torch
from ultralytics import YOLO
from pathlib import Path

sizes = {
    'n': 'runs/detect/yolo26n_6cls_ep200_bs64/weights/best.pt',
    's': 'runs/detect/yolo26s_6cls_ep200_bs64/weights/best.pt',
    'm': 'runs/detect/yolo26m_6cls_ep200_bs32/weights/best.pt',
    'l': 'runs/detect/yolo26l_6cls_ep200_bs16/weights/best.pt',
    'x': 'runs/detect/yolo26x_6cls_ep200_bs8/weights/best.pt',
}

for size, pt_path in sizes.items():
    print(f"\nYOLO-{size.upper()} NMS 없이 변환 중...")
    out_path = Path(f"converted/yolo_{size}/yolo_raw.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(pt_path)
    torch_model = model.model
    torch_model.eval()

    # ── NMS 제거: export 플래그 활성화 ──────────────────────────
    for m in torch_model.modules():
        if hasattr(m, 'export'):
            m.export = True

    dummy = torch.zeros(1, 3, 640, 640)

    # export 후 출력 shape 먼저 확인
    with torch.no_grad():
        out = torch_model(dummy)
        if isinstance(out, (list, tuple)):
            print(f"  출력 shape들: {[x.shape for x in out]}")
        else:
            print(f"  출력 shape: {out.shape}")

    torch.onnx.export(
        torch_model,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"images": {0: "batch"}},
        do_constant_folding=True,
    )
    print(f"  저장: {out_path}")

    # 저장된 ONNX 출력 shape 확인
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    raw = sess.run(None, {"images": dummy.numpy()})
    print(f"  ONNX 출력 shape: {raw[0].shape}")
    print(f"  col4 범위: {raw[0][0, 4, :].min():.4f} ~ {raw[0][0, 4, :].max():.4f}")