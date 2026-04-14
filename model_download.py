from huggingface_hub import hf_hub_download
from pathlib import Path

# 본인 캡스톤 프로젝트 폴더에 맞춰 넣을 것
SAVE_DIR = Path(r"C:\Users\SBL-336server2\Desktop\컴공 캡스톤\models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

for model_name in MODELS:
    print(f"⬇️  다운로드 중: {model_name} ...")
    hf_hub_download(
        repo_id   = "Ultralytics/YOLO26",
        filename  = model_name,
        local_dir = str(SAVE_DIR),
    )
    print(f"✅ 완료: {SAVE_DIR / model_name}")

print("\n🎉 모든 모델 다운로드 완료!")