import sys
import cv2
import torch

sys.path.insert(0, r"C:\Users\SBL-336server2\Desktop\컴공 캡스톤\Depth_Anything_V2\metric_depth")
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"사용 디바이스: {DEVICE}")

dataset   = 'vkitti'
max_depth = 80

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

model = DepthAnythingV2(**{**model_configs['vits'], 'max_depth': max_depth})

model.load_state_dict(
    torch.load(
        f'checkpoints/depth_anything_v2_metric_{dataset}_vits.pth',
        map_location=DEVICE,      # ← 'cpu' 대신 DEVICE로 직접 로드
        weights_only=False,       # ← FutureWarning 명시적으로 처리
    )
)

model = model.to(DEVICE).eval() # ← load 이후에 to(DEVICE) 재확인

img   = cv2.imread('your_image.jpg')
depth = model.infer_image(img)
print(f"depth shape: {depth.shape}, min: {depth.min():.2f}m, max: {depth.max():.2f}m")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── 깊이맵 시각화 함수 ──────────────────────────────────────────
def visualize_depth(img_bgr, depth, save_path=None):
    """
    img_bgr : cv2.imread로 읽은 원본 이미지
    depth   : model.infer_image() 출력 (HxW, 미터 단위)
    """
    # 원본 이미지 RGB 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 깊이맵 컬러맵 적용 (inferno: 가까울수록 밝음)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth_colored = (cm.inferno(depth_norm)[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 원본 이미지
    axes[0].imshow(img_rgb)
    axes[0].set_title('원본 이미지', fontsize=13)
    axes[0].axis('off')

    # 컬러 깊이맵
    axes[1].imshow(depth_colored)
    axes[1].set_title('Depth Map (컬러)', fontsize=13)
    axes[1].axis('off')

    # 수치 히트맵 (실제 미터값 표시)
    im = axes[2].imshow(depth, cmap='plasma')
    axes[2].set_title('Depth Map (미터)', fontsize=13)
    axes[2].axis('off')
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('거리 (m)', fontsize=11)

    plt.suptitle(
        f"min: {depth.min():.2f}m  /  max: {depth.max():.2f}m  /  mean: {depth.mean():.2f}m",
        fontsize=11, y=1.02
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장 완료: {save_path}")

    plt.show()


# ── 실행 ────────────────────────────────────────────────────────
img   = cv2.imread('your_image.jpg')
depth = model.infer_image(img)

visualize_depth(img, depth, save_path='depth_result.png')