import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_feature_map_channels(feature_map: torch.Tensor,
                                   save_dir: str,
                                   cmap: str = "jet",
                                   show_colorbar: bool = False):
    """
    将多通道特征图的每一个通道单独保存为图片

    Args:
        feature_map (torch.Tensor): 输入特征图 [1, C, H, W]
        save_dir (str): 保存目录
        cmap (str): 颜色映射，默认 'jet'
        show_colorbar (bool): 是否显示颜色条
    """
    assert feature_map.dim() == 4 and feature_map.size(0) == 1

    fmap = feature_map.squeeze(0).detach().cpu().numpy()  # [C, H, W]
    C = fmap.shape[0]

    os.makedirs(save_dir, exist_ok=True)

    for i in range(C):
        channel = fmap[i]  # 单通道 [H, W]
        # 归一化到 [0,1]
        rng = channel.ptp()
        if rng == 0:
            channel_norm = np.zeros_like(channel, dtype=np.float32)
        else:
            channel_norm = (channel - channel.min()) / (rng + 1e-8)

        save_path = os.path.join(save_dir, f"ch_{i:03d}.png")

        if not show_colorbar:
            # 直接保存，无 colorbar
            plt.imsave(save_path, channel_norm, cmap=cmap)
        else:
            # 保存带 colorbar 的版本
            plt.close('all')
            fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
            im = ax.imshow(channel, cmap=cmap)
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        print(f"✅ 保存通道 {i} → {save_path}")
