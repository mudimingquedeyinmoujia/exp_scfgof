# conda install cmake=3.22
# cmake -DCMAKE_CUDA_COMPILER=/home/songgaochao/anaconda3/envs/gof/bin/nvcc .
# ps -o user= -p
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# for param in pc.mlp_cov.parameters():print(param)
# for param in pc.mlp_opacity.parameters():print(param)
# for param in pc.mlp_color.parameters():print(param)
# mask_anchor = torch.isnan(self._anchor_feat).any(dim=1).nonzero(as_tuple=True)[0]
import torch
from einops import repeat
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
from pytorch3d.transforms import quaternion_apply, quaternion_invert

a=torch.zeros(1)

if __name__ == "__main__":
    # 生成示例数据
    print(a.item())