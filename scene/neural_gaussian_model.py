#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from einops import repeat
import torch.nn.functional as F
from scipy.spatial import KDTree


class NGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self,
                 feat_dim: int = 32,
                 n_offsets: int = 5,
                 voxel_size: float = 0.01,
                 update_depth: int = 3,
                 update_init_factor: int = 100,
                 update_hierachy_factor: int = 4,
                 use_feat_bank: bool = False,
                 use_kernel_mlp: bool = False,
                 appearance_dim: int = 32,
                 ratio: int = 1,
                 add_opacity_dist: bool = False,
                 add_cov_dist: bool = False,
                 add_color_dist: bool = False,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank
        self.use_kernel_mlp = use_kernel_mlp

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.anchor_tree = None

        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        # self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        # self.mlp_opacity = nn.Sequential(
        #     nn.Linear(feat_dim + 3 + self.opacity_dist_dim, feat_dim),
        #     nn.ReLU(True),
        #     nn.Linear(feat_dim, n_offsets),
        #     nn.Tanh()
        # ).cuda()

        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        if self.use_kernel_mlp:
            self.mlp_kernel_sdf = nn.Sequential(
                nn.Linear(feat_dim + 6, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 1),
            ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim + 3 + self.color_dist_dim + self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).cuda()

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()

        if self.use_kernel_mlp:
            self.mlp_kernel_sdf().eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_kernel_mlp:
            self.mlp_kernel_sdf().train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._anchor,
         self._offset,
         self._local,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    def generate_neural_gaussians_xyz(pc, viewpoint_camera, visible_mask=None, is_training=False):
        ## view frustum filtering for acceleration
        if visible_mask is None:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

        feat = pc._anchor_feat[visible_mask]
        anchor = pc.get_anchor[visible_mask]
        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]
        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if pc.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)

            bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                   feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                   feat[:, ::1, :1] * bank_weight[:, :, 2:]
            feat = feat.squeeze(dim=-1)  # [n, c]

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]
        if pc.appearance_dim > 0:
            camera_indicies = torch.ones_like(cat_local_view[:, 0], dtype=torch.long,
                                              device=ob_dist.device) * viewpoint_camera.uid
            # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
            appearance = pc.get_appearance(camera_indicies)

        # get offset's opacity
        # if pc.add_opacity_dist:
        #     neural_opacity = pc.get_opacity_mlp(cat_local_view)  # [N, k]
        # else:
        #     neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

        neural_opacity = pc.get_opacity_mlp(feat)

        # opacity mask generation

        neural_opacity = neural_opacity.reshape([-1, 1])
        # print(f'neural opacity shape {neural_opacity.shape}')
        mask = (neural_opacity > -2)
        mask = mask.view(-1)
        # import ipdb; ipdb.set_trace()
        # select opacity

        opacity = neural_opacity[mask]

        # get offset's color
        if pc.appearance_dim > 0:
            if pc.add_color_dist:
                color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            else:
                color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
        else:
            if pc.add_color_dist:
                color = pc.get_color_mlp(cat_local_view)
            else:
                color = pc.get_color_mlp(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [mask]

        # get offset's cov
        if pc.add_cov_dist:
            scale_rot = pc.get_cov_mlp(cat_local_view)
        else:
            scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])  # [mask]

        # offsets
        offsets = grid_offsets.view([-1, 3])  # [mask]

        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * (1+torch.sigmoid(repeat_dist))
        rot = pc.rotation_activation(scale_rot[:, 3:7])

        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_anchor + offsets
        if xyz.numel() == 0:
            import ipdb;
            ipdb.set_trace()
        # print(f'xyz shape {xyz.shape}')
        # print(f'scaling shape {scaling.shape}')
        # print(f'opacity shape {opacity.shape}')
        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, color, opacity, scaling, rot

    def sample_sdf_points(self, xyz, mask):
        """
            高效地根据给定的参数采样点，并返回采样点在原始xyz中的索引。

            参数:
            xyz (torch.Tensor): 形状为(N, 3)的张量，表示空间中的点集。
            scaling (float): 用于控制正态分布的缩放。
            n_points (int): 想要获取的采样点总数。
            mask (torch.Tensor, optional): 一维的bool张量，用于遮盖一些xyz点。默认为None。

            返回:
            (torch.Tensor, torch.Tensor): 一个元组，包含采样得到的点集和这些点在原始xyz中的索引。
            """
        n_points = self.sample_sdf_num
        scaling = self.sample_sdf_scale
        if mask is not None:
            # 应用mask过滤点和索引
            valid_indices = torch.arange(xyz.shape[0])[mask]
            filtered_xyz = xyz[mask]
        else:
            valid_indices = torch.arange(xyz.shape[0])
            filtered_xyz = xyz

        # 假设所有点的权重相等（简化示例），创建一个均匀的概率分布
        probabilities = torch.ones(filtered_xyz.shape[0]) / filtered_xyz.shape[0]
        cum_probs = probabilities.cumsum(dim=-1)

        # 基于累积概率分布进行加权采样，以获得随机索引
        random_indices = torch.multinomial(cum_probs, num_samples=n_points, replacement=True)
        final_indices = valid_indices[random_indices]

        # 为选中的点添加正态分布的随机偏移
        noise = torch.randn(final_indices.shape[0], 3) * scaling
        sampled_xyz = xyz[final_indices] + noise.cuda()
        self.tmp_sampled_sdf = sampled_xyz
        self.tmp_sampled_sdf_ind = final_indices
        return sampled_xyz

    def compute_sdf_vectorized(self, query_points):
        # 使用KD树找到最近的k个中心点及其特征
        k = self.kernel_num
        features = self._anchor_feat
        mlp = self.mlp_kernel_sdf
        tree = self.anchor_tree
        dists, idx = tree.query(query_points.cpu().numpy(), k=k)

        # 获取最近的中心点和特征
        nearest_centers = self._anchor[idx.reshape(-1)].view(*idx.shape, -1)
        nearest_features = features[idx.reshape(-1)].view(*idx.shape, -1, features.size(1))

        # 计算高斯权重
        dists_tensor = torch.tensor(dists, dtype=torch.float32)
        weights = torch.exp(-dists_tensor ** 2)
        weights /= weights.sum(dim=1, keepdim=True)

        # 准备MLP的输入
        query_points_expanded = query_points.cuda().unsqueeze(1).expand(-1, k, -1).contiguous().view(-1, 3)
        inputs = torch.cat([query_points_expanded, nearest_centers.view(-1, 3), nearest_features.view(-1, 32)], dim=1)

        # 计算SDF值
        sdf_values = mlp(inputs).view(-1, k)
        # 计算加权SDF值
        # import ipdb;ipdb.set_trace()
        weighted_sdf = torch.sum(weights.cuda() * sdf_values, dim=1)

        return weighted_sdf

    def set_anchor_tree(self):
        if not self.use_kernel_mlp:
            return
        anchor_numpy = self._anchor.cpu().detach().clone().numpy()
        self.anchor_tree = KDTree(anchor_numpy)
        self._anchor.cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0 * self.scaling_activation(self._scaling)

    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_kernel_sdf_mlp(self):
        return self.mlp_kernel_sdf

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def pass_scaling_with_3D_filter(self, scaling_act, mask):
        scales = scaling_act
        scales = torch.square(scales) + torch.square(self.filter_3D[mask])
        scales = torch.sqrt(scales)
        return scales

    def pass_opacity_with_3D_filter(self, opacity_act, scaling_act, mask):
        opacity = opacity_act
        # apply 3D filter
        scales = scaling_act

        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3D[mask])
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacity * coef[..., None]

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        print("Computing 3D filter")
        # TODO consider focal length and image width
        with torch.no_grad():
            xyz, _, _, _, _ = self.generate_neural_gaussians_xyz(cameras[0])
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
            # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]

            xyz_to_cam = torch.norm(xyz_cam, dim=1)

            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2  # TODO remove hard coded value

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0

            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))

            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(
                torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15),
                torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))

            valid = torch.logical_and(valid_depth, in_screen)

            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
        try:
            distance[~valid_points] = distance[valid_points].max()
        except:
            import ipdb;
            ipdb.set_trace()

        # TODO remove hard coded value
        # TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]
        print(f'filter 3D  shape {self.filter_3D.shape}')

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        self.set_anchor_tree()

    def training_setup(self, training_args):
        self.kernel_num = training_args.kernel_num
        self.sample_sdf_num = training_args.sample_sdf_num
        self.sample_sdf_scale = training_args.sample_sdf_scale
        self.tmp_sampled_sdf = None
        self.tmp_sampled_sdf_ind = None

        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0] * self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init,
                 "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init,
                 "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init,
                 "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale,
                 "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale,
                 "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},  # 指的是anchor的
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init,
                 "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},

            ]

        if self.use_kernel_mlp:
            l.append({'params': self.mlp_kernel_sdf.parameters(), 'lr': training_args.mlp_kernel_sdf_lr_init,
                      "name": "mlp_kernel_sdf"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=training_args.offset_lr_delay_mult,
                                                       max_steps=training_args.offset_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                            lr_final=training_args.mlp_opacity_lr_final,
                                                            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                        lr_final=training_args.mlp_cov_lr_final,
                                                        lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                        max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                          lr_final=training_args.mlp_color_lr_final,
                                                          lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                          max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_kernel_mlp:
            self.mlp_kernel_sdf_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_kernel_sdf_lr_init,
                                                                   lr_final=training_args.mlp_kernel_sdf_lr_final,
                                                                   lr_delay_mult=training_args.mlp_kernel_sdf_lr_delay_mult,
                                                                   max_steps=training_args.mlp_kernel_sdf_lr_max_steps)

        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                                    lr_final=training_args.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                               lr_final=training_args.appearance_lr_final,
                                                               lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                               max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_kernel_mlp and param_group["name"] == "mlp_kernel_sdf":
                lr = self.mlp_kernel_sdf_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1] * self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                           np.asarray(plydata.elements[0]["y"]),
                           np.asarray(plydata.elements[0]["z"])), axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        self._anchor_feat = nn.Parameter(
            torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(
            torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # statis grad information to guide liftting.
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask,
                        anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0  # num xyz

        temp_opacity = temp_opacity.view([-1, self.n_offsets])  # [anchor, offsets] 不透明度[-1,1]--> [0,1]
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)  # 每一个anchor的不透明度累积 [anchor,1]

        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1  # anchor 访问统计

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(
            -1)  # [anchor,offsets]
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask  # 不透明度>0的mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter  # visibility filter radii > 0
        # combined_mask: 选择的anchor里面，不透明度>0，且radii>0的点
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm  # n_xyz 梯度累加 在视图里面，radii>0的梯度
        self.offset_denom[combined_mask] += 1  # n_xyz 访问统计 视图里面，radii>0

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'mlp' in group['name'] or \
                    'conv' in group['name'] or \
                    'feat_base' in group['name'] or \
                    'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)
        ori_anchor_num = self._anchor.shape[0]
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        prune_anchor_num = self._anchor.shape[0]
        print(f'prune anchor from {ori_anchor_num} to {prune_anchor_num}')

    def anchor_growing(self, grads, threshold, offset_mask):
        ##
        init_length = self.get_anchor.shape[0] * self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold * ((self.update_hierachy_factor // 2) ** i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            # random pick
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.get_anchor.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')],
                                           dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)
            if torch.isnan(all_xyz).any():
                import ipdb;
                ipdb.set_trace()
                raise Exception('fucking NAN of all_xyz')
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.get_anchor / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True,
                                                                        dim=0)
            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i * chunk_size:(
                                                                                                                            i + 1) * chunk_size,
                                                                                         :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size  # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[
                    candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][
                    remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_anchor_demon = torch.cat(
                    [self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]

    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N] 需要剔除的

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)
        self.set_anchor_tree()
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def check_nan(self):

        if not torch.isnan(self.get_anchor).any() and \
                not torch.isnan(self._scaling).any() and not torch.isnan(self._anchor_feat).any():
            # 如果没有NaN元素，直接返回原张量
            return None

        print('find NAN in self.get_anchor/self._scaling/self._anchor_feat')
        # 使用torch.isnan检查NaN值，然后使用~操作符取反
        # 使用all(dim=1)来保留所有元素都不是NaN的行
        mask_anchor = torch.isnan(self.get_anchor).any(dim=1)
        mask_anchor_rows = mask_anchor.nonzero(as_tuple=True)[0]

        mask_scaling = torch.isnan(self._scaling).any(dim=1)
        mask_scaling_rows = mask_scaling.nonzero(as_tuple=True)[0]

        mask_feat = torch.isnan(self._anchor_feat).any(dim=1)
        mask_feat_rows = mask_feat.nonzero(as_tuple=True)[0]

        print(f'anchor: {mask_anchor_rows}, scaling: {mask_scaling_rows}, feat: {mask_feat_rows}')

        mask1 = torch.logical_and(mask_anchor, mask_scaling)
        mask_all = torch.logical_and(mask1, mask_feat)

        return mask_all

    def prune_nan_anchor(self):
        # # prune anchors
        # prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
        # anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
        # prune_mask = torch.logical_and(prune_mask, anchors_mask)  # [N] 需要剔除的
        prune_mask = self.check_nan()
        if prune_mask is None:
            return False
        print('pruning NAN ing ...')
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0] > 0:
            self.prune_anchor(prune_mask)

        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

        return True

    def save_mlp_checkpoints(self, path, mode='split'):  # split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity,
                                          (torch.rand(1, self.feat_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim + 3 + self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (
                torch.rand(1, self.feat_dim + 3 + self.color_dist_dim + self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()
            if self.use_kernel_mlp:
                self.mlp_kernel_sdf.eval()
                kernel_sdf_mlp = torch.jit.trace(self.mlp_kernel_sdf, (
                    torch.rand(1, self.feat_dim + 6).cuda()))
                kernel_sdf_mlp.save(os.path.join(path, 'kernel_sdf_mlp.pt'))
                self.mlp_kernel_sdf.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3 + 1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            save_dict = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict(),
            }
            if self.use_feat_bank:
                save_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
                save_dict['appearance'] = self.embedding_appearance.state_dict()

            elif self.appearance_dim > 0:
                save_dict['appearance'] = self.embedding_appearance.state_dict()
            else:
                save_dict['kernel_sdf_mlp'] = self.mlp_kernel_sdf.state_dict()

            torch.save({save_dict}, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError

    def load_mlp_checkpoints(self, path, mode='split'):  # split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_kernel_mlp:
                self.mlp_kernel_sdf = torch.jit.load(os.path.join(path, 'kernel_sdf_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_kernel_mlp:
                self.mlp_kernel_sdf.load_state_dict(checkpoint['kernel_sdf_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
