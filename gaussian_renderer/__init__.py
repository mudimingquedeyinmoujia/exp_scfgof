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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.neural_gaussian_model import NGaussianModel
from utils.sh_utils import eval_sh
from einops import repeat
from utils import mesh_depth_tool


def generate_neural_gaussians(viewpoint_camera, pc: NGaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    if torch.isnan(anchor).any():
        import ipdb;ipdb.set_trace()
        raise Exception('fucking NAN of anchor')

    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    # process inf
    # inf_mask = torch.isinf(ob_view)
    # rows_with_inf = inf_mask.any(dim=1)
    # ob_view[rows_with_inf] = 0

    if torch.isnan(ob_view).any():
        import ipdb;ipdb.set_trace()
        raise Exception('fucking NAN of ob_view')
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
    # print(f'-- neural opacity shape {neural_opacity.shape}') # num xyz, -1,1
    mask = (neural_opacity > 0)
    mask = mask.view(-1)

    # import ipdb; ipdb.set_trace()
    # select opacity
    opacity = neural_opacity[mask] # maybe -1

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
    # print(f'--- xyz shape {xyz.shape}')
    # print(f'--- scaling shape {scaling.shape}')
    # print(f'--- opacity shape {opacity.shape}')
    scaling_final = pc.pass_scaling_with_3D_filter(scaling, mask)
    opacity = pc.pass_opacity_with_3D_filter(opacity, scaling, mask)

    if torch.isnan(xyz).any() or torch.isnan(color).any() or torch.isnan(opacity).any() or torch.isnan(
            scaling).any() or torch.isnan(rot).any():
        print('fucking NAN of generate_neural GS')
        import ipdb;ipdb.set_trace()
    if is_training:
        return xyz, color, opacity, scaling_final, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling_final, rot


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float, scaling_modifier=1.0,
           override_color=None, subpixel_offset=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:  # False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:  # False
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


global_iter = 0


def render_neuralGS(viewpoint_camera, pc: NGaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float,
                    scaling_modifier=1.0,
                    override_color=None, subpixel_offset=None, visible_mask=None, retain_grad=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc,
                                                                                            visible_mask,
                                                                                            is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask,
                                                                      is_training=is_training)
        mask = None
        neural_opacity = None
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    xyz_back=xyz.detach().clone()
    # means3D = xyz
    # means2D = screenspace_points
    # opacity = opacity

    # view2gaussian_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    if torch.isnan(xyz).any() or torch.isnan(color).any() or torch.isnan(opacity).any() or torch.isnan(scaling).any() or torch.isnan(rot).any():
        print('fucking NAN before rasterizer')
        import ipdb;ipdb.set_trace()


    rendered_image, radii = rasterizer(
        means3D=xyz.cuda(),
        means2D=screenspace_points.cuda(),
        shs=None,
        colors_precomp=color.cuda(),
        opacities=opacity.cuda(),
        scales=scaling.cuda(),
        rotations=rot.cuda(),
        cov3D_precomp=None,
        view2gaussian_precomp=None)

    # with open('render_debug12_log.txt', 'a') as file:
    #     file.write(f'xyz: {xyz.shape}, {xyz.dtype} ')
    #     file.write(f'radii: {radii.shape}, {radii.dtype}, {radii.device}\n')

    visb_f = radii > 0  # int32 = xyz.shape (float32)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visb_f,
            "radii": None,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "xyz_back": xyz_back
            }


def integrate(points3D, viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float,
              scaling_modifier=1.0, override_color=None, subpixel_offset=None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    view2gaussian_precomp = None
    # pipe.compute_view2gaussian_python = True
    if pipe.compute_view2gaussian_python:
        view2gaussian_precomp = pc.get_view2gaussian(raster_settings.viewmatrix)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
        points3D=points3D,
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        view2gaussian_precomp=view2gaussian_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def integrate_neuralGS(points3D, viewpoint_camera, pc: NGaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float,
                       scaling_modifier=1.0, override_color=None, subpixel_offset=None, visible_mask=None,
                       retain_grad=False):
    """
    integrate Gaussians to the points, we also render the image for visual comparison.

    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc,
                                                                                            visible_mask,
                                                                                            is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask,
                                                                      is_training=is_training)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, alpha_integrated, color_integrated, radii = rasterizer.integrate(
        points3D=points3D,
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3D_precomp=None,
        view2gaussian_precomp=None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


def prefilter_voxel(viewpoint_camera, pc: NGaussianModel, pipe, bg_color: torch.Tensor, kernel_size: float,
                    scaling_modifier=1.0,
                    override_color=None, subpixel_offset=None, not_filter=True):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    if not_filter:
        return torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=pc.get_anchor.device)
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True,
                                          device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2),
                                      dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D=means3D,
                                           scales=scales[:, :3],
                                           rotations=rotations,
                                           cov3D_precomp=cov3D_precomp)

    return radii_pure > 0
