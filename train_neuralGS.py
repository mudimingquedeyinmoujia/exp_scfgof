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

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import torchvision
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render_neuralGS, prefilter_voxel
import sys
from scene import NeuralScene, NGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils import mesh_depth_tool
from torch.nn.utils import clip_grad_norm_

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import time
from utils.vis_utils import apply_depth_colormap, save_points, colormap
from utils.depth_utils import depths_to_points, depth_to_normal


@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()

    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1

    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image


def get_edge_aware_distortion_map(gt_image, distortion_map):
    grad_img_left = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, :-2]), 0)
    grad_img_right = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 1:-1, 2:]), 0)
    grad_img_top = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, :-2, 1:-1]), 0)
    grad_img_bottom = torch.mean(torch.abs(gt_image[:, 1:-1, 1:-1] - gt_image[:, 2:, 1:-1]), 0)
    max_grad = torch.max(torch.stack([grad_img_left, grad_img_right, grad_img_top, grad_img_bottom], dim=-1), dim=-1)[0]
    # pad
    max_grad = torch.exp(-max_grad)
    max_grad = torch.nn.functional.pad(max_grad, (1, 1, 1, 1), mode="constant", value=0)
    return distortion_map * max_grad


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
             ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    print(dataset.images)
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians)
    # gaussians.training_setup(opt)
    gaussians = NGaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
                               dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.use_kernel_mlp,
                               dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist,
                               dataset.add_color_dist)
    scene = NeuralScene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    for idx, camera in enumerate(scene.getTrainCameras() + scene.getTestCameras()):
        camera.idx = idx

    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    viewpoint_stack_p3d = []
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            assert len(viewpoint_stack_p3d) == 0
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_p3ds = mesh_depth_tool.convert_camera_from_gs_to_pytorch3d(viewpoint_stack)
            for cam in viewpoint_p3ds:
                viewpoint_stack_p3d.append(cam)

        rand_index = randint(0, len(viewpoint_stack) - 1)

        viewpoint_cam = viewpoint_stack.pop(rand_index)
        viewpoint_cam_p3d = viewpoint_stack_p3d.pop(rand_index)

        # Pick a random high resolution camera
        # if random.random() < 0.3 and dataset.sample_more_highres:
        #     viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index) - 1)]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background,
                                             kernel_size=dataset.kernel_size)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        # print(f'iteration {iteration}')
        render_pkg = render_neuralGS(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size,
                                     visible_mask=voxel_visible_mask, retain_grad=retain_grad)

        rendering, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity, xyz_back = \
            render_pkg[
                "render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg[
                "selection_mask"], \
                render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg['xyz_back']

        # rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        image = rendering[:3, :, :]

        # rgb Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        # use L1 loss for the transformed image if using decoupled appearance
        # if dataset.use_decoupled_appearance:
        #     Ll1 = L1_loss_appearance(image, gt_image, gaussians, viewpoint_cam.idx)
        scaling_reg = scaling.prod(dim=1).mean()
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(image, gt_image)) + 0.01 * scaling_reg

        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
        distortion_loss = distortion_map.mean()

        # depth normal consistency
        depth = rendering[6, :, :]

        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)

        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])

        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        depth_normal_loss = normal_error.mean()

        lambda_distortion = opt.lambda_distortion if iteration >= opt.distortion_from_iter else 0.0
        lambda_depth_normal = opt.lambda_depth_normal if iteration >= opt.depth_normal_from_iter else 0.0
        # Final loss
        loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion
        sdf_loss = torch.zeros(1)
        if iteration >= opt.loss_sdf_start and iteration % opt.loss_sdf_interval == 0 and gaussians.use_kernel_mlp:
            sampled_sdf = gaussians.sample_sdf_points(xyz_back, visibility_filter)  # (1e5,3)
            sdf_gt = mesh_depth_tool.cal_sdf_basedon_depth(
                viewpoint_cam.image_height, viewpoint_cam.image_width, viewpoint_cam_p3d, sampled_sdf, depth)
            sdf_esti = gaussians.compute_sdf_vectorized(sampled_sdf)
            # import ipdb;ipdb.set_trace()
            sdf_loss = mesh_depth_tool.weighted_sdf_loss(sdf_esti, sdf_gt)
            print(sdf_loss)
            loss = loss + opt.weight_sdf * sdf_loss

        loss.backward()

        iter_end.record()

        is_save_images = True
        if is_save_images and (iteration % opt.save_images_interval == 0):
            with torch.no_grad():
                eval_cam = allCameras[random.randint(0, len(allCameras) - 1)]
                voxel_visible_mask2 = prefilter_voxel(viewpoint_cam, gaussians, pipe, background,
                                                      kernel_size=dataset.kernel_size)
                rendering = render_neuralGS(eval_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size,
                                            visible_mask=voxel_visible_mask2)["render"]
                image = rendering[:3, :, :]
                # transformed_image = L1_loss_appearance(image, eval_cam.original_image.cuda(), gaussians, eval_cam.idx, return_transformed_image=True)
                normal = rendering[3:6, :, :]
                normal = torch.nn.functional.normalize(normal, p=2, dim=0)

            # transform to world space
            c2w = (eval_cam.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
            normal = normal2.reshape(3, *normal.shape[1:])
            normal = (normal + 1.) / 2.

            depth = rendering[6, :, :]
            depth_normal, _ = depth_to_normal(eval_cam, depth[None, ...])
            depth_normal = (depth_normal + 1.) / 2.
            depth_normal = depth_normal.permute(2, 0, 1)

            gt_image = eval_cam.original_image.cuda()

            depth_map = apply_depth_colormap(depth[..., None], rendering[7, :, :, None], near_plane=None,
                                             far_plane=None)
            depth_map = depth_map.permute(2, 0, 1)

            accumlated_alpha = rendering[7, :, :, None]
            colored_accum_alpha = apply_depth_colormap(accumlated_alpha, None, near_plane=0.0, far_plane=1.0)
            colored_accum_alpha = colored_accum_alpha.permute(2, 0, 1)

            distortion_map = rendering[8, :, :]
            distortion_map = colormap(distortion_map.detach().cpu().numpy()).to(normal.device)

            row0 = torch.cat([gt_image, image, depth_normal, normal], dim=2)
            row1 = torch.cat([depth_map, colored_accum_alpha, distortion_map, image], dim=2)

            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0, 1)

            os.makedirs(f"{dataset.model_path}/log_images", exist_ok=True)
            torchvision.utils.save_image(image_to_show, f"{dataset.model_path}/log_images/{iteration}.jpg")
            torchvision.utils.save_image(depth_map, f"{dataset.model_path}/log_images/{iteration}_depth_uid{viewpoint_cam.uid}.jpg")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, sdf_loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render_neuralGS, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification of GOF
            # if iteration < opt.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #                                                          radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
            #         gaussians.compute_3D_filter(cameras=trainCameras)
            #
            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()
            #
            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=trainCameras)

            # Densification of SCFD
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask,
                                          voxel_visible_mask)

                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold,
                                            grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                    gaussians.compute_3D_filter(cameras=trainCameras)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            if iteration % 100 == 0 and iteration > opt.update_until:
                if iteration < opt.update_until - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            if torch.isnan(gaussians.get_anchor).any():
                import ipdb;
                ipdb.set_trace()
                raise Exception('fucking NAN of anchor opt before')
            if torch.isnan(gaussians._anchor_feat).any():
                import ipdb;
                ipdb.set_trace()
                raise Exception('fucking NAN of feat opt before')

            # Optimizer step
            if iteration < opt.iterations:
                # clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                # nan_flag = gaussians.prune_nan_anchor()
                # if nan_flag:
                #     print('find NAN, so computing 3D filter')
                #     gaussians.compute_3D_filter(cameras=trainCameras)

                if torch.isnan(gaussians.get_anchor).any():
                    import ipdb;
                    ipdb.set_trace()
                    raise Exception('fucking NAN of anchor opt')
                if torch.isnan(gaussians._anchor_feat).any():
                    import ipdb;
                    ipdb.set_trace()
                    raise Exception('fucking NAN of feat opt')

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, sdf_loss, l1_loss, elapsed, testing_iterations, scene: NeuralScene,
                    renderFunc,
                    renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)

        tb_writer.add_scalar('train_loss_patches/sdf_loss', sdf_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        # tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_anchor.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask3 = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    rendering = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask3)[
                        "render"]
                    image = rendering[:3, :, :]
                    normal = rendering[3:6, :, :]
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/errormap".format(viewpoint.image_name),
                                             (gt_image[None] - image[None]).abs(), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)

        torch.cuda.empty_cache()
        scene.gaussians.train()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # parser.add_argument("--gpu", type=str, default='-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # if args.gpu != '-1':
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    #     os.system("echo $CUDA_VISIBLE_DEVICES")
    #     print(f'using GPU {args.gpu}')

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # torch.cuda.set_device(torch.device("cuda:0")) # fucking work ?
    torch.cuda.set_device(torch.device(0))  # fucking work ?

    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    if args.warmup:
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
                 args.checkpoint_iterations, args.start_checkpoint, args.debug_from, ply_path=new_ply_path)
    # All done
    print("\nTraining complete.")
