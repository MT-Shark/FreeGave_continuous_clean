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
import shutil
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel_incremental_net
from utils.general_utils import safe_state, get_linear_noise_func,quaternion_multiply
import uuid
import einops
from tqdm import tqdm
import lpips
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random
from itertools import groupby
import torchvision
import time
import json
from evaluator import MetricsEvaluatorConfig, MetricsEvaluator

# Configure evaluator
config = MetricsEvaluatorConfig(
    device=torch.device("cuda"),
    enable_semantics=False,
    enable_imaging=False,
    enable_basic_metrics=False,
    enable_fourier=True
)
evaluator = MetricsEvaluator(config)

def static_train(scene, gaussians , dataset, opt, pipe, iterations = 3000 ):
    d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
    ema_loss_for_log = 0.0
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    progress_bar = tqdm(range(1 , iterations + 1), desc="Training progress")
    viewpoint_stack = scene.getInitCameras().copy()
    for iteration in range(1, iterations+1):
        # Render
        if not viewpoint_stack:
            viewpoint_stack = scene.getInitCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam.load2device()
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, depth_filter = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re[
            "depth_filter"]
        # depth = render_pkg_re["depth"]
        
        # Loss
        # gt_image = viewpoint_cam.original_image.cuda()
        # Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # loss.backward()
        
        if args.max_depth > 0:
            depth = viewpoint_cam.depth
            depth_mask = depth < args.max_depth
            depth_mask = depth_mask.unsqueeze(0)  # [1, H, W]，用于广播
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image*depth_mask, gt_image*depth_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        elif args.seg_mask:
            seg = viewpoint_cam.seg
            seg_mask = (seg != 0) & (seg != 255)  # 只保留非背景区域
            seg_mask = seg_mask.unsqueeze(0)
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image*seg_mask, gt_image*seg_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        if iteration % 1000 == 0 :
            torchvision.utils.save_image(image, os.path.join("{0:04}.png".format(iteration)))
            torchvision.utils.save_image(gt_image, os.path.join("{0:04}_gt.png".format(iteration)))
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            # if iteration == opt.iterations:
            #     progress_bar.close()
            
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                radii[visibility_filter])

            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.size_threshold if iteration > opt.opacity_reset_interval else None
        
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                size_threshold)
                    # gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

                if iteration % opt.opacity_reset_interval == 0 and iteration != iterations or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)

    scene.save(iteration)
    return 


def sampleTainingViews(views, sample_num):
    sorted_list = sorted(views, key=lambda x: x.fid)
    incremental_views_group = [sorted(list(v), key=lambda x: x.img_path)    for k, v in groupby(sorted_list , key=lambda x: x.fid)] # 所有views 根据time stamp 分组，组内按照文件名排序
    num_train_views = len(incremental_views_group[0])
    if num_train_views > args.sampleTrainingView:
        index_sample = random.sample(range(num_train_views), args.sampleTrainingView)
        incremental_views_group = [ [v[i] for i in index_sample] for v in incremental_views_group]   
        # 展开incremental_views_group
        views_sampled = [view for group in incremental_views_group for view in group]
        return views_sampled
    else:
        return views

def training(dataset, opt, pipe, testing_iterations, saving_iterations, n_keys, vel_start_time=0.0):

    '''初始化模型以及优化第一帧的GS'''
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    

    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians , shuffle= False )


    # 根据 fps 和 total frame 决定 dt
    train_json_path = os.path.join(args.source_path,'transforms_train.json') # hardcode for physics bench only
    # 读取 JSON 文件并处理
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    # 获取 total_frames 和 fps
    total_frames = data['total_frames']
    fps = data['fps']
    args.dt = 1 / total_frames
    # 输出结果
    print(f"total_frames: {total_frames}, fps: {fps}, args.dt: {args.dt}")
    kwargs = {'dt':args.dt ,  'cn_init': args.cn_init , 
              'with_node_weight': args.with_node_weight , 'cn_KNN_method': args.cn_KNN_method , 'cn_KNN': args.cn_KNN ,
              'cn_hyperdim' : args.cn_hyperdim ,'zero_padding':args.zero_padding }
    deform = deform = DeformModel_incremental_net(control_num= dataset.control_num , **kwargs)
    deform.vel_train_setting(opt)
    deform.train_setting(opt)

    
    gaussians.training_setup(opt)

    
    static_start= time.time()
    if args.DynamicGS_dataset: # 对于这个数据集特别处理
        static_train_dynamicgs(scene , gaussians, dataset, opt, pipe, iterations=args.static_warmup_iter)
    else:
        static_train(scene , gaussians, dataset, opt, pipe, iterations=args.static_warmup_iter)
    static_end = time.time()
    static_total_time= static_end - static_start
    with open(os.path.join(args.model_path,"static_time.txt"), 'w') as file:
        file.write(f"Total Time: {static_total_time}\n")

    gaussian_num_dict = {0: gaussians.get_xyz.shape[0]}
    control_num = dataset.control_num
    indices = torch.randperm(control_num)
    sample_xyz =gaussians.get_xyz[indices[:control_num]]  # 从 gaussians.get_xyz 中采样sample_num个点
    deform.control_node_warp.init(sample_xyz)
    deform.control_node_warp.train_setting(opt)
        
    #print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


    ''' 获得相机位姿 并且根据最大训练时间重新分配'''
    train_views = scene.getTrainCameras()
    val_views = scene.getValCameras()
    

    # 获得 train 相机位姿
    sorted_list = sorted(train_views, key=lambda x: x.fid)
    incremental_views_group = [sorted(list(v), key=lambda x: x.img_path)    for k, v in groupby(sorted_list , key=lambda x: x.fid)] # 所有views 根据time stamp 分组，组内按照文件名排序

    # 获得val 相机位姿
    sorted_val_list = sorted(val_views, key=lambda x: x.fid)
    incremental_val_views_group = [sorted(list(v), key=lambda x: x.img_path)   for k, v in groupby(sorted_val_list , key=lambda x: x.fid)] # 所有views 根据time stamp 分组，组内按照文件名排序

    if not args.noStatic_mask:
        render_path = os.path.join(args.model_path,"increment_{}_{}".format(args.iter_basis , args.iter_gs))
    else:
        render_path = os.path.join(args.model_path,"increment_{}_{}_noStaticMask".format(args.iter_basis , args.iter_gs))
    os.makedirs(render_path,exist_ok=True)
    os.makedirs(os.path.join(render_path, "test"), exist_ok=True)
    os.makedirs(os.path.join(render_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(render_path, "val"), exist_ok=True)
    
    
    ''' 初始化各个变量'''
    gaussians.training_setup(opt)
    d_xyz_n = torch.zeros_like(deform.control_node_warp.nodes).cuda()
    d_rot_n= torch.zeros(control_num,4).cuda()
    d_rot_n[...,0] = 1 
    d_scaling_n = torch.zeros(control_num,3).cuda()
    d_xyz = torch.zeros_like(gaussians.get_xyz).cuda()
    d_rot= torch.zeros_like(gaussians.get_rotation).cuda()
    d_rot[...,0] = 1 
    d_scaling = torch.zeros_like(gaussians.get_scaling).cuda()
    use_control_node = True

    weights_opt_count = 0 # incremental phase
    weights_opt_time = 0 # incremental phase
    gs_opt_count = 0 # per frame
    gs_opt_time = 0
    gs_opt_iter = args.iter_gs

    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
    psnr_dict = {'train':[] , 'val':[] , 'trainingview_test':[] , 'novelview_test': []}
    lpips_dict = {'train':[] , 'val':[] , 'trainingview_test':[], 'novelview_test': []}
    ssim_dict = {'train':[] , 'val':[] , 'trainingview_test':[], 'novelview_test': []}
    dy_syn_dict = {'trainingview_test':[], 'novelview_test': []}
    dy_syn_previous_dict = {'trainingview_test':[], 'novelview_test': []}
    dy_tra_log_dict = {'trainingview_test':[], 'novelview_test': []}
    dy_tra_kl_dict = {'trainingview_test':[], 'novelview_test': []}
    dy_tra_previous_dict = {'trainingview_test':[], 'novelview_test': []}

    print('Incremental Training Starts!')
    start1 = time.time()

    '''开始增量训练'''
    for idx, view_group in enumerate(tqdm(incremental_views_group, desc="Rendering progress")):
        if idx == 0 :
            continue
        if idx >= len(incremental_val_views_group) - args.predict_steps:
            break
        weights_opt_count += 1
        weights_opt_start = time.time()
        if idx < 10: # 前10轮优化control node
            opt_controlNode = True
        else:
            opt_controlNode = False
        for i in range(args.iter_basis):# 进入优化循环
            # 从view_group随机抽取一个相机视角
            view = random.choice(view_group)
            view.load2device()
            

            # TODO: 
            nodes = deform.control_node_warp.nodes
            deform_code = deform.code_field(gaussians.get_xyz.detach())
            deform_seg = deform.code_field.seg(deform_code)
            deform_code_node = deform.code_field(nodes.detach())
            deform_seg_node = deform.code_field.seg(deform_code_node)

            n_weight, _, nn_idxs = deform.control_node_warp.cal_nn_weight(gaussians.get_xyz.detach(),deform_seg,deform_seg_node)

            # d_scaling_incre = torch.zeros_like(gaussians.get_scaling)
            d_xyz_incre_n, d_rot_incre_n = deform.incremental_step(nodes.detach() + d_xyz_n.detach(),view.fid.expand(nodes.shape[0], 1) , deform_seg_node, dt = args.dt)
            
            d_xyz_total_n = d_xyz_n.detach() + d_xyz_incre_n
            d_rot_total_n = quaternion_multiply(d_rot_incre_n, d_rot_n.detach())
            d_xyz_total, d_rot_total , d_scaling = deform.control_node_warp.cal_deform(n_weight.detach(),nn_idxs.detach(), d_xyz_total_n, d_rot_total_n, d_scaling_n,  method = args.cn_interpolate_method)
            # Render
            
            render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total, d_rot_total, d_scaling, dataset.is_6dof)
            image= render_pkg_re["render"]
            
            # # 计算loss
            # gt_image = view.original_image.cuda()
            # Ll1 = l1_loss(image, gt_image)
            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            
            # 根据 depth，只计算近处物体的 loss
            if args.max_depth > 0:
                depth = view.depth
                depth_mask = depth < args.max_depth
                depth_mask = depth_mask.unsqueeze(0)  # [1, H, W]，用于广播
                gt_image = view.original_image.cuda()
                Ll1 = l1_loss(image*depth_mask, gt_image*depth_mask)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            elif args.seg_mask:
                seg = view.seg
                seg_mask = (seg != 0) & (seg != 255)  # 只保留非背景区域
                seg_mask = seg_mask.unsqueeze(0)
                gt_image = view.original_image.cuda()
                Ll1 = l1_loss(image*seg_mask, gt_image*seg_mask)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            else:
                gt_image = view.original_image.cuda()
                Ll1 = l1_loss(image, gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                
            # backward
            loss.backward()
            # update

            deform.vel_optimizer.step()
            deform.vel_optimizer.zero_grad()
            deform.optimizer.step()
            deform.optimizer.zero_grad()
            if opt_controlNode: # 前10轮
                deform.control_node_warp.optimizer.step()
                deform.control_node_warp.optimizer.zero_grad()

            # view.load2device('cpu')
            # can include gs into optimization
        weights_opt_end = time.time()
        weights_opt_time += (weights_opt_end-weights_opt_start)
        
        '''额外对gaussian 优化'''
        gs_opt_iter = args.iter_gs
        gs_opt_count += 1
        gs_opt_start = time.time()
        for i in range(gs_opt_iter):
            view = random.choice(view_group)
            view.load2device()
            # Render
            render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total.detach(), d_rot_total.detach(), d_scaling.detach(), dataset.is_6dof)
            image = render_pkg_re["render"]
            # 计算loss
            gt_image = view.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            # 根据 depth，只计算近处物体的 loss
            if args.max_depth > 0:
                depth = view.depth
                depth_mask = depth < args.max_depth
                depth_mask = depth_mask.unsqueeze(-1)  # [H, W, 1]，用于广播
                loss = (loss * depth_mask).sum() / depth_mask.sum().clamp(min=1)
            # backward
            loss.backward()
            # update
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            # deform.optimizer.zero_grad()
        gs_opt_end= time.time()
        gs_opt_time += (gs_opt_end - gs_opt_start)
        for view in view_group:
            view.load2device('cpu')
        '''end'''
        if use_control_node:
            d_xyz_n = d_xyz_total_n
            d_rot_n = d_rot_total_n
        d_xyz = d_xyz_total
        d_rot = d_rot_total
        
        if view.fid.item() <= args.incremental_warmup_time:
            continue
        if view.fid.item() >= args.incremental_max_time and not args.skip_render :
            return
        end1 = time.time()
        total_time = (end1 - start1)
        with open(os.path.join(args.model_path,"warmup_time.txt"), 'w') as file:
            file.write(f"Total Time: {total_time}\n")
            
        ''' 预测接下来几帧，并且计算metrics'''
        if not args.skip_render:
            test_render = os.path.join(render_path, "test", "{0:04}".format(idx) , "render")
            test_gt = os.path.join(render_path, "test", "{0:04}".format(idx) , "gt")
            test_mask = os.path.join(render_path, "test", "{0:04}".format(idx) , "mask")
            os.makedirs(test_render, exist_ok=True)
            os.makedirs(test_gt, exist_ok=True)
            os.makedirs(test_mask, exist_ok=True)

            # os.makedirs(os.path.join(render_path, "train", "{0:04}".format(idx)), exist_ok=True)
            train_render = os.path.join(render_path, "train", "{0:04}".format(idx) , "render")
            train_gt = os.path.join(render_path, "train", "{0:04}".format(idx) , "gt")
            train_mask = os.path.join(render_path, "train", "{0:04}".format(idx) , "mask")
            os.makedirs(train_render, exist_ok=True)
            os.makedirs(train_gt, exist_ok=True)
            os.makedirs(train_mask, exist_ok=True)

            os.makedirs(os.path.join(render_path, "val", "{0:04}".format(idx)), exist_ok=True)
            val_render = os.path.join(render_path, "val", "{0:04}".format(idx) , "render")
            val_gt = os.path.join(render_path, "val", "{0:04}".format(idx) , "gt")
            val_mask = os.path.join(render_path, "val", "{0:04}".format(idx) , "mask")
            os.makedirs(val_render, exist_ok=True)
            os.makedirs(val_gt, exist_ok=True)
            os.makedirs(val_mask, exist_ok=True)


        with torch.no_grad():
            psnr_list_tmp = []
            ssim_list_tmp = []
            lpips_list_tmp = []
            for index, view in  enumerate(view_group): # 训练视角
                fid_tmp = incremental_views_group[idx][0].fid.cuda()
                view.load2device() 
                # 渲染图像
                render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total, d_rot_total, d_scaling, dataset.is_6dof)
                image = render_pkg_re["render"]  # [3, H, W]
                
                # 初始化 mask 为全 1（与 image 形状匹配）
                mask = torch.ones_like(image, dtype=torch.float32)  # [3, H, W]

                # 构造 depth mask（如果启用）
                if args.max_depth > 0:
                    depth = view.depth  # [H, W]
                    depth_mask = (depth < args.max_depth).float().unsqueeze(0)  # [1, H, W]
                    mask = mask * depth_mask  # 广播乘法，mask 仍为 [3, H, W]
                elif args.seg_mask:
                    seg = view.seg
                    seg_mask = (seg != 0) & (seg != 255)  # 只保留非背景区域
                    seg_mask = seg_mask.unsqueeze(0)
                    mask = mask * seg_mask

                # 应用 mask 到图像和 GT
                gt_image = view.original_image
                render_masked = image * mask
                gt_masked = gt_image * mask

                # 计算指标
                psnr_list_tmp.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                ssim_list_tmp.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                lpips_list_tmp.append(lpips_fn(render_masked, gt_masked).item())

                # 不渲染当前时刻
                # if not args.skip_render and fid_tmp > 0.5 and fid_tmp < 0.75 :
                #     torchvision.utils.save_image(render_masked, os.path.join(train_render, f"view{view.view}_{view.image_name}_masked.jpg"))
                #     torchvision.utils.save_image(gt_masked, os.path.join(train_gt, f"{view.view}_{view.image_name}_masked.jpg"))

                
                view.load2device('cpu')
            mean_psnr = torch.tensor(psnr_list_tmp).mean().item()
            mean_ssim = torch.tensor(ssim_list_tmp).mean().item()
            mean_lpips = torch.tensor(lpips_list_tmp).mean().item()
            psnr_dict['train'].append(mean_psnr)
            ssim_dict['train'].append(mean_ssim)
            lpips_dict['train'].append(mean_lpips)

            psnr_list_tmp = []
            ssim_list_tmp = []
            lpips_list_tmp = []
            for index,  view in enumerate(incremental_val_views_group[idx]): # val视角
                fid_tmp = incremental_views_group[idx][0].fid.cuda()
                view.load2device()
                render_pkg_re = render(view, gaussians, pipe, background, d_xyz, d_rot, d_scaling, dataset.is_6dof)
                image = render_pkg_re["render"]
                # 初始化 mask 为全 1（与 image 形状匹配）
                mask = torch.ones_like(image, dtype=torch.float32)  # [3, H, W]

                # 构造 depth mask（如果启用）
                if args.max_depth > 0:
                    depth = view.depth  # [H, W]
                    depth_mask = (depth < args.max_depth).float().unsqueeze(0)  # [1, H, W]
                    mask = mask * depth_mask  # 广播乘法，mask 仍为 [3, H, W]
                elif args.seg_mask:
                    seg = view.seg
                    seg_mask = (seg != 0) & (seg != 255)  # 只保留非背景区域
                    seg_mask = seg_mask.unsqueeze(0)
                    mask = mask * seg_mask

                # 应用 mask 到图像和 GT
                gt_image = view.original_image
                render_masked = image * mask
                gt_masked = gt_image * mask

                # 计算指标
                psnr_list_tmp.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                ssim_list_tmp.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                lpips_list_tmp.append(lpips_fn(render_masked, gt_masked).item())

                # 不渲染当前时刻
                # if not args.skip_render and fid_tmp > 0.5 and fid_tmp < 0.75 :
                #     torchvision.utils.save_image(render_masked, os.path.join(val_render, f"view{view.view}_{view.image_name}_masked.jpg"))
                #     torchvision.utils.save_image(gt_masked, os.path.join(val_gt, f"view{view.view}_{view.image_name}_masked.jpg"))

                view.load2device('cpu')

            mean_psnr = torch.tensor(psnr_list_tmp).mean().item()
            mean_ssim = torch.tensor(ssim_list_tmp).mean().item()
            mean_lpips = torch.tensor(lpips_list_tmp).mean().item()
            psnr_dict['val'].append(mean_psnr)
            ssim_dict['val'].append(mean_ssim)
            lpips_dict['val'].append(mean_lpips)
                    
                    
            # # 渲染增量的test 视角
            length_train = len(incremental_views_group)
            test_num = args.predict_steps

            idx1 = min(length_train, idx + test_num + 1) # 基于当前的未来的视角可能在 train,val中

            os.makedirs(args.model_path, exist_ok=True)
            count  = 0
            
            d_xyz_n_tmp = d_xyz_n
            d_rot_n_tmp = d_rot_n
            d_xyz_tmp = d_xyz
            d_rot_tmp = d_rot
            d_scaling_tmp = d_scaling   
            psnr_list_tmp_training_view = []
            ssim_list_tmp_training_view = []
            lpips_list_tmp_training_view = []

            psnr_list_tmp_novel_view = []
            ssim_list_tmp_novel_view = []
            lpips_list_tmp_novel_view = []



            from collections import defaultdict

            pred_dict_train = defaultdict(list)
            gt_dict_train = defaultdict(list)
            pred_dict_val = defaultdict(list)
            gt_dict_val = defaultdict(list)
            for j in range(idx + 1, idx1):
                '''基于现在的weights 以及 xyz 向前一步'''
                fid_tmp = incremental_views_group[idx][0].fid.cuda()
                # 调用deform_incremental内的函数，得到 d_xyz, d_rot, d_scale
                d_xyz_incre_n_tmp, d_rot_incre_n_tmp = deform.incremental_step(nodes + d_xyz_n_tmp.detach(), fid_tmp.expand(nodes.shape[0], 1) , deform_seg_node.detach() , dt = args.dt)
                # 调用 cal_deform ， 输入权重矩阵，index，nodes的位移旋转，得到其余gs的增量位移旋转
                d_xyz_incre_tmp , d_rot_incre_tmp , d_scaling = deform.control_node_warp.cal_deform(n_weight , nn_idxs , d_xyz_incre_n_tmp , d_rot_incre_n_tmp , d_scaling_n,  method = args.cn_interpolate_method)
                d_xyz_total_tmp = d_xyz_incre_tmp + d_xyz_tmp.detach()
                d_rot_total_tmp = quaternion_multiply(d_rot_incre_tmp, d_rot_tmp.detach())

                # Render
                for view in incremental_views_group[j]:
                    view.load2device()
                    render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total_tmp, d_rot_total_tmp, d_scaling, dataset.is_6dof)
                    image = render_pkg_re["render"]
                        # 初始化 mask 为全 1（与 image 形状匹配）
                    mask = torch.ones_like(image, dtype=torch.float32)  # [3, H, W]

                    # 构造 depth mask（如果启用）
                    if args.max_depth > 0:
                        depth = view.depth  # [H, W]
                        depth_mask = (depth < args.max_depth).float().unsqueeze(0)  # [1, H, W]
                        mask = mask * depth_mask  # 广播乘法，mask 仍为 [3, H, W]
                    elif args.seg_mask:
                        seg = view.seg
                        seg_mask = (seg != 0) & (seg != 255)  # 只保留非背景区域
                        seg_mask = seg_mask.unsqueeze(0)
                        mask = mask * seg_mask

                    # 应用 mask 到图像和 GT
                    gt_image = view.original_image
                    render_masked = image * mask
                    gt_masked = gt_image * mask
                    pred_dict_train[view.view].append(render_masked) # 如果 pred_dict[view.view] 不存在则先 初始化为 空列表
                    gt_dict_train[view.view].append(gt_masked) # 如果 gt_dict[view.view] 不存在则先 初始化为 空列表

                    # 计算指标
                    psnr_list_tmp_training_view.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    ssim_list_tmp_training_view.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    lpips_list_tmp_training_view.append(lpips_fn(render_masked, gt_masked).item())

                    # 保存 masked 图像（如果不跳过）
                    if not args.skip_render and fid_tmp > 0 and fid_tmp < 0.5 :
                        torchvision.utils.save_image(render_masked, os.path.join(test_render, f"view{view.view}_{view.image_name}_masked.jpg"))
                        torchvision.utils.save_image(gt_masked, os.path.join(test_gt, f"view{view.view}_{view.image_name}_masked.jpg"))
                    count+=1
                    view.load2device('cpu')

                for view in incremental_val_views_group[j]:
                    view.load2device()
                    render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total_tmp, d_rot_total_tmp, d_scaling, dataset.is_6dof)
                    image = render_pkg_re["render"]
                        # 初始化 mask 为全 1（与 image 形状匹配）
                    mask = torch.ones_like(image, dtype=torch.float32)  # [3, H, W]

                    # 构造 depth mask（如果启用）
                    if args.max_depth > 0:
                        depth = view.depth  # [H, W]
                        depth_mask = (depth < args.max_depth).float().unsqueeze(0)  # [1, H, W]
                        mask = mask * depth_mask  # 广播乘法，mask 仍为 [3, H, W]
                    elif args.seg_mask:
                        seg = view.seg
                        seg_mask = (seg != 0) & (seg != 255)  # 只保留非背景区域
                        seg_mask = seg_mask.unsqueeze(0)
                        mask = mask * seg_mask

                    # 应用 mask 到图像和 GT
                    gt_image = view.original_image
                    render_masked = image * mask
                    gt_masked = gt_image * mask
                    pred_dict_val[view.view].append(render_masked) # 如果 pred_dict[view.view] 不存在则先 初始化为 空列表
                    gt_dict_val[view.view].append(gt_masked) # 如果 gt_dict[view.view] 不存在则先 初始化为 空列表

                    # 计算指标
                    psnr_list_tmp_novel_view.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    ssim_list_tmp_novel_view.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    lpips_list_tmp_novel_view.append(lpips_fn(render_masked, gt_masked).item())

                    # 保存 masked 图像（如果不跳过）
                    if not args.skip_render and fid_tmp > 0 and fid_tmp < 0.5 :
                        torchvision.utils.save_image(render_masked, os.path.join(test_render, f"view{view.view}_{view.image_name}_masked.jpg"))
                        torchvision.utils.save_image(gt_masked, os.path.join(test_gt, f"view{view.view}_{view.image_name}_masked.jpg"))
                    count+=1
                    view.load2device('cpu')

                d_xyz_n_tmp = d_xyz_n_tmp + d_xyz_incre_n_tmp
                d_xyz_tmp = d_xyz_total_tmp
                d_rot_tmp = d_rot_total_tmp
                
                pass

            dy_syn_list_tmp_training_view = []
            dy_syn_previous_list_tmp_training_view = []
            dy_tra_log_list_tmp_training_view = []
            dy_tra_kl_list_tmp_training_view = []
            dy_tra_previous_list_tmp_training_view = []
            

            for key,value in pred_dict_train.items():
                pred_batch = torch.stack(pred_dict_train[key]).unsqueeze(0)
                gt_batch = torch.stack(gt_dict_train[key]).unsqueeze(0)
                batch = {
                    "gt": gt_batch ,      # (B, T, C, H, W) in [0, 1]
                    "output": pred_batch,
                "captions": ["None"]
                }
                results = evaluator.evaluate(batch=batch)
                dy_syn_list_tmp_training_view.append(float(results.get('DySyn', float('nan'))))
                dy_syn_previous_list_tmp_training_view.append(float(results.get('DySyn_previous', float('nan'))))
                dy_tra_log_list_tmp_training_view.append(float(results.get('DyTra_log', float('nan'))))
                dy_tra_kl_list_tmp_training_view.append(float(results.get('DyTra_kl', float('nan'))))
                dy_tra_previous_list_tmp_training_view.append(float(results.get('DyTra_previous', float('nan'))))

            mean_psnr_trainingview = torch.tensor(psnr_list_tmp_training_view).mean().item()
            mean_ssim_trainingview = torch.tensor(ssim_list_tmp_training_view).mean().item()
            mean_lpips_trainingview = torch.tensor(lpips_list_tmp_training_view).mean().item() 
            mean_dy_syn_trainingview = torch.tensor(dy_syn_list_tmp_training_view).mean().item()
            mean_dy_syn_previous_trainingview = torch.tensor(dy_syn_previous_list_tmp_training_view).mean().item()
            mean_dy_tra_log_trainingview = torch.tensor(dy_tra_log_list_tmp_training_view).mean().item()
            mean_dy_tra_kl_trainingview = torch.tensor(dy_tra_kl_list_tmp_training_view).mean().item()
            mean_dy_tra_previous_trainingview = torch.tensor(dy_tra_previous_list_tmp_training_view).mean().item()
            psnr_dict['trainingview_test'].append(mean_psnr_trainingview)
            ssim_dict['trainingview_test'].append(mean_ssim_trainingview)
            lpips_dict['trainingview_test'].append(mean_lpips_trainingview)
            dy_syn_dict['trainingview_test'].append(mean_dy_syn_trainingview)
            dy_syn_previous_dict['trainingview_test'].append(mean_dy_syn_previous_trainingview)
            dy_tra_log_dict['trainingview_test'].append(mean_dy_tra_log_trainingview)
            dy_tra_kl_dict['trainingview_test'].append(mean_dy_tra_kl_trainingview)
            dy_tra_previous_dict['trainingview_test'].append(mean_dy_tra_previous_trainingview)

            dy_syn_list_tmp_novel_view = []
            dy_syn_previous_list_tmp_novel_view = []
            dy_tra_log_list_tmp_novel_view = []
            dy_tra_kl_list_tmp_novel_view = []
            dy_tra_previous_list_tmp_novel_view = []
            for key,value in pred_dict_val.items():
                pred_batch = torch.stack(pred_dict_val[key]).unsqueeze(0)
                gt_batch = torch.stack(gt_dict_val[key]).unsqueeze(0)
                batch = {
                    "gt": gt_batch ,      # (B, T, C, H, W) in [0, 1]
                    "output": pred_batch,
                "captions": ["None"]
                }
                results = evaluator.evaluate(batch=batch)
                dy_syn_list_tmp_novel_view.append(float(results.get('DySyn', float('nan'))))
                dy_syn_previous_list_tmp_novel_view.append(float(results.get('DySyn_previous', float('nan'))))
                dy_tra_log_list_tmp_novel_view.append(float(results.get('DyTra_log', float('nan'))))
                dy_tra_kl_list_tmp_novel_view.append(float(results.get('DyTra_kl', float('nan'))))
                dy_tra_previous_list_tmp_novel_view.append(float(results.get('DyTra_previous', float('nan'))))

            mean_psnr_novelview = torch.tensor(psnr_list_tmp_novel_view).mean().item()
            mean_ssim_novelview = torch.tensor(ssim_list_tmp_novel_view).mean().item()
            mean_lpips_novelview = torch.tensor(lpips_list_tmp_novel_view).mean().item()
            mean_dy_syn_novelview = torch.tensor(dy_syn_list_tmp_novel_view).mean().item()
            mean_dy_syn_previous_novelview = torch.tensor(dy_syn_previous_list_tmp_novel_view).mean().item()
            mean_dy_tra_log_novelview = torch.tensor(dy_tra_log_list_tmp_novel_view).mean().item()
            mean_dy_tra_kl_novelview = torch.tensor(dy_tra_kl_list_tmp_novel_view).mean().item()
            mean_dy_tra_previous_novelview = torch.tensor(dy_tra_previous_list_tmp_novel_view).mean().item()

            psnr_dict['novelview_test'].append(mean_psnr_novelview)
            ssim_dict['novelview_test'].append(mean_ssim_novelview)
            lpips_dict['novelview_test'].append(mean_lpips_novelview)
            dy_syn_dict['novelview_test'].append(mean_dy_syn_novelview)
            dy_syn_previous_dict['novelview_test'].append(mean_dy_syn_previous_novelview)
            dy_tra_log_dict['novelview_test'].append(mean_dy_tra_log_novelview)
            dy_tra_kl_dict['novelview_test'].append(mean_dy_tra_kl_novelview)
            dy_tra_previous_dict['novelview_test'].append(mean_dy_tra_previous_novelview)

            del pred_dict_train, gt_dict_train , pred_dict_val, gt_dict_val


            
        
    end1 = time.time()
    total_time = (end1 - start1)
    avg_time = total_time / len(incremental_views_group)

    weights_opt_time_avg = weights_opt_time / weights_opt_count
    gs_opt_time_avg = gs_opt_time / gs_opt_count
    with open(os.path.join(render_path,"running_time.txt"), 'w') as file:
        file.write(f"Total Time: {total_time}\n")
        file.write(f"Average Time Per Frame: {avg_time}\n")
        file.write(f"Average weights optimization time Per Frame: {weights_opt_time_avg}\n")
        file.write(f"Average gs optimization time Per Frame: {gs_opt_time_avg}\n")        
    with  open(os.path.join(args.model_path,"gs_num_record.txt"), 'w')as json_file:
        json.dump(gaussian_num_dict, json_file)
    
    psnr_dict['train_mean'] = np.mean(psnr_dict['train'])
    ssim_dict['train_mean'] = np.mean(ssim_dict['train'])
    lpips_dict['train_mean'] = np.mean(lpips_dict['train'])

    psnr_dict['val_mean'] = np.mean(psnr_dict['val'])
    ssim_dict['val_mean'] = np.mean(ssim_dict['val'])
    lpips_dict['val_mean'] = np.mean(lpips_dict['val'])

    psnr_dict['trainingview_test_mean'] = np.mean(psnr_dict['trainingview_test'])
    ssim_dict['trainingview_test_mean'] = np.mean(ssim_dict['trainingview_test'])
    lpips_dict['trainingview_test_mean'] = np.mean(lpips_dict['trainingview_test'])
    dy_syn_dict['trainingview_test_mean'] = np.mean(dy_syn_dict['trainingview_test'])
    dy_syn_previous_dict['trainingview_test_mean'] = np.mean(dy_syn_previous_dict['trainingview_test'])
    dy_tra_log_dict['trainingview_test_mean'] = np.mean(dy_tra_log_dict['trainingview_test'])
    dy_tra_kl_dict['trainingview_test_mean'] = np.mean(dy_tra_kl_dict['trainingview_test'])
    dy_tra_previous_dict['trainingview_test_mean'] = np.mean(dy_tra_previous_dict['trainingview_test'])

    psnr_dict['novelview_test_mean'] = np.mean(psnr_dict['novelview_test'])
    ssim_dict['novelview_test_mean'] = np.mean(ssim_dict['novelview_test'])
    lpips_dict['novelview_test_mean'] = np.mean(lpips_dict['novelview_test'])    
    dy_syn_dict['novelview_test_mean'] = np.mean(dy_syn_dict['novelview_test'])
    dy_syn_previous_dict['novelview_test_mean'] = np.mean(dy_syn_previous_dict['novelview_test'])
    dy_tra_log_dict['novelview_test_mean'] = np.mean(dy_tra_log_dict['novelview_test'])
    dy_tra_kl_dict['novelview_test_mean'] = np.mean(dy_tra_kl_dict['novelview_test'])
    dy_tra_previous_dict['novelview_test_mean'] = np.mean(dy_tra_previous_dict['novelview_test'])

    with open(os.path.join(render_path, "psnr.json"), 'w') as file:
        json.dump(psnr_dict, file, indent=4)

    with open(os.path.join(render_path, "ssim.json"), 'w') as file:
        json.dump(ssim_dict, file, indent=4)

    with open(os.path.join(render_path, "lpips.json"), 'w') as file:
        json.dump(lpips_dict, file, indent=4)
    
    with open(os.path.join(render_path, "DySyn.json"), 'w') as file:
        json.dump(dy_syn_dict, file, indent=4)

    with open(os.path.join(render_path, "DySyn_previous.json"), 'w') as file:
        json.dump(dy_syn_previous_dict, file, indent=4)

    with open(os.path.join(render_path, "DyTra_log.json"), 'w') as file:
        json.dump(dy_tra_log_dict, file, indent=4)

    with open(os.path.join(render_path, "DyTra_kl.json"), 'w') as file:
        json.dump(dy_tra_kl_dict, file, indent=4)

    with open(os.path.join(render_path, "DyTra_previous.json"), 'w') as file:
        json.dump(dy_tra_previous_dict, file, indent=4)

    print('Incremental Training Completes!')
    return





def reassign_train_val_test(train_views, val_views, test_views, train_max_time=0.9):
    '''仅仅考虑test 时间戳大于 train_max_time 的情况'''
    # 创建新的视图列表
    new_train_views, new_val_views, new_test_views = [], [], []

    # 处理 train_views
    for view in train_views:
        if view.fid <= train_max_time:
            new_train_views.append(view)
        else:
            new_test_views.append(view)

    # 处理 val_views
    for view in val_views:
        if view.fid <= train_max_time:
            new_val_views.append(view)
        else:
            new_test_views.append(view)

    # 处理 test_views
    if test_views is not None:
        for i, view in enumerate(test_views):
            if view.fid > train_max_time:
                new_test_views.append(view)

    return new_train_views, new_val_views, new_test_views


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[40001])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--vel_start_time', type=float, default=0.0)
    parser.add_argument('--incremental_warmup_time', type=float, default=0.25) # period of global optimization
    parser.add_argument('--incremental_max_time', type=float, default=0.7) # max training timestamps
    parser.add_argument('--incremental_warmup_iterations', type=int, default=10000) # training iterations for global optimization
    parser.add_argument('--deform_max_time', type=float, default=0.2)
    parser.add_argument('--skip_warmup', action='store_true', help="Skip the warmup phase and load pre-trained parameters")
    parser.add_argument('--skip_render', action='store_true', default=False)
    parser.add_argument('--static_warmup_iter', type=int, default=3000)
    parser.add_argument('--warmup_iter_basis', type=int, default=75)
    parser.add_argument('--warmup_iter_gs', type=int, default=150)
    parser.add_argument('--iter_basis', type=int, default=20)
    parser.add_argument('--iter_gs', type=int, default=100)
    parser.add_argument('--noStatic_mask', action='store_true', default=False)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--sampleTrainingView', type=int, default=100) # 用于采样训练视角，当视角过多则使用全部

    parser.add_argument('--integrate_ksteps', type=int, default=5)
    parser.add_argument('--dt', type=float, default=1/88)
    parser.add_argument('--zero_padding', action='store_true', default=False)
    parser.add_argument('--cn_interpolate_method', type=str, default="linear")
    parser.add_argument('--predict_steps', type=int, default=10)

    parser.add_argument('--cn_init', type=str, default="zero")
    parser.add_argument('--with_node_weight', action='store_true', default=False)
    parser.add_argument('--cn_KNN', type=int, default=3)
    parser.add_argument('--cn_hyperdim', type=int, default=16) # 0 means KNN using xyz only
    parser.add_argument('--cn_KNN_method', type=str, default="deform_seg")

    parser.add_argument('--DynamicGS_dataset', action='store_true', default=False)
    parser.add_argument('--opt_CN_everyIter', action='store_true', default=False)
    parser.add_argument('--opt_CN_radius', action='store_true', default=False)

    # parser.add_argument('--seg_mask', action='store_true', help='Use segmentation mask')
    # parser.add_argument('--max_depth', type=float, default=10)




    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.K)

    # All done
    print("\nTraining complete.")
