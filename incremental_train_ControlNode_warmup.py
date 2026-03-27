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
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

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
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam.load2device()
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, depth_filter = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re[
            "depth_filter"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
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

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)

    scene.save(iteration)
    return 
def static_train_dynamicgs(scene, gaussians , dataset, opt, pipe, iterations = 3000 ):
    d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
    ema_loss_for_log = 0.0
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    progress_bar = tqdm(range(1 , iterations + 1), desc="Training progress")
    viewpoint_stack = scene.getInitCameras().copy()
    val_view_stack = scene.getValCameras().copy()
    for iteration in range(iterations):
        # Render
        if not viewpoint_stack:
            viewpoint_stack = scene.getInitCameras().copy()
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        viewpoint_cam.load2device()
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, depth_filter = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re[
            "depth_filter"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        # if iteration >0 and iteration%1000==0 :
        #     render_path = os.path.join(args.model_path,"increment_{}_{}_noStaticMask".format(args.iter_basis , args.iter_gs))
        #     os.makedirs(os.path.join(render_path, "static_vis", "{0:04}".format(iteration)), exist_ok=True)
        #     static_render = os.path.join(render_path, "static_vis", "{0:04}".format(iteration) , "render")
        #     static_gt = os.path.join(render_path, "static_vis", "{0:04}".format(iteration) , "gt")
        #     os.makedirs(static_render, exist_ok=True)
        #     os.makedirs(static_gt, exist_ok=True)
            
        #     for j, view in enumerate(val_view_stack[::135]):
        #         view.load2device()
        #         render_pkg_re = render(view, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        #         image= render_pkg_re["render"] 
    
        #         gt_image = viewpoint_cam.original_image.cuda()
        #         torchvision.utils.save_image(image, os.path.join(static_render, view.image_name + ".png"))
        #         torchvision.utils.save_image(view.original_image, os.path.join(static_gt, view.image_name + ".png"))

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
            if iteration <= 5000:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration >= 500 and iteration % 100 == 0:
                    if iteration>=3000:
                        size_threshold = opt.size_threshold if iteration > opt.opacity_reset_interval else None
                        opacity_threshold = 0.005 if iteration == 5000 else 0.005
                        gaussians.densify_and_prune(0.0002, opacity_threshold, 1000 , # scene.cameras_extent,
                                                    size_threshold)
                    else:
                        size_threshold = opt.size_threshold if iteration > opt.opacity_reset_interval else None
                        opacity_threshold = 0.005 if iteration == 5000 else 0.005
                        gaussians.densify_and_prune(0.0002, opacity_threshold, 1000,
                                                    size_threshold)
                    # gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())
                
                    
                if (iteration % 3000 == 0 and iteration > 0) or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.update_learning_rate(iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)

    scene.save(iteration)
    return 
def training(dataset, opt, pipe, testing_iterations, saving_iterations, n_keys, vel_start_time=0.0):
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree)
    # deform = DeformModel_incremental(dataset.is_blender, dataset.is_6dof, max_incremental_time=dataset.max_time, max_warm_up_time=args.incremental_warmup_time , vel_start_time=vel_start_time)
    # deform.train_setting(opt)

    # scene = Scene(dataset, gaussians)
    # gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians = GaussianModel(dataset.sh_degree)
    kwargs = {'dt':args.dt ,  'cn_init': args.cn_init , 
              'with_node_weight': args.with_node_weight , 'cn_KNN_method': args.cn_KNN_method , 'cn_KNN': args.cn_KNN ,
              'cn_hyperdim' : args.cn_hyperdim ,'zero_padding':args.zero_padding }
    deform = deform = DeformModel_incremental_net(control_num= dataset.control_num , **kwargs)
    deform.vel_train_setting(opt)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians )
    gaussians.training_setup(opt)

    
    static_start= time.time()
    if args.DynamicGS_dataset:
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
    '''Start Incremental Training!'''

    

    

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
    # xyz = gaussians.get_xyz.clone()
    warmup_stetps = args.warmup_steps
    d_xyz_n = torch.zeros_like(deform.control_node_warp.nodes).cuda()
    d_rot_n= torch.zeros(control_num,4).cuda()
    d_rot_n[...,0] = 1 
    d_scaling_n = torch.zeros(control_num,3).cuda()

    k = args.integrate_ksteps
    d_xyz = torch.zeros_like(gaussians.get_xyz).cuda()
    d_rot= torch.zeros_like(gaussians.get_rotation).cuda()
    d_rot[...,0] = 1 
    d_scaling = torch.zeros_like(gaussians.get_scaling).cuda()
    use_control_node = False
    weights_opt_count = 0 # incremental phase
    weights_opt_time = 0 # incremental phase
    gs_opt_count = 0 # per frame
    gs_opt_time = 0
    gs_opt_iter = args.iter_gs
    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
    psnr_dict = {'train':[] , 'val':[] , 'trainingview_test':[] , 'novelview_test': []}
    lpips_dict = {'train':[] , 'val':[] , 'trainingview_test':[], 'novelview_test': []}
    ssim_dict = {'train':[] , 'val':[] , 'trainingview_test':[], 'novelview_test': []}
    print('Incremental Training Starts!')
    start1 = time.time()
    '''遍历所有时刻'''
    for idx, view_group in enumerate(tqdm(incremental_views_group, desc="Rendering progress")):
        if idx == 0 :
            continue
        if idx >= len(incremental_val_views_group) - args.predict_steps:
            break

        if view_group[0].fid.item() <= args.incremental_warmup_time :
            # k_views_group = incremental_views_group[idx : idx + k]  
            if view_group[0].fid.item() < args.incremental_warmup_time * (1/2): # 前一半先不用control node
                # 获取接下来 k 个时刻view_group，初始化 render list，存放 k+1 个时刻的渲染图片
                # warmup_iter_basis = args.warmup_iter_basis if idx>1 else 50 # very very very wierd bug
                k_views_group = incremental_views_group[idx : idx + k] 
                for i in range(25):
                        # 随机抽取一个视角
                    v = random.randint(0,len(view_group)-1)
                    views = [group[v] for group  in k_views_group] # 同个视角 不同时刻  

                    deform_code = deform.code_field(gaussians.get_xyz.detach())
                    deform_seg = deform.code_field.seg(deform_code)
                    
                    view_first = views[0]
                    view_last = views[-1]
                    view_first.load2device()
                    view_last.load2device()
                    gt_image_first = view_first.original_image.cuda()
                    gt_image_last = view_last.original_image.cuda()

                    d_xyz_incre_first, d_rot_incre_first = deform.incremental_step(gaussians.get_xyz.detach() + d_xyz.detach(),time_emb=view_first.fid.expand(gaussians.get_xyz.shape[0], 1),deform_seg=deform_seg,dt=args.dt)
                                
                    d_xyz_total_first = d_xyz.detach() + d_xyz_incre_first 
                    d_rot_total_first = quaternion_multiply(d_rot_incre_first, d_rot.detach()) 

                    render_pkg_re_first = render(view_first, gaussians, pipe, background, d_xyz_total_first, d_rot_total_first, d_scaling, dataset.is_6dof)
                    image_first= render_pkg_re_first["render"]
                    Ll1_first = l1_loss(image_first, gt_image_first) 
                    loss = (1.0 - opt.lambda_dssim) * Ll1_first + opt.lambda_dssim * (1.0 - ssim(image_first, gt_image_first)) 
                    loss.backward()
                    deform.vel_optimizer.step()
                    deform.vel_optimizer.zero_grad()
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    del render_pkg_re_first

                    deform_code = deform.code_field(gaussians.get_xyz.detach())
                    deform_seg = deform.code_field.seg(deform_code)
                    d_xyz_incre_last, d_rot_incre_last = deform.incremental_step(gaussians.get_xyz.detach() + d_xyz.detach(),time_emb=view_last.fid.expand(gaussians.get_xyz.shape[0], 1),deform_seg=deform_seg,dt=k/88)
                    d_xyz_total_last = d_xyz.detach() + d_xyz_incre_last 
                    d_rot_total_last = quaternion_multiply(d_rot_incre_last, d_rot.detach())
                    render_pkg_re_last = render(view_last, gaussians, pipe, background, d_xyz_total_last, d_rot_total_last, d_scaling, dataset.is_6dof)
                    image_last= render_pkg_re_last["render"]
                    Ll1_last  = l1_loss(image_last, gt_image_last) 
                    loss = (1.0 - opt.lambda_dssim) * Ll1_last + opt.lambda_dssim * (1.0 - ssim(image_last, gt_image_last)) 
                    loss.backward()
                    deform.vel_optimizer.step()
                    deform.vel_optimizer.zero_grad()
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    del render_pkg_re_last
                    print(loss.item())
                    d_xyz_total = d_xyz_total_first
                    d_rot_total = d_rot_total_first
                    view_first.load2device('cpu')
                    view_last.load2device('cpu')
            else:
                use_control_node = True
                size_threshold = opt.size_threshold
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                gaussian_num_dict[idx] = gaussians.get_xyz.shape[0]
                for i in range(args.warmup_iter_basis):
                    # 随机抽取一个视角
                
                    # 计算segcode
                    nodes = deform.control_node_warp.nodes
                    deform_code = deform.code_field(gaussians.get_xyz.detach())
                    deform_seg = deform.code_field.seg(deform_code)
                    deform_code_node = deform.code_field(nodes.detach())
                    deform_seg_node = deform.code_field.seg(deform_code_node)


                    # 计算control node 权重矩阵
                    n_weight, _, nn_idxs = deform.control_node_warp.cal_nn_weight(gaussians.get_xyz.detach(),deform_seg,deform_seg_node)

                    view_first = random.choice(view_group)
                    view_first.load2device()
                    
                    gt_image_first = view_first.original_image.cuda()
                    
                    d_xyz_incre_n_first, d_rot_incre_n_first = deform.incremental_step(nodes.detach() + d_xyz_n.detach(),view_first.fid.expand(nodes.shape[0], 1) , deform_seg_node, dt = args.dt)
              
                    d_xyz_total_n_first = d_xyz_n.detach() + d_xyz_incre_n_first 
                    d_rot_total_n_first = quaternion_multiply(d_rot_incre_n_first, d_rot_n.detach())

                    d_xyz_total_first, d_rot_total_first , d_scaling = deform.control_node_warp.cal_deform(n_weight.detach(),nn_idxs.detach(), d_xyz_total_n_first, d_rot_total_n_first, d_scaling_n ,  method = args.cn_interpolate_method)

                    render_pkg_re_first = render(view_first, gaussians, pipe, background, d_xyz_total_first, d_rot_total_first, d_scaling, dataset.is_6dof)
                    image_first= render_pkg_re_first["render"]
                    
                    Ll1_first = l1_loss(image_first, gt_image_first)
                    loss = (1.0 - opt.lambda_dssim) * Ll1_first + opt.lambda_dssim * (1.0 - ssim(image_first, gt_image_first)) 
         
                    
                    # print(rgbloss.item(), floss.item())
                    print(loss.item())
                    loss.backward()
                    deform.vel_optimizer.step() # vel weights
                    deform.vel_optimizer.zero_grad()
                    deform.control_node_warp.optimizer.step() # control node
                    deform.control_node_warp.optimizer.zero_grad()
                    deform.optimizer.step()
                    deform.optimizer.zero_grad()
                    # 存储以下值是为了传递到下一个时刻，表示node 和 gs 最新优化的前进一步的结果
                    d_xyz_total_n = d_xyz_total_n_first
                    d_rot_total_n = d_rot_total_n_first
                    d_xyz_total = d_xyz_total_first
                    d_rot_total = d_rot_total_first      
                
                    view_first.load2device('cpu')
                    # view_last.load2device('cpu')               
            
        else: 
            weights_opt_count += 1
            weights_opt_start = time.time()
            for i in range(args.iter_basis):# 进入优化循环
                # 从view_group随机抽取一个相机视角
                view = random.choice(view_group)
                view.load2device()
                

                # TODO: 
                nodes = deform.control_node_warp.nodes
                deform_code_node = deform.code_field(nodes.detach())
                deform_seg_node = deform.code_field.seg(deform_code_node)

                # d_scaling_incre = torch.zeros_like(gaussians.get_scaling)
                d_xyz_incre_n, d_rot_incre_n = deform.incremental_step(nodes.detach() + d_xyz_n.detach(),view.fid.expand(nodes.shape[0], 1) , deform_seg_node, dt = args.dt)
                
                d_xyz_total_n = d_xyz_n.detach() + d_xyz_incre_n
                d_rot_total_n = quaternion_multiply(d_rot_incre_n, d_rot_n.detach())
                d_xyz_total, d_rot_total , d_scaling = deform.control_node_warp.cal_deform(n_weight.detach(),nn_idxs.detach(), d_xyz_total_n, d_rot_total_n, d_scaling_n , method = args.cn_interpolate_method)
                # Render
                
                render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total, d_rot_total, d_scaling, dataset.is_6dof)
                image= render_pkg_re["render"]

                # 计算loss
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
                view.load2device('cpu')
                # can include gs into optimization
            weights_opt_end = time.time()
            weights_opt_time += (weights_opt_end-weights_opt_start)
        
        '''额外对gaussian 优化'''
        if view_group[0].fid.item() <= args.incremental_warmup_time :
            gs_opt_iter = args.warmup_iter_gs
        else:
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

            # backward
            loss.backward()
            # update
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            # deform.optimizer.zero_grad()
        gs_opt_end= time.time()
        gs_opt_time += (gs_opt_end - gs_opt_start)
        '''end'''
        if use_control_node:
            d_xyz_n = d_xyz_total_n
            d_rot_n = d_rot_total_n
        d_xyz = d_xyz_total
        d_rot = d_rot_total
        
        if args.skip_render or view.fid.item() <= args.incremental_warmup_time:
            continue
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
                view.load2device()
                render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total, d_rot_total, d_scaling, dataset.is_6dof)
                image = render_pkg_re["render"]
                if not args.skip_render:
                    torchvision.utils.save_image(image, os.path.join(train_render, view.image_name + f"_{index}.jpg"))
                    torchvision.utils.save_image(view.original_image, os.path.join(train_gt, view.image_name + f"_{index}.jpg"))
                if view.mask_path:
                    # shutil.copy(view.mask_path, os.path.join(train_mask, view.image_name+".npy"))
                    if 'npy' in view.mask_path:
                        mask = np.load(view.mask_path)
                    elif 'npz' in view.mask_path:
                        mask = np.load(view.mask_path)['mask']
                    mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                else:
                    # 创建一个和 image 相同 H W 的内容全部为 1 的 mask
                    mask = np.ones((view.original_image.shape[1], view.original_image.shape[2]))
                    mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                
                # if not args.skip_render:
                #     if view.mask_path:
                #         mask_path = os.path.join(train_mask, view.image_name+".npz")
                #         np.savez_compressed(mask_path, mask = mask.detach().cpu().numpy())

                render_masked = (image * mask).float()
                gt_masked =  (view.original_image* mask).float()

                # 根据 depth，只算近处物体的遮罩
                if args.max_depth > 0:
                    depth = view.depth
                    depth_mask = depth < args.max_depth
                    depth_mask = depth_mask.unsqueeze(-1)  # [H, W, 1]，用于广播
                    render_masked = render_masked * depth_mask
                    gt_masked = gt_masked * depth_mask

                    
                psnr_list_tmp.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                ssim_list_tmp.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                lpips_list_tmp.append(lpips_fn(render_masked, gt_masked).item())
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
                view.load2device()
                render_pkg_re = render(view, gaussians, pipe, background, d_xyz, d_rot, d_scaling, dataset.is_6dof)
                image = render_pkg_re["render"]
                if not args.skip_render:
                    torchvision.utils.save_image(image, os.path.join(val_render, view.image_name + f"_{index}.jpg"))
                    torchvision.utils.save_image(view.original_image, os.path.join(val_gt, view.image_name + f"_{index}.jpg"))

                if view.mask_path:
                    if 'npy' in view.mask_path:
                        mask = np.load(view.mask_path)
                    elif 'npz' in view.mask_path:
                        mask = np.load(view.mask_path)['mask']
                    mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                    
                else:
                    # 创建一个和 image 相同 H W 的内容全部为 1 的 mask
                    mask = np.ones((view.original_image.shape[1], view.original_image.shape[2]))
                    mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                    # 将 mask 保存到路径 os.path.join(train_mask, "{0:04}.npy".format(count))

                if not args.skip_render:
                    if view.mask_path:
                        mask_path = os.path.join(val_mask, view.image_name+".npz")
                        np.savez_compressed(mask_path, mask = mask.detach().cpu().numpy())

                render_masked = (image * mask).float()
                gt_masked =  (view.original_image* mask).float()
                psnr_list_tmp.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                ssim_list_tmp.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                lpips_list_tmp.append(lpips_fn(render_masked, gt_masked).item())
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
            for j in range(idx + 1, idx1):
                '''基于现在的weights 以及 xyz 向前一步'''
                fid_tmp = incremental_views_group[j][0].fid.cuda()
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
                    if not args.skip_render and fid_tmp > 0.8:
                        torchvision.utils.save_image(image, os.path.join(test_render, "{0:04}.png".format(count)))
                        torchvision.utils.save_image(view.original_image, os.path.join(test_gt,"{0:04}.png".format(count)))

                    if view.mask_path:
                        # shutil.copy(view.mask_path, os.path.join(test_mask,"{0:04}.npy".format(count)))
                        if 'npy' in view.mask_path:
                            mask = np.load(view.mask_path)
                        elif 'npz' in view.mask_path:
                            mask = np.load(view.mask_path)['mask']
                        mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                        
                    else:
                        # 创建一个和 image 相同 H W 的内容全部为 1 的 mask
                        mask = np.ones((view.original_image.shape[1], view.original_image.shape[2]))
                        mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                        
                    # if not args.skip_render:
                    #     if view.mask_path:
                    #         mask_path = os.path.join(test_mask, "{0:04}.npz".format(count))
                    #         np.savez_compressed(mask_path, mask = mask.detach().cpu().numpy())

                    render_masked = (image * mask).float()
                    gt_masked =  (view.original_image* mask).float()
                    psnr_list_tmp_training_view.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    ssim_list_tmp_training_view.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    lpips_list_tmp_training_view.append(lpips_fn(render_masked, gt_masked).item())
                    count += 1
                    view.load2device('cpu')

                for view in incremental_val_views_group[j]:
                    view.load2device()
                    render_pkg_re = render(view, gaussians, pipe, background, d_xyz_total_tmp, d_rot_total_tmp, d_scaling, dataset.is_6dof)
                    image = render_pkg_re["render"]
                    if not args.skip_render and fid_tmp > 0.8:
                        torchvision.utils.save_image(image, os.path.join(test_render,"{0:04}.png".format(count)))
                        torchvision.utils.save_image(view.original_image, os.path.join(test_gt,"{0:04}.png".format(count)))
                    if view.mask_path:
                        # shutil.copy(view.mask_path, os.path.join(test_mask,"{0:04}.npy".format(count)))
                        if 'npy' in view.mask_path:
                            mask = np.load(view.mask_path)
                        elif 'npz' in view.mask_path:
                            mask = np.load(view.mask_path)['mask']
                        mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                    else:
                        # 创建一个和 image 相同 H W 的内容全部为 1 的 mask
                        mask = np.ones((view.original_image.shape[1], view.original_image.shape[2]))
                        mask = torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1).cuda()
                    
                    # if not args.skip_render:
                    #     mask_path = os.path.join(test_mask, "{0:04}.npz".format(count))
                    #     np.savez_compressed(mask_path, mask = mask.detach().cpu().numpy())

                    render_masked = (image * mask).float()
                    gt_masked =  (view.original_image* mask).float()
                    psnr_list_tmp_novel_view.append(psnr(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    ssim_list_tmp_novel_view.append(ssim(render_masked.unsqueeze(0), gt_masked.unsqueeze(0)).item())
                    lpips_list_tmp_novel_view.append(lpips_fn(render_masked, gt_masked).item())
                    count += 1
                    view.load2device('cpu')

                d_xyz_n_tmp = d_xyz_n_tmp + d_xyz_incre_n_tmp
                d_xyz_tmp = d_xyz_total_tmp
                d_rot_tmp = d_rot_total_tmp
                
                pass
            
            mean_psnr_trainingview = torch.tensor(psnr_list_tmp_training_view).mean().item()
            mean_ssim_trainingview = torch.tensor(ssim_list_tmp_training_view).mean().item()
            mean_lpips_trainingview = torch.tensor(lpips_list_tmp_training_view).mean().item()
            psnr_dict['trainingview_test'].append(mean_psnr_trainingview)
            ssim_dict['trainingview_test'].append(mean_ssim_trainingview)
            lpips_dict['trainingview_test'].append(mean_lpips_trainingview)

            mean_psnr_novelview = torch.tensor(psnr_list_tmp_novel_view).mean().item()
            mean_ssim_novelview = torch.tensor(ssim_list_tmp_novel_view).mean().item()
            mean_lpips_novelview = torch.tensor(lpips_list_tmp_novel_view).mean().item()
            psnr_dict['novelview_test'].append(mean_psnr_novelview)
            ssim_dict['novelview_test'].append(mean_ssim_novelview)
            lpips_dict['novelview_test'].append(mean_lpips_novelview)

            
        
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
    psnr_dict['novelview_test_mean'] = np.mean(psnr_dict['novelview_test'])
    ssim_dict['novelview_test_mean'] = np.mean(ssim_dict['novelview_test'])
    lpips_dict['novelview_test_mean'] = np.mean(lpips_dict['novelview_test'])
    with open(os.path.join(render_path,"psnr.json"), 'w') as file:
        json.dump(psnr_dict, file)
    with open(os.path.join(render_path,"ssim.json"), 'w') as file:
        json.dump(ssim_dict, file)
    with open(os.path.join(render_path,"lpips.json"), 'w') as file:
        json.dump(lpips_dict, file)
    print('Incremental Training Completes!')
    
    return


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


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # images = torch.tensor([], device="cuda")
                # gts = torch.tensor([], device="cuda")
                images = []
                gts = []
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    # deform_code = scene.gaussians.get_deform_code
                    deform_code = deform.code_field(xyz)
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0).cpu()
                    gt_image = torch.clamp(viewpoint.original_image.cpu(), 0.0, 1.0)
                    # images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    # gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
                    images.append(image)
                    gts.append(gt_image)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                images = torch.stack(images, dim=0)
                gts = torch.stack(gts, dim=0)
                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr

def reassign_train_val_test(train_views, val_views, test_views, train_max_time=0.9):
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
    for i, view in enumerate(test_views):
        if view.fid > train_max_time:
            new_test_views.append(view)
        else:
            if (3 * 22 <= i < 4 * 22) or (8 * 22 <= i < 9 * 22) or (15 * 22 <= i < 16 * 22):
                new_val_views.append(view)
            else:
                new_train_views.append(view)

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
    parser.add_argument('--incremental_max_time', type=float, default=0.75) # max training timestamps
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
