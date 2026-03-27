import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import DeformNetwork, CodeField , ControlNodeWarp
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func, quaternion_multiply
from utils.velocity_field_utils import VelBasis, VelocityWarpper, AccBasis, SegVel
import einops
import numpy as np
from functorch import vmap, jacrev
from utils.general_utils import rotation_to_quaternion
epsilon = 1e-6
class DeformModel_incremental:
    def __init__(self, control_num = 10000 , **kwargs):
        self.d_gate = False
        self.v_gate = False # ~self.d_gate
        deform_code_dim = 16 # 16
        self.code_field = CodeField(D=4, W=128, input_ch=3, output_ch=deform_code_dim, multires=8).cuda()
        # self.deform = DeformNetwork(D=8, W=256, input_ch=3, hyper_ch=deform_code_dim, multires=8,
        #                             is_blender=is_blender, is_6dof=is_6dof, gated=self.d_gate).cuda()
        # self.deform = DeformNetwork(D=6, W=128, input_ch=3, hyper_ch=deform_code_dim, multires=8,
        #                             is_blender=is_blender, is_6dof=is_6dof, gated=self.d_gate).cuda()
        # self.vel_net = VelBasis(deform_code_dim=deform_code_dim).cuda()
        # self.vel_net = SegVel(deform_code_dim=deform_code_dim, hidden_dim=128, layers=5).cuda()
        self.K = 15
        if control_num > 1 :
            self.control_node_warp = ControlNodeWarp(node_num=control_num , K=kwargs['cn_KNN'] , skinning=False, hyper_dim=kwargs['cn_hyperdim'] ,**kwargs).cuda()
        # self.vel_net = VelMLP().cuda()
        # self.vel = VelocityWarpper(self.vel_net)
        # self.acc = AccBasis(deform_code_dim).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        # self.max_incremental_time = max_incremental_time
        # self.max_warm_up_time = max_warm_up_time
        # self.vel_start_time = vel_start_time
        # self.deform_max_time = deform_max_time

        self.weights_list = []
        self.weights_timestamps = []

        self.d_xyz_list = []
        self.d_rot_list = []
        self.d_scale_list = []
        self.deformation_timestamps = []

        self.static_mask = None
        self.motion_mask = None

    

    

    def train_setting(self, training_args):
        l = [
            {'params': list(self.code_field.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0001, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform_incremental/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.code_field.state_dict(), os.path.join(out_weights_path, 'code_field.pth'))
        #torch.save({'weights_list':self.weights_list},os.path.join(out_weights_path, 'weights_list.pth'))

    def save_incremental_weights(self, model_path, iteration):
        if iteration == -1:
            iteration = searchForMaxIteration(os.path.join(model_path, "deform_incremental"))
        out_weights_path = os.path.join(model_path, "deform_incremental/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save({'incremental_weights_list':self.weights_list , 'incremental_weights_timestamps': self.weights_timestamps},
                   os.path.join(out_weights_path, 'incremental_weights_list.pth'))

    
    def load_weights(self, model_path, iteration=-1):
        ''' Load the weights of the warm up model '''
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_incremental"))
        else:
            loaded_iter = iteration
        
        code_field_weights_path = os.path.join(model_path, "deform_incremental/iteration_{}/code_field.pth".format(loaded_iter))
        self.code_field.load_state_dict(torch.load(code_field_weights_path))
        
        
    def load_incremental_weights(self, model_path, iteration=-1):
        ''' Load the weights of the incremental model '''
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_incremental"))
        else:
            loaded_iter = iteration
        
        incremental_weights_path = os.path.join(model_path, "deform_incremental/iteration_{}/incremental_weights_list.pth".format(loaded_iter))
        state = torch.load(incremental_weights_path)
        self.weights_list = state['incremental_weights_list']
        self.weights_timestamps = state['incremental_weights_timestamps']
    
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def increment_init(self, t):
        # t_embed = self.vel_net.embedder(t)
        # weights = self.vel_net.weight_net(t_embed)
        # weights = einops.rearrange(weights, '... (K dim) -> ... K dim', K=self.vel_net.K)
        # TODO: 
        weights = torch.zeros(self.K +1 , 6 ,device='cuda').cuda()
        
        self.weights_list = [weights]
        self.weights_timestamps = [t.item()]
        self.cur_weights = weights
        return
    
    
    
    def update_weights(self , timestamp = None):
        zero_weights = torch.zeros(1,6,device='cuda') 
        d_weights = torch.cat([self.d_weights , zero_weights],dim=0)
        self.cur_weights = self.cur_weights + d_weights
        self.weights_list.append(self.cur_weights)
        if timestamp != None:
            self.weights_timestamps.append(timestamp)
        return 
    
   

    def optimize_init(self, training_args, init_weights=None , cur_weights = None):
        if init_weights is not None:
            # 使用提供的初始化权重
            self.d_weights = nn.Parameter(init_weights.to('cuda'))
            
        else:
            # 使用随机初始化权重
            self.d_weights = nn.Parameter(torch.randn(self.K, 6, device='cuda') / np.sqrt(self.K))
            
        
        l = [
            {'params': self.d_weights,
            'lr': training_args.d_weights_lr,
            "name": "d_basis"}
        ]
        self.weights_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        if cur_weights is None:
            self.cur_weights = self.weights_list[-1]
        else:
            self.cur_weights = cur_weights
        return
    

    def u_func(self, deform_seg, weights , xyz):
        # x, y, z = xt[..., 0].clamp(-1., 1.), xt[..., 1].clamp(-1., 1.), xt[..., 2].clamp(-1., 1.)
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        zeros = xyz[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        basis = torch.stack([b1, b2, b3, b4, b5, b6], dim=-2)
        u = torch.einsum('...ij,...ki->...kj', basis, weights)
        u = torch.einsum('...k,...kj->...j', deform_seg, u)
        return u , u
    

    def incremental_step(self, xyz, deform_seg , cur_weights = None, next_weights = None , dt=1/88 ): # 实际上next_weights 表示的是2倍的average weights (due to the implementation coincidence, but it works)
        ''' integrate one time slot'''
        if cur_weights == None and next_weights == None: # need to learn weights
            cur_weights = self.cur_weights.detach()
            zero_weights = torch.zeros(1,6,device='cuda') 
            d_weights = torch.cat([self.d_weights , zero_weights],dim=0)
            mid_weights = (cur_weights + d_weights)/2 # not intuitive, but it works (too late to fix it...)
        else: 
            mid_weights = next_weights /2 # directly use learned weights

        # mid_weights = average_weights
        rot_prev = torch.zeros(xyz.shape[0], 3, 3, dtype=xyz.dtype, device=xyz.device)
        rot_prev[:, 0, 0] = 1
        rot_prev[:, 1, 1] = 1
        rot_prev[:, 2, 2] = 1
        identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device).unsqueeze(0).expand(xyz.shape[0], 3, 3)
        
        # get time step
        # Runge-Kutta 2

        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        zeros = xyz[..., -1] * 0.
        ones = zeros + 1.

        b1 = torch.stack([ones, zeros, zeros], dim=-1)
        b2 = torch.stack([zeros, ones, zeros], dim=-1)
        b3 = torch.stack([zeros, zeros, ones], dim=-1)
        b4 = torch.stack([zeros, z, -y], dim=-1)
        b5 = torch.stack([-z, zeros, x], dim=-1)
        b6 = torch.stack([y, -x, zeros], dim=-1)

        Bmatrix = torch.stack([b1, b2, b3, b4, b5, b6], dim=-2)

        # cur_weights = einops.rearrange(cur_weights, '... (K dim) -> ... K dim', K=self.K)
        v_cur = torch.einsum('...ij,...ki->...kj', Bmatrix, cur_weights)
        v_cur = torch.einsum('...k,...kj->...j', deform_seg.detach(), v_cur)
        p_mid = xyz.detach() + 0.5 * dt * v_cur

        N = p_mid.shape[0]
        jac_v , v_mid = vmap(jacrev(self.u_func,argnums = -1 , has_aux = True))(deform_seg, mid_weights.unsqueeze(0).expand(N, -1, -1), p_mid )
        d_xyz = dt * v_mid
        drot = identity + dt * jac_v[..., :3, :3]
        rot_prev = torch.bmm(drot, rot_prev)
        rot_prev = rotation_to_quaternion(rot_prev)
        
        return d_xyz, rot_prev
        
    
    def incremental_integrate(self, pos_init, deform_seg, t1, t2, dt=1/88):
        ''' integrate from t1 to t2'''
        '''
        t1: float
        t2: tensor of shape (N, 1)
        '''
        # 从 self.weights_timestamps 中找到相同的时间戳，根据list的index，找到对应的weights。如果没有相同的时间戳，则报错
        # if t1 not in self.weights_timestamps:
        #     raise ValueError(f"Timestamp {t1} not found in weights_timestamps")
        #t1 = t1.unsqueeze(0).expand(t2.shape[0],-1)
        xyz_prev = pos_init
        t_curr = t1
        unfinished = (t2 > t_curr).squeeze(-1)
        if t1 in self.weights_timestamps:
            cur_weights = self.weights_list[self.weights_timestamps.index(t1)]
        elif t1 > self.weights_timestamps[-1]:
            cur_weights = self.weights_list[-1]
        else:
            raise ValueError(f"Timestamp {t1} not found in weights_timestamps")
        rot_prev = torch.zeros(pos_init.shape[0], 4, dtype=pos_init.dtype, device=pos_init.device)
        rot_prev[:, 0] = 1
        incremental_max_time = self.weights_timestamps[-1]
        while unfinished.any():
            if any(ts > t_curr for ts in self.weights_timestamps):
                next_timestamp = min([ts for ts in self.weights_timestamps if ts > t_curr])
                next_weights = self.weights_list[self.weights_timestamps.index(next_timestamp)]
                if t2[0, 0].item() - epsilon > next_timestamp:
                    pass
                else:
                    next_timestamp = t2[0, 0].item()
                     
            else:
                if t_curr + dt < t2[0, 0].item() - epsilon:
                    next_timestamp = t_curr + dt
                    next_weights = cur_weights
                else:
                    next_timestamp = t2[0, 0].item()
                    next_weights = cur_weights
                
            dt_cur = next_timestamp - t_curr 
            d_xyz, d_rot = self.incremental_step(xyz_prev[unfinished], deform_seg[unfinished], cur_weights, next_weights, dt=dt_cur)
            xyz_cur = xyz_prev[unfinished] + d_xyz
            xyz_prev[unfinished] = xyz_cur
            rot_prev[unfinished] = quaternion_multiply(d_rot, rot_prev[unfinished])
            t_curr = next_timestamp
            cur_weights = next_weights
            unfinished = (t2 > t_curr).squeeze(-1)
        return xyz_prev, rot_prev
        


    
