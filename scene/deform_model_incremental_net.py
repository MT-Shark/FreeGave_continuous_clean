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
class DeformModel_incremental_net:
    def __init__(self, control_num = 10000 , **kwargs):
        self.d_gate = False
        self.v_gate = False # ~self.d_gate
        deform_code_dim = 16 # 16
        self.code_field = CodeField(D=4, W=128, input_ch=3, output_ch=deform_code_dim, multires=8).cuda()
        self.vel_net = SegVel(deform_code_dim=deform_code_dim, hidden_dim=128, layers=5, zero_padding= kwargs['zero_padding']).cuda()
        self.vel = VelocityWarpper(self.vel_net)
        self.K = 16
        if control_num > 1 :
            self.control_node_warp = ControlNodeWarp(node_num=control_num , K=kwargs['cn_KNN'] , skinning=False, hyper_dim=kwargs['cn_hyperdim'] ,**kwargs).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

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
    def vel_train_setting(self, training_args):
        l = [
            {'params': list(self.vel.parameters()),
             'lr': training_args.position_lr_init,
             "name": "vel"}
        ]
        self.vel_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform_incremental/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.code_field.state_dict(), os.path.join(out_weights_path, 'code_field.pth'))
        #torch.save({'weights_list':self.weights_list},os.path.join(out_weights_path, 'weights_list.pth'))


    
    def load_weights(self, model_path, iteration=-1):
        ''' Load the weights of the warm up model '''
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform_incremental"))
        else:
            loaded_iter = iteration
        
        code_field_weights_path = os.path.join(model_path, "deform_incremental/iteration_{}/code_field.pth".format(loaded_iter))
        self.code_field.load_state_dict(torch.load(code_field_weights_path))
        
    
    
    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr


    def u_func(self, deform_code,  xyzt):
        u = self.vel_net.get_vel(deform_code, xyzt)
        return u , u
    
    def incremental_step(self,xyz, time_emb, deform_seg, dt=1/88):
        '''
        xyz: current position
        time_emb: current time
        deform_code: 
        dt: 
        '''
        rot_prev = torch.zeros(xyz.shape[0], 3, 3, dtype=xyz.dtype, device=xyz.device)
        rot_prev[:, 0, 0] = 1
        rot_prev[:, 1, 1] = 1
        rot_prev[:, 2, 2] = 1
        identity = torch.eye(3, dtype=xyz.dtype, device=xyz.device).unsqueeze(0).expand(xyz.shape[0], 3, 3)
        
        # get time step
        # Runge-Kutta 2
        # 获得当前时刻每个点的 vel 
        xyzt = torch.cat([xyz, time_emb], dim=1)
        v_cur = self.vel_net.get_vel(deform_seg , xyzt)
        p_mid = xyz.detach() + 0.5 * dt * v_cur

        # 获得中间时刻的 vel
        xyzt_mid = torch.cat([p_mid.detach(), time_emb + 0.5 * dt], dim=1)
        N = p_mid.shape[0]

        # v_mid = self.vel_net.get_vel(deform_seg , xyzt_mid)
        # with torch.no_grad():
        #     jac_v , _ = vmap(jacrev(self.u_func,argnums = -1 , has_aux = True))(deform_seg, xyzt_mid )
        # jac_v , v_mid = vmap(jacrev(self.u_func,argnums = -1 , has_aux = True))(deform_seg, xyzt_mid )
        v_mid, jac_v = self.vel_net.get_vel_jac(deform_seg , xyzt_mid)

        d_xyz = dt * v_mid 
        drot = identity + dt * jac_v[..., :3, :3]
        rot_prev = torch.bmm(drot, rot_prev)
        rot_prev = rotation_to_quaternion(rot_prev)
        
        return d_xyz, rot_prev    
    
    


        
    
    
        


    
