import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3
from utils.general_utils import build_rotation,rotation_to_quaternion
from functorch import vmap, jacrev, hessian
import einops
import pytorch3d.ops
from utils.dqb_utils import  Rt2dq, dq2unitdq, dq2Rt
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R
def __compute_delta_Rt_ji__(R_wi, t_wi, R_wj, t_wj):
    # R: N,K,3,3; t: N,K,3
    # the stored node R,t are R_wi, t_wi
    # p_t=i_world = R_wi @ p_local + t_wi
    # p_local = R_wi.T @ (p_t=i_world - t_wi)
    # p_t=j_world = R_wj @ p_local + t_wj
    # p_t=j_world = R_wj @ R_wi.T @ (p_t=i_world - t_wi) + t_wj
    # p_t=j_world = (R_wj @ R_wi.T) @ p_t=i_world + t_wj - (R_wj @ R_wi.T) @ t_wi
    assert R_wi.ndim == 4 and R_wi.shape[2:] == (3, 3)
    assert t_wi.ndim == 3 and t_wi.shape[2] == 3
    assert R_wj.ndim == 4 and R_wj.shape[2:] == (3, 3)
    assert t_wj.ndim == 3 and t_wj.shape[2] == 3

    R_ji = torch.einsum("nsij,nskj->nsik", R_wj, R_wi)
    t_ji = t_wj - torch.einsum("nsij,nsj->nsi", R_ji, t_wi)
    return R_ji, t_ji


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class DeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, hyper_ch=8, multires=10, is_blender=False, is_6dof=False, gated=True):
        super(DeformNetwork, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.hyper_ch = hyper_ch
        self.t_multires = 6 if is_blender else 10
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(self.t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(multires, input_ch)
        self.input_ch = xyz_input_ch + time_input_ch + hyper_ch

        # Better for D-NeRF Dataset
        self.time_out = 30

        self.timenet = nn.Sequential(
            nn.Linear(time_input_ch, W), nn.ReLU(inplace=True),
            nn.Linear(W, self.time_out))

        self.linear = nn.ModuleList(
            [nn.Linear(xyz_input_ch + self.time_out + hyper_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch + self.time_out + hyper_ch, W)
                for i in range(D - 1)]
        )

        self.is_blender = is_blender
        self.is_6dof = is_6dof

        if is_6dof:
            self.branch_w = nn.Linear(W, 3)
            self.branch_v = nn.Linear(W, 3)
        else:
            self.gaussian_warp = nn.Linear(W, 3)
        self.gaussian_rotation = nn.Linear(W, 4)
        self.gaussian_scaling = nn.Linear(W, 3)

        self.gated = gated
        # self.gate_func = lambda t: torch.tanh(20 * t ** 2)
        self.gate_func = nn.Sequential(
            nn.Linear(hyper_ch, W), nn.ReLU(inplace=False),
            nn.Linear(W, 1), nn.Sigmoid()
        )

    def get_feature(self, x, t, motion_code):
        t_emb = self.embed_time_fn(t)
        if self.is_blender:
            t_emb = self.timenet(t_emb)  # better for D-NeRF Dataset
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, motion_code], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, motion_code, h], -1)
        if self.gated:
            gate = self.gate_func(motion_code)
        else:
            gate = None
        return h, gate

    def get_gate(self, motion_code):
        gate = self.gate_func(motion_code)
        return gate

    def get_translation(self, x, t, motion_code):
        h, gate = self.get_feature(x, t, motion_code)
        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)

        if self.gated:
            # gate = self.gate_func(t)
            d_xyz = d_xyz * gate
        return d_xyz

    def forward(self, x, t, motion_code):
        h, gate = self.get_feature(x, t, motion_code)

        if self.is_6dof:
            w = self.branch_w(h)
            v = self.branch_v(h)
            theta = torch.norm(w, dim=-1, keepdim=True)
            w = w / theta + 1e-5
            v = v / theta + 1e-5
            screw_axis = torch.cat([w, v], dim=-1)
            d_xyz = exp_se3(screw_axis, theta)
        else:
            d_xyz = self.gaussian_warp(h)

        scaling = self.gaussian_scaling(h)
        # scaling = 0.0
        rotation = self.gaussian_rotation(h)

        if self.gated:
            # gate = self.gate_func(t)
            d_xyz = d_xyz * gate
            # rotation[..., 1:] = rotation[..., 1:] * gate
            rotation = rotation * gate
            rotation[..., 1:] = rotation[..., 1:] + 1

            scaling = scaling * gate

        rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)

        return d_xyz, rotation, scaling

    def get_vel(self, z, t):
        u = vmap(jacrev(self.get_translation, argnums=1))(z, t)
        return einops.rearrange(u, '... output 1 -> ... output')

    def get_acc(self, z, t):
        a = vmap(hessian(self.get_translation, argnums=1))(z, t)
        return einops.rearrange(z, '... output 1 1 -> ... output')

    def get_local_hessian(self, z, x, t):
        def D(z, x, t):
            d_xyz, rotation, scaling = self.forward(z, t)
            rsx = build_rotation(rotation, False) @ (torch.exp(scaling) * x) + d_xyz
            return rsx

        h = vmap(hessian(D, argnums=(1,2)))(z, x, t)
        return torch.cat([h[1][0], h[1][1]], dim=-1)

    def get_local_vel(self, z, x, t):
        def D(z, x, t):
            d_xyz, rotation, scaling = self.forward(z, t)
            rsx = build_rotation(rotation, False) @ (torch.exp(scaling) * x) + d_xyz
            return rsx

        v = vmap(jacrev(D, argnums=2))(z, x, t)
        v = einops.rearrange(v, '... output 1 -> ... output')
        return v


class CodeField(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, output_ch=8, multires=10):
        super(CodeField, self).__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2]

        self.embed_fn, xyz_input_ch = get_embedder(multires, input_ch)

        self.linear = nn.ModuleList(
            [nn.Linear(xyz_input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + xyz_input_ch, W)
                for i in range(D - 1)]
        )
        self.output = nn.Sequential(
            nn.Linear(W, output_ch),
            # nn.Softmax(dim=-1)
        )

        self.seg = nn.Sequential(
            nn.Linear(output_ch, output_ch * 4),
            nn.ReLU(inplace=False),
            nn.Linear(output_ch * 4, output_ch * 4),
            nn.ReLU(inplace=False),
            nn.Linear(output_ch * 4, output_ch),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, h], -1)
        motion_code = self.output(h)
        return motion_code

class ControlNodeWarp(nn.Module):
    def __init__(self, node_num=512, K=3,  with_node_weight=True,  skinning=False, hyper_dim=2, **kwargs):
        super().__init__()
        self.K = K
        self.name = 'node'
        self.with_node_weight = with_node_weight
        self.hyper_dim = hyper_dim if not skinning else 0  # skinning should not be with hyper
        self.skinning = skinning  # As skin model, discarding KNN weighting
        if 'cn_KNN_method' in list(kwargs.keys()):
            self.KNN_method = kwargs['cn_KNN_method']
        else:
            self.KNN_method = 'xyz'
        self.register_buffer('inited', torch.tensor(False))
        self.nodes = nn.Parameter(torch.randn(node_num, 3))
        # self.nodes_feature = nn.Parameter(torch.randn(node_num, self.hyper_dim)) # use deform_seg instead of deform_code
        # self.nodes = torch.randn(node_num, 3)
        # self.nodes_feature = torch.randn(node_num, self.hyper_dim)
        if not self.skinning:
            self._node_radius = nn.Parameter(torch.randn(node_num))
            if self.with_node_weight:
                self._node_weight = nn.Parameter(torch.zeros_like(self.nodes[:, :1]), requires_grad=with_node_weight)
        # if init_pcl is not None:
        #     self.init(init_pcl)

        # Node colors for visualization
        # self.nodes_color_visualization = torch.ones_like(self.nodes)

        # Cached nn_weight to speed up
        self.cached_nn_weight = False
        self.nn_weight, self.nn_dist, self.nn_idxs = None, None, None
    
    @property
    def node_radius(self):
        return torch.exp(self._node_radius)
    
    @property
    def node_weight(self):
        return torch.sigmoid(self._node_weight)
    
    @property
    def node_num(self):
        return self.nodes.shape[0]
    
    def train_setting(self, training_args):
        l = [
            {'params': self.parameters(),
             'lr': training_args.controlNode_lr ,
             "name": "nodes"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0001, eps=1e-15)

        radius_l = [
            {'params': [self._node_radius],
             'lr': training_args.controlNode_lr,
             "name": "node_radius"}
        ]
        self.radius_optimizer = torch.optim.Adam(radius_l, lr=0.0001, eps=1e-15)


    def init(self,  init_pcl, features = None,  **kwargs):
        """
        Initialize the nodes with given point cloud positions
        Args:
            init_pcl: tensor of shape (N, 3) containing initial point positions
            **kwargs: additional arguments
        """
        scene_range = init_pcl.max() - init_pcl.min()
        self.nodes = nn.Parameter(init_pcl)
        # self.nodes_feature = nn.Parameter(features)
        # self.nodes = init_pcl
        # self.nodes_feature = features
        self._node_radius = nn.Parameter(torch.log(.1 * scene_range + 1e-7) * torch.ones([self.node_num]).float().to(scene_range.device))
        if self.with_node_weight:
            if 'cn_init' in kwargs.keys():
                if kwargs['cn_init'] == 'one':
                    self._node_weight = nn.Parameter(torch.ones_like(torch.ones_like(self.nodes[:, :1]),device='cuda'))
                else:
                    self._node_weight = nn.Parameter(torch.zeros_like(torch.zeros_like(self.nodes[:, :1]),device='cuda'))
            else:    
                self._node_weight = nn.Parameter(torch.zeros_like(torch.zeros_like(self.nodes[:, :1]),device='cuda'))
        
       
    def expand_time(self, t):
        N = self.nodes.shape[0]
        t = t.unsqueeze(0).expand(N, -1)
        return t

    # def cal_nn_weight(self, x, feature, node_feature ):
    #     if self.skinning:
    #         nn_weight = torch.softmax(feature, dim=-1)
    #         nn_idx = torch.arange(0, self.node_num, dtype=torch.long).cuda()
    #         return nn_weight, None, nn_idx
    #     else:
    #         if self.cached_nn_weight and self.nn_weight is not None:
    #             return self.nn_weight, self.nn_dist, self.nn_idxs
    #         else:
    #             if self.hyper_dim > 0 and feature is not None:
    #                 x = torch.cat([x.detach()*0.5, feature[..., :self.hyper_dim]], dim=-1)  # cat with hyper coor
    #             K = self.K 
    #             # Weights of control nodes
    #             nodes = self.nodes
    #             if self.hyper_dim > 0 and feature is not None:
    #                 nodes = torch.cat([nodes*0.5, node_feature], dim=-1)  # Freeze the first 3 coordinates for deformation mlp input
    #             '''可以替换方法 '''
    #             nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
    #             nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K

    #             #if gs_kernel:
    #             nn_radius = self.node_radius[nn_idxs]  # N, K
    #             nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))  # N, K
    #             if self.with_node_weight:
    #                 nn_node_weight = self.node_weight[nn_idxs]
    #                 nn_weight = nn_weight * nn_node_weight[..., 0]
    #             nn_weight = nn_weight + 1e-7
    #             nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
    #             if self.cached_nn_weight:
    #                 self.nn_weight = nn_weight
    #                 self.nn_dist = nn_dist
    #                 self.nn_idxs = nn_idxs
    #             return nn_weight, nn_dist, nn_idxs

    def cal_nn_weight(self, x, feature, node_feature ):
        if self.skinning:
            nn_weight = torch.softmax(feature, dim=-1)
            nn_idx = torch.arange(0, self.node_num, dtype=torch.long).cuda()
            return nn_weight, None, nn_idx
        else:
            if self.cached_nn_weight and self.nn_weight is not None:
                return self.nn_weight, self.nn_dist, self.nn_idxs
            else:
                if self.hyper_dim > 0 and feature is not None and self.KNN_method=='deform_seg':
                    x = torch.cat([x.detach()*0.5, feature[..., :self.hyper_dim]], dim=-1)  # cat with hyper coor
                    nodes = torch.cat([self.nodes*0.5, node_feature], dim=-1)
                   
                elif self.KNN_method=='xyz':
                    nodes = self.nodes
                elif self.KNN_method=='deform_seg_only'and feature is not None and node_feature is not None:
                    x = feature[..., :self.hyper_dim]
                    nodes = node_feature[..., :self.hyper_dim]
                else:
                    nodes = self.nodes   
                K = self.K 
                # Weights of control nodes
                
                # if self.hyper_dim > 0 and feature is not None:
                #     nodes = torch.cat([nodes*0.5, node_feature], dim=-1)  # Freeze the first 3 coordinates for deformation mlp input
                '''可以替换方法 '''
                nn_dist, nn_idxs, _ = pytorch3d.ops.knn_points(x[None], nodes[None], None, None, K=K)  # N, K
                nn_dist, nn_idxs = nn_dist[0], nn_idxs[0]  # N, K

                #if gs_kernel:
                nn_radius = self.node_radius[nn_idxs]  # N, K
                nn_weight = torch.exp(- nn_dist / (2 * nn_radius ** 2))  # N, K
                if self.with_node_weight:
                    nn_node_weight = self.node_weight[nn_idxs]
                    nn_weight = nn_weight * nn_node_weight[..., 0]
                nn_weight = nn_weight + 1e-7
                nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # N, K
                if self.cached_nn_weight:
                    self.nn_weight = nn_weight
                    self.nn_dist = nn_dist
                    self.nn_idxs = nn_idxs
                return nn_weight, nn_dist, nn_idxs
                # else:
                #     nn_weight = torch.softmax(- nn_dist / temperature, dim=-1)
                #     return nn_weight, nn_dist, nn_idxs
    def init_nnweight(self, xyz , deform_seg , node_deform_seg = None ):
        if node_deform_seg == None:
            nn_weight, nn_dist, nn_idxs = self.cal_nn_weight(xyz , deform_seg , self.node_deform_seg)
        else:
            nn_weight, nn_dist, nn_idxs = self.cal_nn_weight(xyz , deform_seg , node_deform_seg)
        self.nn_weight = nn_weight
        self.nn_idxs = nn_idxs
        return self.nn_weight , self.nn_idxs
    

    

    
    def cal_deform(self, nn_weight, nn_idx, node_dxyz, node_drot, node_dscale, method='linear'):
        if method == 'linear':
            translate = (node_dxyz[nn_idx] * nn_weight[..., None]).sum(dim=1)    
            rotation = (node_drot[nn_idx] * nn_weight[..., None]).sum(dim=1)
            scale = (node_dscale[nn_idx] * nn_weight[..., None]).sum(dim=1)
            rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)
            return translate, rotation, scale

        elif method == 'dqb':
            
            sk_R_tq = q2R(node_drot)
            sk_t_tq = node_dxyz
            sk_dq_tq = Rt2dq(sk_R_tq, sk_t_tq)  # N,K,8

            # Dual Quaternion Blending
            dq = torch.einsum("nki,nk->ni", sk_dq_tq[nn_idx], nn_weight)  # N,8
            dq = dq2unitdq(dq)
            R_tq, t_tq = dq2Rt(dq)  # N,3,3; N,3

            # 旋转矩阵转四元数
            rotation = rotation_to_quaternion(R_tq)
            translate = t_tq
            scale = (node_dscale[nn_idx] * nn_weight[..., None]).sum(dim=1)

            return translate, rotation, scale



    # def node_deform(self, network , t, node_deform_code, **kwargs):
        

    #     values = network.step( self.nodes ,  t , node_deform_code , kwargs['dt'] , kwargs['deform_max_time']) # xyz, time_emb, deform_code, dt=1/88, deform_max_time=0.7
    
    #     return values
    
    # def forward(self, network, x, t, deform_code, deform_seg, node_deform_code, node_deform_seg, **kwargs):
    #     '''
    #     x : [N, 3] postion of gs
    #     t : [N , } queried time 
    #     deform_seg : [N, C] deform_seg of target gs
    #     node_deform_code : deform_code of control node
    #     node_deform_seg : deform_sef of control node
    #     '''
    #     if len(t.shape) == 1:
    #         t = self.expand_time(t)
    #     x = x.detach()
    #     # rot_bias = torch.tensor([1., 0, 0, 0]).float().to(x.device)
    #     # Calculate nn weights: [N, K]
    #     if self.KNN_method == 'deform_seg':
    #         nn_weight, _, nn_idx = self.cal_nn_weight(x=x, feature=deform_seg ,node_feature = node_deform_seg)
    #     elif self.KNN_method == 'deform_code':
    #         nn_weight, _, nn_idx = self.cal_nn_weight(x=x, feature=deform_code ,node_feature = node_deform_code)
    #     elif self.KNN_method == 'xyz':
    #         nn_weight, _, nn_idx = self.cal_nn_weight(x=x, feature=None ,node_feature = None)
    #     node_trans, node_rot, node_scale  = self.node_deform(network , t , node_deform_code, **kwargs)
        
    #     # Obtain translation
    #     # if self.local_frame:
    #     #     local_rot = node_attrs['local_rotation'] + rot_bias
    #     #     local_rot_matrix = quaternion_to_matrix(local_rot)
    #     #     nn_nodes = self.nodes[nn_idx,...,:3].detach()
    #     #     Ax = torch.einsum('nkab,nkb->nka', local_rot_matrix[nn_idx], x[:, None] - nn_nodes) + nn_nodes + node_trans[nn_idx]
    #     #     Ax_avg = (Ax * nn_weight[..., None]).sum(dim=1)
    #     #     translate = Ax_avg - x
    #     # else:
    #     translate = (node_trans[nn_idx] * nn_weight[..., None]).sum(dim=1)    
    #     rotation = (node_rot[nn_idx] * nn_weight[..., None]).sum(dim=1)
    #     scale = (node_scale[nn_idx] * nn_weight[..., None]).sum(dim=1)
    #     rotation = rotation / torch.norm(rotation, dim=-1, keepdim=True)
    #     return translate , rotation , scale