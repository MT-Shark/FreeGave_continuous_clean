import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torch.nn.functional as F

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

    
class FlowGenerator:
    def __init__(self, model_type='large'):
        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()
        self.model = raft_large(weights=self.weights).eval().cuda()

    
    def preprocess(self, img_batch1 , img_batch2):
        """输入: [B,3,H,W], 输出: 归一化+填充后的张量"""
        normalized1 , normalized2 = self.transforms(img_batch1 , img_batch2) # 自动归一化
        padder = InputPadder(normalized1.shape)
        return padder.pad(normalized1.cuda()),  padder.pad(normalized2.cuda()) , padder
        
    def postprocess(self, flow_padded, padder):
        """裁剪填充区域并转换到CPU"""
        return padder.unpad(flow_padded)
    
    def __call__(self, img_batch1 , img_batch2):
    
        # 预处理
        batch1, batch2 , padder = self.preprocess(img_batch1 , img_batch2)
        
        
        # 批量推理
        with torch.no_grad():
            flows_padded = self.model(batch1[0], batch2[0])[-1]
        
        # 后处理
        return torch.stack([self.postprocess(f, padder) for f in flows_padded])