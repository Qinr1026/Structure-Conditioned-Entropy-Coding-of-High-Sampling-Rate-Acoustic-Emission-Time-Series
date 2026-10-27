"""Exact encoder/decoder with coalesced symbols and GPU-generated stage layout."""
from typing import List
import numpy as np
import torch
from torch import nn,Tensor
from scec.deployment.decode_fast import SCECDecodeFastCodec
from scec.deployment.stream_compiled import EncoderFields


@torch.jit.script
def device_layout(q:Tensor):
    n=q.shape[1];i=torch.arange(n,device=q.device)
    indices=[torch.arange(0,n,8,device=q.device),torch.arange(4,max(4,n),8,device=q.device),torch.arange(2,max(2,n),4,device=q.device),torch.arange(1,max(1,n),2,device=q.device)]
    masks=[torch.zeros((1,n,1),dtype=torch.bool,device=q.device),(i.remainder(8)==0).view(1,n,1),(i.remainder(4)==0).view(1,n,1),(i.remainder(2)==0).view(1,n,1)]
    return indices,masks


class DeviceEncoder(nn.Module):
    def __init__(self,m):
        super().__init__();self.net=EncoderFields(m)

    def forward(self,q:Tensor,c:Tensor,profile:Tensor,base:List[Tensor]):
        indices,masks=device_layout(q)
        return self.net(q,c,profile,base,indices,masks)


class SCECFastIOCodec(SCECDecodeFastCodec):
    def __init__(self,model):
        super().__init__(model)
        self.device_encoder=torch.jit.freeze(torch.jit.script(DeviceEncoder(model).eval()),optimize_numerics=False)
        self._profile_tensors=[torch.tensor([i],device=self.device) for i in range(len(self._steps))]
        self.use_device_layout=True

    def _profile_tensor(self,pi):
        return self._profile_tensors[pi]

    def _upload_symbols(self,q,c):
        groups=len(c)
        packed=torch.from_numpy(np.concatenate([q.ravel(),c])).to(self.device)
        return packed[:8*groups].view(1,groups,8),packed[8*groups:].view(1,groups)

    def _device_layout(self,qt):
        return device_layout(qt)

    @torch.inference_mode()
    def predict(self,record):
        q,c,pi=record.q,record.c,record.profile;groups=len(c)
        qt,ct=self._upload_symbols(q,c);pt=self._profile_tensor(pi)
        base=list(self._shared_fields(pt,groups,(groups,pi)))
        return self.device_encoder(qt,ct,pt,base).cpu().numpy()
