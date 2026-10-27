"""Validated scripted execution of the unchanged SCEC probability network.

Numerical graph rewrites and JIT fusion are disabled deliberately: the default
freeze optimizations changed entropy probabilities and failed byte equality.
The JIT fusion settings are process-global; use a dedicated codec process.
"""
import torch
from torch import nn
from torch.nn import functional as F
from scec.deployment.fast import SCECFastCodec


class CommonNetwork(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.input_projection=m.input_projection
        self.blocks=m.blocks
        self.output_projection=m.output_projection

    def forward(self,q,c,mask,s,pl,sl,cl,gl,freq,res,step):
        mask_values=mask.to(q.dtype)
        normalized=torch.sign(q.float())*torch.log1p(q.abs().float())/4.0
        features=torch.cat([normalized*mask_values,(q!=0).float()*mask_values,
            c.float().unsqueeze(-1)*mask_values,mask_values,pl,sl,cl.unsqueeze(-1),
            freq,res,step],dim=-1)
        h=F.gelu(self.input_projection(features.transpose(1,2)))
        for block in self.blocks:h=block(h)
        correction=self.output_projection(h).transpose(1,2)
        nonanchor=(~mask).float()
        outp=torch.sigmoid(pl+nonanchor*2.5*torch.tanh(correction[...,:8])).clamp(1e-4,1-1e-4)
        outs=s*torch.exp(nonanchor*2.2*torch.tanh(correction[...,8:16]))
        outc=torch.sigmoid(cl+(~mask[...,0]).float()*2.5*torch.tanh(correction[...,16])).clamp(1e-4,1-1e-4)
        outg=torch.sigmoid(gl+nonanchor*2.5*torch.tanh(correction[...,17:25])).clamp(1e-4,1-1e-4)
        return outp,outs.clamp_min(.03),outc,outg


class SCECScriptedCodec(SCECFastCodec):
    def __init__(self,model):
        super().__init__(model)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        self.common_net=torch.jit.freeze(torch.jit.script(CommonNetwork(model).eval()),optimize_numerics=False)

    def _common(self,q,c,profile,known,key):
        groups=q.shape[1]
        p,s,pc,g,pl,sl,cl,gl,freq,res,step=self._shared_fields(profile,groups,key)
        if not self.model.neural or not known.any():return p,s,pc,g
        mask=torch.from_numpy(known).to(self.device).view(1,groups,1)
        return self.common_net(q,c,mask,s,pl,sl,cl,gl,freq,res,step)
