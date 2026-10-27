"""Compile the whole four-stage encoder probability schedule, with exact math."""
from typing import List
import numpy as np
import torch
from torch import nn,Tensor
from scec.deployment.stream import SCECStreamCodec,CorrectionNetwork


class EncoderFields(nn.Module):
    __constants__=['bands','conditioned','neural']
    def __init__(self,m):
        super().__init__();self.net=CorrectionNetwork(m)
        self.bands=m.bands;self.conditioned=m.conditioned;self.neural=m.neural
        self.register_buffer('support_offset',m.state_support_offset)
        self.register_buffer('scale_offset',m.state_scale_offset)
        self.register_buffer('sign_offset',m.state_sign_offset)

    def forward(self,q:Tensor,c:Tensor,profile:Tensor,base:List[Tensor],indices:List[Tensor],masks:List[Tensor]):
        p,s,pc,g,pl,sl,cl,gl,freq,res,step=base
        groups=q.shape[1]
        normalized=torch.sign(q.float())*torch.log1p(q.abs().float())/4.0
        support=(q!=0).float();coset=c.float().unsqueeze(-1)
        state=(2*c.long()).unsqueeze(-1).expand(-1,-1,8).clone()
        state[:,:,7]+=q[:,:,:7].long().sum(-1).remainder(2)
        band=torch.arange(groups,device=q.device)*self.bands//groups
        j=torch.arange(8,device=q.device).view(1,1,8)
        op=self.support_offset[profile[:,None,None],band[None,:,None],state,j][0]
        os=self.scale_offset[profile[:,None,None],band[None,:,None],state,j][0]
        og=self.sign_offset[profile[:,None,None],band[None,:,None],state,j][0]
        parts=torch.jit.annotate(List[Tensor],[])
        for stage in range(4):
            ti=indices[stage]
            if self.neural and stage>0:
                corr=self.net(normalized,support,coset,masks[stage],pl,sl,cl,freq,res,step)[0].index_select(0,ti)
                pp=torch.sigmoid(pl[0].index_select(0,ti)+2.5*torch.tanh(corr[:,:8])).clamp(1e-4,1-1e-4)
                ss=(s[0].index_select(0,ti)*torch.exp(2.2*torch.tanh(corr[:,8:16]))).clamp_min(.03)
                cp=torch.sigmoid(cl[0].index_select(0,ti)+2.5*torch.tanh(corr[:,16])).clamp(1e-4,1-1e-4)
                gg=torch.sigmoid(gl[0].index_select(0,ti)+2.5*torch.tanh(corr[:,17:25])).clamp(1e-4,1-1e-4)
            else:
                pp=p[0].index_select(0,ti);ss=s[0].index_select(0,ti)
                cp=pc[0].index_select(0,ti);gg=g[0].index_select(0,ti)
            if self.conditioned:
                pp=torch.sigmoid(torch.logit(pp.clamp(1e-5,1-1e-5))+op.index_select(0,ti)).clamp(1e-5,1-1e-5)
                ss=(ss*torch.exp(os.index_select(0,ti))).clamp_min(.03)
                gg=torch.sigmoid(torch.logit(gg.clamp(1e-5,1-1e-5))+og.index_select(0,ti)).clamp(1e-5,1-1e-5)
            parts.append(torch.cat([cp[:,None],pp,ss,gg],dim=1))
        return torch.cat(parts)


class SCECCompiledStreamCodec(SCECStreamCodec):
    def __init__(self,model):
        super().__init__(model)
        self.encoder_net=torch.jit.freeze(torch.jit.script(EncoderFields(model).eval()),optimize_numerics=False)

    @torch.inference_mode()
    def predict(self,record):
        q,c,pi=record.q,record.c,record.profile;groups=len(c)
        qt=torch.from_numpy(q).unsqueeze(0).to(self.device)
        ct=torch.from_numpy(c).unsqueeze(0).to(self.device)
        pt=torch.tensor([pi],device=self.device)
        base=list(self._shared_fields(pt,groups,(groups,pi)))
        known=np.zeros(groups,bool);masks=[];indices=[]
        for ids in record.ids:
            masks.append(torch.from_numpy(known.copy()).view(1,groups,1).to(self.device))
            indices.append(torch.from_numpy(ids).to(self.device));known[ids]=True
        return self.encoder_net(qt,ct,pt,base,indices,masks).cpu().numpy()
