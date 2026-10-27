"""Transfer only the coset/parity probabilities that a legal decoder can use."""
from typing import List
import numpy as np
import torch
from torch import Tensor
from scec.deployment.decode_fast import DecodeFields
from scec.deployment.pinned_graph import PinnedOutputEntry
from scec.deployment.cached_transform import SCECCachedTransformCodec
from scec.deployment.quantized_fast import njit
from scec.deployment.native_graph import compact_probabilities,scatter_coordinates


class CompactDecodeFields(DecodeFields):
    def __init__(self,model):
        super().__init__(model)
        # First seven coordinates use state 0 or 2; the eighth uses all four.
        self.register_buffer('coordinate',torch.tensor(list(range(7))*2+[7]*4,device=model.steps.device))
        self.register_buffer('state',torch.tensor([0]*7+[2]*7+list(range(4)),device=model.steps.device))

    def forward(self,q:Tensor,c:Tensor,profile:Tensor,base:List[Tensor],ids:Tensor,known:Tensor,use_context:bool):
        p,s,pc,g,pl,sl,cl,gl,freq,res,step=base
        if self.neural and use_context:
            normalized=torch.sign(q.float())*torch.log1p(q.abs().float())/4.0
            support=(q!=0).float();coset=c.float().unsqueeze(-1)
            corr=self.net(normalized,support,coset,known,pl,sl,cl,freq,res,step)[0].index_select(0,ids)
            pp=torch.sigmoid(pl[0].index_select(0,ids)+2.5*torch.tanh(corr[:,:8])).clamp(1e-4,1-1e-4)
            ss=(s[0].index_select(0,ids)*torch.exp(2.2*torch.tanh(corr[:,8:16]))).clamp_min(.03)
            cp=torch.sigmoid(cl[0].index_select(0,ids)+2.5*torch.tanh(corr[:,16])).clamp(1e-4,1-1e-4)
            gg=torch.sigmoid(gl[0].index_select(0,ids)+2.5*torch.tanh(corr[:,17:25])).clamp(1e-4,1-1e-4)
        else:
            pp=p[0].index_select(0,ids);ss=s[0].index_select(0,ids)
            cp=pc[0].index_select(0,ids);gg=g[0].index_select(0,ids)
        band=ids*self.bands//q.shape[1]
        if self.conditioned:
            op=self.support_offset[profile[0],band[:,None],self.state[None,:],self.coordinate[None,:]]
            os=self.scale_offset[profile[0],band[:,None],self.state[None,:],self.coordinate[None,:]]
            og=self.sign_offset[profile[0],band[:,None],self.state[None,:],self.coordinate[None,:]]
            pp=torch.sigmoid(torch.logit(pp.clamp(1e-5,1-1e-5)).index_select(1,self.coordinate)+op).clamp(1e-5,1-1e-5)
            ss=(ss.index_select(1,self.coordinate)*torch.exp(os)).clamp_min(.03)
            gg=torch.sigmoid(torch.logit(gg.clamp(1e-5,1-1e-5)).index_select(1,self.coordinate)+og).clamp(1e-5,1-1e-5)
        else:
            pp=pp.index_select(1,self.coordinate);ss=ss.index_select(1,self.coordinate);gg=gg.index_select(1,self.coordinate)
        return torch.cat([cp[:,None],pp,ss,gg],dim=1)


class CompactEntry(PinnedOutputEntry):
    def run(self,stage):
        if stage<0:return super().run(stage)
        if stage==0:self.base=self.fields()
        return self.codec.compact_decode_net(self.q,self.c,self.profile,self.base,
                                             self.indices[stage],self.masks[stage],stage>0)


@njit(cache=True,nogil=True)
def compact_select(fields,c,q,ids,last):
    width=1 if last else 7;total=len(ids)*width
    p=np.empty(total,np.float32);s=np.empty(total,np.float32);g=np.empty(total,np.float32)
    for k in range(len(ids)):
        i=ids[k];base=7*c[i]
        if last:
            parity=0
            for j in range(7):parity+=q[i,j]
            base=14+2*c[i]+parity%2
        for jj in range(width):
            source=base+jj;target=k*width+jj
            p[target]=fields[k,1+source];s[target]=fields[k,19+source];g[target]=fields[k,37+source]
    return p,s,g


class SCECCompactGraphCodec(SCECCachedTransformCodec):
    def __init__(self,model,**kwargs):
        super().__init__(model,**kwargs)
        self.compact_decode_net=torch.jit.freeze(torch.jit.script(CompactDecodeFields(model).eval()),optimize_numerics=False)

    def _entry(self,groups,role):
        if self.device.type!='cuda' or groups>self.max_graph_groups:return None
        cache=self._graphs[role]
        if groups not in cache:
            if len(cache)>=self.max_graph_shapes:return None
            cache[groups]=CompactEntry(self,groups,role)
        return cache[groups]

    def _decode_stage(self,packed,c,q,ids,bp,bg,bs,bernoulli,laplace):
        for last in (False,True):
            pp,sp,gp=compact_select(packed,c,q,ids,last)
            nz=bp.decode(bernoulli,pp);sp,gp=compact_probabilities(nz,sp,gp)
            signs=np.empty(0,np.int32);mags=np.empty(0,np.int32)
            if len(sp):
                signs=bg.decode(bernoulli,gp)
                mags=bs.decode(laplace,np.zeros(len(sp),np.float32),sp)
            scatter_coordinates(q,ids,nz,signs,mags,last)
