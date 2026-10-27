"""Exact-state enumeration avoids intra-stage GPU round trips in SCEC decode.

Probabilities for all four legal coset/parity states are prepared in parallel.
The CPU selects a state only after its required symbols have been decoded.
The network, model weights, probability arithmetic and byte format are unchanged.
"""
from typing import List
import numpy as np
import torch
from torch import nn,Tensor
import constriction
from scipy.fft import idct
from scec.quantization import restore_doubled
from scec.reference import HEADER,MAGIC
from scec.deployment.stream import CorrectionNetwork
from scec.deployment.stream_compiled import SCECCompiledStreamCodec


class DecodeFields(nn.Module):
    __constants__=['bands','conditioned','neural']
    def __init__(self,m):
        super().__init__();self.net=CorrectionNetwork(m)
        self.bands=m.bands;self.conditioned=m.conditioned;self.neural=m.neural
        self.register_buffer('support_offset',m.state_support_offset)
        self.register_buffer('scale_offset',m.state_scale_offset)
        self.register_buffer('sign_offset',m.state_sign_offset)

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
        state=torch.arange(4,device=q.device)
        j=torch.arange(8,device=q.device)
        if self.conditioned:
            op=self.support_offset[profile[0],band[:,None,None],state[None,:,None],j[None,None,:]]
            os=self.scale_offset[profile[0],band[:,None,None],state[None,:,None],j[None,None,:]]
            og=self.sign_offset[profile[0],band[:,None,None],state[None,:,None],j[None,None,:]]
            pp=torch.sigmoid(torch.logit(pp.clamp(1e-5,1-1e-5))[:,None,:]+op).clamp(1e-5,1-1e-5)
            ss=(ss[:,None,:]*torch.exp(os)).clamp_min(.03)
            gg=torch.sigmoid(torch.logit(gg.clamp(1e-5,1-1e-5))[:,None,:]+og).clamp(1e-5,1-1e-5)
        else:
            pp=pp[:,None,:].expand(-1,4,-1);ss=ss[:,None,:].expand(-1,4,-1);gg=gg[:,None,:].expand(-1,4,-1)
        return torch.cat([cp[:,None],pp.reshape(-1,32),ss.reshape(-1,32),gg.reshape(-1,32)],dim=1)


class SCECDecodeFastCodec(SCECCompiledStreamCodec):
    def __init__(self,model):
        super().__init__(model)
        self.decode_net=torch.jit.freeze(torch.jit.script(DecodeFields(model).eval()),optimize_numerics=False)
        self.use_device_layout=False

    def _profile_tensor(self,pi):
        return torch.tensor([pi],device=self.device)

    def _upload_symbols(self,q,c):
        return (torch.from_numpy(q).unsqueeze(0).to(self.device),
                torch.from_numpy(c).unsqueeze(0).to(self.device))

    @torch.inference_mode()
    def decode(self,payload):
        if len(payload)<HEADER.size:raise ValueError('truncated header')
        magic,version,pi,flags,bands,length,step,maximum,*lengths=HEADER.unpack_from(payload)
        expected=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        if magic!=MAGIC or version!=1 or flags!=expected or bands!=self.model.bands:raise ValueError('model/header mismatch')
        if length<1 or pi>=len(self._steps) or not np.isclose(step,float(self._steps[pi])):raise ValueError('invalid profile or length')
        if HEADER.size+sum(lengths)!=len(payload) or any(n%4 for n in lengths):raise ValueError('invalid lengths')
        streams=[];offset=HEADER.size
        for n in lengths:
            streams.append(constriction.stream.stack.AnsCoder(np.frombuffer(payload[offset:offset+n],dtype=np.uint32)));offset+=n
        cc,bp,bg,bs=streams;groups=(length+7)//8
        q=np.zeros((groups,8),np.int32);c=np.zeros(groups,np.int32);known=np.zeros(groups,bool)
        pt=self._profile_tensor(pi)
        base=list(self._shared_fields(pt,groups,(groups,pi)))
        index=np.arange(groups)
        masks=[index%8==0,index%8==4,index%4==2,index%2==1]
        bernoulli=constriction.stream.model.Bernoulli(perfect=False)
        laplace=constriction.stream.model.QuantizedLaplace(0,maximum)
        layout=None
        for stage,mask in enumerate(masks):
            ids=np.flatnonzero(mask);n=len(ids)
            if n==0:continue
            qt,ct=self._upload_symbols(q,c)
            if self.use_device_layout:
                if layout is None:layout=self._device_layout(qt)
                ti,kt=layout[0][stage],layout[1][stage]
            else:
                ti=torch.from_numpy(ids).to(self.device)
                kt=torch.from_numpy(known).view(1,groups,1).to(self.device)
            packed=self.decode_net(qt,ct,pt,base,ti,kt,stage>0).cpu().numpy()
            if self.model.quantizer=='e8':c[ids]=cc.decode(bernoulli,np.ascontiguousarray(packed[:,0]))
            fields=[packed[:,i:i+32].reshape(n,4,8) for i in [1,33,65]]
            rows=np.arange(n);state=2*c[ids]
            for first,last in [(0,7),(7,8)]:
                if first==7:state=2*c[ids]+q[ids,:7].sum(1)%2
                pp,sp,gp=[np.ascontiguousarray(v[rows,state,first:last].ravel()) for v in fields]
                nz=bp.decode(bernoulli,pp).astype(bool)
                values=np.zeros(len(pp),np.int32);nn=int(nz.sum())
                if nn:
                    signs=bg.decode(bernoulli,np.ascontiguousarray(gp[nz])).astype(bool)
                    mags=bs.decode(laplace,np.zeros(nn,np.float32),np.ascontiguousarray(sp[nz])).astype(np.int32)
                    values[nz]=(mags+1)*np.where(signs,-1,1)
                q[ids,first:last]=values.reshape(-1,last-first)
            known[ids]=True
        y=restore_doubled(c,q)*.5 if self.model.quantizer=='e8' else q.ravel()
        return idct(y*step,norm='ortho')[:length]
