"""Transfer-coalesced SCEC encoder; preserves the released decoder and stream.

All content-dependent work is performed per record. Only model/profile/length
fields use the same one-entry cache as the reference deployment path. State
offsets are gathered once per encoded record, since encoder symbols are fixed.
"""
import numpy as np
import torch
import constriction
from scec.deployment.inference import SCECCodec
from scec.entropy import ProgressiveNeuralE8ANS
from scec.reference import HEADER,MAGIC


class SCECFastCodec(SCECCodec):
    @torch.inference_mode()
    def encode_bytes(self,x,step):
        x=np.asarray(x,dtype=np.float64);length=len(x)
        padded=np.pad(x,(0,(-length)%8),mode='reflect' if length>1 else 'edge')
        pi=int(np.argmin(abs(self._steps-step)))
        if not np.isclose(float(self._steps[pi]),step):raise ValueError('unknown step')
        q,c=self.quantize(padded,step);groups=len(c)
        qt=torch.from_numpy(q).unsqueeze(0).to(self.device)
        ct=torch.from_numpy(c).unsqueeze(0).to(self.device)
        pt=torch.tensor([pi],device=self.device)
        offsets=None
        if self.model.conditioned:
            state=(2*ct.long()).unsqueeze(-1).expand(-1,-1,8).clone()
            state[:,:,7]+=qt[:,:,:7].long().sum(-1).remainder(2)
            band=torch.arange(groups,device=self.device)*self.model.bands//groups
            j=torch.arange(8,device=self.device).view(1,1,8)
            idx=(pt[:,None,None],band[None,:,None],state,j)
            offsets=[getattr(self.model,'state_'+name+'_offset')[idx] for name in ['support','scale','sign']]
        known=np.zeros(groups,bool);parts=[];cs=[];qs=[];counts=[]
        index=np.arange(groups)
        masks=[index%8==0,index%8==4,index%4==2,index%2==1]
        for mask in masks:
            common=self._common(qt,ct,pt,known,(groups,pi))
            p,s,pc,g=common
            if offsets is not None:
                p=torch.sigmoid(torch.logit(p.clamp(1e-5,1-1e-5))+offsets[0]).clamp(1e-5,1-1e-5)
                s=(s*torch.exp(offsets[1])).clamp_min(.03)
                g=torch.sigmoid(torch.logit(g.clamp(1e-5,1-1e-5))+offsets[2]).clamp(1e-5,1-1e-5)
            ids=np.flatnonzero(mask);ti=torch.from_numpy(ids).to(self.device)
            n=len(ids);counts.append(n);cs.append(c[ids])
            parts.append(pc[0].index_select(0,ti).reshape(-1))
            for value in [p,s,g]:
                selected=value[0].index_select(0,ti)
                parts.extend([selected[:,:7].reshape(-1),selected[:,7:].reshape(-1)])
            qs.extend([q[ids,:7].reshape(-1),q[ids,7:].reshape(-1)])
            known|=mask
        # One blocking device-to-host transfer for all four causal stages.
        packed=torch.cat(parts).cpu().numpy()
        cps=[];ps=[];ss=[];gs=[];offset=0
        for n in counts:
            cps.append(packed[offset:offset+n]);offset+=n
            for target in [ps,ss,gs]:
                target.append(packed[offset:offset+8*n]);offset+=8*n
        coder=constriction.stream.stack.AnsCoder()
        if self.model.quantizer=='e8':
            coder.encode_reverse(np.concatenate(cs),constriction.stream.model.Bernoulli(perfect=False),np.ascontiguousarray(np.concatenate(cps),dtype=np.float32))
        cb=ProgressiveNeuralE8ANS._ans_bytes(coder)
        maximum,bp,bg,bs=ProgressiveNeuralE8ANS._encode_coordinates(np.concatenate(qs),np.concatenate(ps),np.concatenate(ss),np.concatenate(gs))
        streams=[cb,bp,bg,bs]
        flags=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        return HEADER.pack(MAGIC,1,pi,flags,self.model.bands,length,float(step),maximum,*map(len,streams))+b''.join(streams)
