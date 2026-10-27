"""Selected-field SCEC inference and bounded cross-record encoder pipeline.

The network, symbols, stage order and byte format are unchanged. CPU preparation,
GPU prediction and CPU entropy coding are explicit stages. A record is never
used as a probability context for another record. No signal results are cached.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import constriction
from scec.deployment.quantized_fast import SCECQuantizedFastCodec
from scec.entropy import ProgressiveNeuralE8ANS
from scec.reference import HEADER,MAGIC


class CorrectionNetwork(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.input_projection=m.input_projection;self.blocks=m.blocks
        self.output_projection=m.output_projection

    def forward(self,normalized,support,coset,mask,pl,sl,cl,freq,res,step):
        mv=mask.to(torch.int32)
        features=torch.cat([normalized*mv,support*mv,coset*mv,mv,pl,sl,
                            cl.unsqueeze(-1),freq,res,step],dim=-1)
        h=F.gelu(self.input_projection(features.transpose(1,2)))
        for block in self.blocks:h=block(h)
        return self.output_projection(h).transpose(1,2)


@dataclass
class Prepared:
    length:int
    step:float
    profile:int
    q:np.ndarray
    c:np.ndarray
    ids:list


class SCECStreamCodec(SCECQuantizedFastCodec):
    def __init__(self,model):
        super().__init__(model)
        self.correction_net=torch.jit.freeze(torch.jit.script(CorrectionNetwork(model).eval()),optimize_numerics=False)

    def prepare(self,x,step):
        x=np.asarray(x,dtype=np.float64);length=len(x)
        if length<1:raise ValueError('empty waveform')
        padded=np.pad(x,(0,(-length)%8),mode='reflect' if length>1 else 'edge')
        pi=int(np.argmin(abs(self._steps-step)))
        if not np.isclose(float(self._steps[pi]),step):raise ValueError('unknown step')
        q,c=self.quantize(padded,step);i=np.arange(len(c))
        ids=[np.flatnonzero(m) for m in [i%8==0,i%8==4,i%4==2,i%2==1]]
        return Prepared(length,float(step),pi,q,c,ids)

    @torch.inference_mode()
    def predict(self,record):
        q,c,pi=record.q,record.c,record.profile;groups=len(c)
        qt=torch.from_numpy(q).unsqueeze(0).to(self.device)
        ct=torch.from_numpy(c).unsqueeze(0).to(self.device)
        pt=torch.tensor([pi],device=self.device)
        p,s,pc,g,pl,sl,cl,gl,freq,res,step=self._shared_fields(pt,groups,(groups,pi))
        normalized=torch.sign(qt.float())*torch.log1p(qt.abs().float())/4.0
        support=(qt!=0).float();coset=ct.float().unsqueeze(-1)
        offsets=None
        if self.model.conditioned:
            state=(2*ct.long()).unsqueeze(-1).expand(-1,-1,8).clone()
            state[:,:,7]+=qt[:,:,:7].long().sum(-1).remainder(2)
            band=torch.arange(groups,device=self.device)*self.model.bands//groups
            j=torch.arange(8,device=self.device).view(1,1,8)
            idx=(pt[:,None,None],band[None,:,None],state,j)
            offsets=[getattr(self.model,'state_'+name+'_offset')[idx][0] for name in ['support','scale','sign']]
        known=np.zeros(groups,bool);parts=[]
        for ids in record.ids:
            ti=torch.from_numpy(ids).to(self.device)
            if self.model.neural and known.any():
                mask=torch.from_numpy(known).to(self.device).view(1,groups,1)
                corr=self.correction_net(normalized,support,coset,mask,pl,sl,cl,freq,res,step)[0].index_select(0,ti)
                # Current target groups are always unknown (nonanchor == 1).
                pp=torch.sigmoid(pl[0].index_select(0,ti)+2.5*torch.tanh(corr[:,:8])).clamp(1e-4,1-1e-4)
                ss=(s[0].index_select(0,ti)*torch.exp(2.2*torch.tanh(corr[:,8:16]))).clamp_min(.03)
                cp=torch.sigmoid(cl[0].index_select(0,ti)+2.5*torch.tanh(corr[:,16])).clamp(1e-4,1-1e-4)
                gg=torch.sigmoid(gl[0].index_select(0,ti)+2.5*torch.tanh(corr[:,17:25])).clamp(1e-4,1-1e-4)
            else:
                pp,ss,cp,gg=[v[0].index_select(0,ti) for v in [p,s,pc,g]]
            if offsets is not None:
                pp=torch.sigmoid(torch.logit(pp.clamp(1e-5,1-1e-5))+offsets[0].index_select(0,ti)).clamp(1e-5,1-1e-5)
                ss=(ss*torch.exp(offsets[1].index_select(0,ti))).clamp_min(.03)
                gg=torch.sigmoid(torch.logit(gg.clamp(1e-5,1-1e-5))+offsets[2].index_select(0,ti)).clamp(1e-5,1-1e-5)
            parts.append(torch.cat([cp[:,None],pp,ss,gg],dim=1))
            known[ids]=True
        return torch.cat(parts).cpu().numpy()

    def finish(self,record,packed):
        cs=[];qs=[];ps=[];ss=[];gs=[];offset=0
        for ids in record.ids:
            n=len(ids);fields=packed[offset:offset+n];offset+=n
            cs.append(record.c[ids])
            for jj in [slice(0,7),slice(7,8)]:
                qs.append(record.q[ids,jj].ravel())
                for target,begin in [(ps,1),(ss,9),(gs,17)]:
                    target.append(fields[:,begin+jj.start:begin+jj.stop].ravel())
        coder=constriction.stream.stack.AnsCoder()
        if self.model.quantizer=='e8':
            coder.encode_reverse(np.concatenate(cs),constriction.stream.model.Bernoulli(perfect=False),np.ascontiguousarray(packed[:,0],dtype=np.float32))
        cb=ProgressiveNeuralE8ANS._ans_bytes(coder)
        maximum,bp,bg,bs=ProgressiveNeuralE8ANS._encode_coordinates(np.concatenate(qs),np.concatenate(ps),np.concatenate(ss),np.concatenate(gs))
        streams=[cb,bp,bg,bs]
        flags=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        return HEADER.pack(MAGIC,1,record.profile,flags,self.model.bands,record.length,record.step,maximum,*map(len,streams))+b''.join(streams)

    def encode_bytes(self,x,step):
        record=self.prepare(x,step)
        return self.finish(record,self.predict(record))

    def encode_many(self,records,step,max_inflight=2):
        """Yield ordered complete streams with bounded memory, including pool setup.

        Two CPU workers prepare future records and entropy-code prior predictions;
        the caller alone executes GPU inference and owns the model-field cache.
        Input records must not be mutated until their output has been yielded.
        """
        if max_inflight<1:raise ValueError('max_inflight must be positive')
        source=iter(records);prepared=deque();finished=deque()
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='scec-cpu') as pool:
            for _ in range(max_inflight):
                try:x=next(source)
                except StopIteration:break
                prepared.append(pool.submit(self.prepare,x,step))
            while prepared:
                while finished and finished[0].done():yield finished.popleft().result()
                record=prepared.popleft().result()
                prediction=self.predict(record)
                finished.append(pool.submit(self.finish,record,prediction))
                try:x=next(source)
                except StopIteration:pass
                else:prepared.append(pool.submit(self.prepare,x,step))
                while finished and (finished[0].done() or len(finished)>=max_inflight):
                    yield finished.popleft().result()
            while finished:yield finished.popleft().result()
