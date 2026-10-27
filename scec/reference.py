"""Actual ANS codec with staged E8-state-dependent coordinate probabilities.

Four stacks and one 37-byte header for every factorial variant. All payload
bytes count. Shared models are out of band and their sizes must be reported.
"""
import struct,time
import numpy as np
import torch
import constriction
from scipy.fft import idct
from scec.quantization import symbols
from scec.quantization import restore_doubled
from scec.entropy import ProgressiveNeuralE8ANS
from scec.schedule import progressive_masks
HEADER=struct.Struct('<4sBBBHIf'+'I'*5)
MAGIC=b'SCE8'

class StateCodec:
    def __init__(self,model):
        self.model=model.eval();self.device=next(model.parameters()).device
    def masks(self,groups):
        a,t=progressive_masks(groups,8,torch.device('cpu'))
        return [a.numpy(),*[v.numpy() for v in t]]
    def quantize(self,padded,step):
        return symbols(padded,step,self.model.quantizer)
    @torch.no_grad()
    def fields(self,q,c,pi,known,common=None):
        qt=torch.from_numpy(q).unsqueeze(0).to(self.device)
        ct=torch.from_numpy(c).unsqueeze(0).to(self.device)
        pt=torch.tensor([pi],device=self.device)
        if common is None:
            common=self.model.common_fields(qt,ct,pt,torch.from_numpy(known).to(self.device))
        f=self.model.condition(common,qt,ct,pt)
        return [v[0].cpu().numpy() for v in f],common
    def encode(self,x,step):
        started=time.perf_counter();x=np.asarray(x,dtype=np.float64);length=len(x)
        padded=np.pad(x,(0,(-length)%8),mode='reflect' if length>1 else 'edge')
        pi=int(np.argmin(abs(self.model.steps.cpu().numpy()-step)))
        if not np.isclose(float(self.model.steps[pi]),step):raise ValueError('unknown step')
        q,c=self.quantize(padded,step);groups=len(c);known=np.zeros(groups,bool)
        cs=[];cps=[];qs=[];ps=[];ss=[];gs=[]
        for mask in self.masks(groups):
            (p,s,pc,g),_=self.fields(q,c,pi,known)
            cs.append(c[mask]);cps.append(pc[mask])
            # Decode first seven values before computing the last-coordinate state.
            for jj in [slice(0,7),slice(7,8)]:
                qs.append(q[mask,jj].ravel());ps.append(p[mask,jj].ravel())
                ss.append(s[mask,jj].ravel());gs.append(g[mask,jj].ravel())
            known|=mask
        coder=constriction.stream.stack.AnsCoder()
        if self.model.quantizer=='e8':
            coder.encode_reverse(np.concatenate(cs),constriction.stream.model.Bernoulli(perfect=False),
                np.ascontiguousarray(np.concatenate(cps),dtype=np.float32))
        cbytes=ProgressiveNeuralE8ANS._ans_bytes(coder)
        maximum,bp,bg,bs=ProgressiveNeuralE8ANS._encode_coordinates(np.concatenate(qs),
            np.concatenate(ps),np.concatenate(ss),np.concatenate(gs))
        streams=[cbytes,bp,bg,bs]
        flags=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        header=HEADER.pack(MAGIC,1,pi,flags,self.model.bands,length,float(step),maximum,*map(len,streams))
        y=restore_doubled(c,q)*.5 if self.model.quantizer=='e8' else q.ravel()
        return header+b''.join(streams),idct(y*step,norm='ortho')[:length],1000*(time.perf_counter()-started)
    def decode(self,payload):
        magic,version,pi,flags,bands,length,step,maximum,*lengths=HEADER.unpack_from(payload)
        expected=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        if magic!=MAGIC or version!=1 or flags!=expected or bands!=self.model.bands:raise ValueError('model/header mismatch')
        streams=[];offset=HEADER.size
        for n in lengths:
            if n%4:raise ValueError('unaligned stack')
            streams.append(constriction.stream.stack.AnsCoder(np.frombuffer(payload[offset:offset+n],dtype=np.uint32)));offset+=n
        if offset!=len(payload):raise ValueError('invalid lengths')
        cc,bp,bg,bs=streams;groups=(length+7)//8
        q=np.zeros((groups,8),np.int32);c=np.zeros(groups,np.int32);known=np.zeros(groups,bool)
        for mask in self.masks(groups):
            (_,_,pc,_),common=self.fields(q,c,pi,known)
            if self.model.quantizer=='e8':
                c[mask]=cc.decode(constriction.stream.model.Bernoulli(perfect=False),np.ascontiguousarray(pc[mask],dtype=np.float32))
            for jj in [slice(0,7),slice(7,8)]:
                (p,s,_,g),_=self.fields(q,c,pi,known,common)
                pp=np.ascontiguousarray(p[mask,jj].ravel(),dtype=np.float32)
                sp=np.ascontiguousarray(s[mask,jj].ravel(),dtype=np.float32)
                gp=np.ascontiguousarray(g[mask,jj].ravel(),dtype=np.float32)
                nz=bp.decode(constriction.stream.model.Bernoulli(perfect=False),pp).astype(bool)
                values=np.zeros(len(pp),np.int32);nn=int(nz.sum())
                if nn:
                    sign=bg.decode(constriction.stream.model.Bernoulli(perfect=False),np.ascontiguousarray(gp[nz])).astype(bool)
                    mag=bs.decode(constriction.stream.model.QuantizedLaplace(0,maximum),np.zeros(nn,np.float32),np.ascontiguousarray(sp[nz])).astype(np.int32)
                    v=mag+1;v[sign]*=-1;values[nz]=v
                q[mask,jj]=values.reshape(-1,jj.stop-jj.start)
            known|=mask
        y=restore_doubled(c,q)*.5 if self.model.quantizer=='e8' else q.ravel()
        return idct(y*step,norm='ortho')[:length]
    def roundtrip(self,x,step):
        payload,expected,enc=self.encode(x,step);started=time.perf_counter();decoded=self.decode(payload)
        dec=1000*(time.perf_counter()-started)
        np.testing.assert_array_equal(decoded,expected)
        return dict(payload=payload,reconstructed=decoded,bps=len(payload)*8/len(x),encode_ms=enc,decode_ms=dec)
