"""Bounded CUDA-graph deployment of unchanged SCEC probability arithmetic.

Graph keys depend only on padded record length and encode/decode role. Every
replay overwrites symbols and profile; no waveform predictions are cached.
Graph preparation is explicit deployment overhead (or paid on first use).
Use one calling thread per codec and capture in a dedicated CUDA process.
"""
import weakref
import numpy as np
import torch
import constriction
from scipy.fft import idct
from scec.quantization import restore_doubled
from scec.reference import HEADER,MAGIC
from scec.deployment.fast_io import SCECFastIOCodec,device_layout


class GraphEntry:
    def __init__(self,codec,groups,role):
        self.groups=groups;self.role=role;self.codec=weakref.proxy(codec)
        self.symbols=torch.zeros(9*groups,dtype=torch.int32,device=codec.device)
        self.q=self.symbols[:8*groups].view(1,groups,8)
        self.c=self.symbols[8*groups:].view(1,groups)
        self.profile=torch.zeros(1,dtype=torch.long,device=codec.device)
        self.host=torch.empty(9*groups,dtype=torch.int32,pin_memory=True)
        self.host_array=self.host.numpy()
        self.indices,self.masks=device_layout(self.q)
        m=codec.model
        if groups==m.groups:
            self.frequency,self.resonance=m.normalized_frequency,m.resonance_features
        else:
            f=(torch.arange(groups,dtype=torch.float32,device=codec.device)*8.+3.5)/(groups*8.)
            centers=torch.tensor(m.resonance_centers_normalized,dtype=torch.float32,device=codec.device)
            self.resonance=torch.exp(-.5*((f[:,None]-centers[None,:])/.018)**2).view(1,groups,5)
            self.frequency=f.view(1,groups,1)
        self.graphs=[];self.outputs=[];self.base=None
        for stage in ([-1] if role=='encode' else range(4)):
            if stage>=0 and self.indices[stage].numel()==0:
                self.graphs.append(None);self.outputs.append(None);continue
            # Warming up on a side stream is required before capture.
            stream=torch.cuda.Stream(device=codec.device)
            stream.wait_stream(torch.cuda.current_stream(codec.device))
            with torch.cuda.stream(stream):
                for _ in range(3):output=self.run(stage)
            torch.cuda.current_stream(codec.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=codec._graph_pool if codec.share_graph_pool else None):output=self.run(stage)
            if codec.share_graph_pool and codec._graph_pool is None:codec._graph_pool=graph.pool()
            self.graphs.append(graph);self.outputs.append(output)
            # Populate captured base tensors before subsequent decode stages.
            graph.replay()
            torch.cuda.synchronize(codec.device)

    def fields(self):
        m=self.codec.model;p,s,pc=m.base_fields(self.profile,self.groups)
        g=m.base_sign_fields(self.profile,self.groups)
        pl=torch.logit(p.clamp(1e-4,1-1e-4));sl=torch.log(s.clamp_min(1e-4))
        cl=torch.logit(pc.clamp(1e-4,1-1e-4));gl=torch.logit(g.clamp(1e-4,1-1e-4))
        step=((torch.log(m.steps[self.profile])-torch.log(m.steps).mean())/
              torch.log(m.steps).std(unbiased=False).clamp_min(1e-6)).view(1,1,1).expand(-1,self.groups,-1)
        return [p,s,pc,g,pl,sl,cl,gl,self.frequency,self.resonance,step]

    def run(self,stage):
        if stage<0:return self.codec.device_encoder(self.q,self.c,self.profile,self.fields())
        if stage==0:self.base=self.fields()
        return self.codec.decode_net(self.q,self.c,self.profile,self.base,
                                     self.indices[stage],self.masks[stage],stage>0)

    def predict(self,q,c,pi,stage=0):
        self.host_array[:8*self.groups]=q.ravel()
        self.host_array[8*self.groups:]=c
        self.symbols.copy_(self.host,non_blocking=True)
        self.profile.copy_(self.codec._profile_tensor(pi))
        self.graphs[stage].replay()
        # This synchronization also makes it safe to refill the pinned input.
        return self.outputs[stage].cpu().numpy()


class SCECCUDAGraphCodec(SCECFastIOCodec):
    def __init__(self,model,max_graph_shapes=32,max_graph_groups=32768,share_graph_pool=True):
        super().__init__(model)
        if max_graph_shapes<0 or max_graph_groups<1:raise ValueError('invalid graph capacity')
        self.max_graph_shapes=max_graph_shapes;self.max_graph_groups=max_graph_groups
        # Decode stages replay in capture order; records never replay concurrently.
        # Cross-record outputs are copied to CPU before another graph can overwrite
        # private-pool scratch. Persistent symbols live outside the private pool.
        self.share_graph_pool=share_graph_pool;self._graph_pool=None
        self._graphs={'encode':{},'decode':{}}

    def _entry(self,groups,role):
        if self.device.type!='cuda' or groups>self.max_graph_groups:return None
        cache=self._graphs[role]
        if groups not in cache:
            if len(cache)>=self.max_graph_shapes:return None
            cache[groups]=GraphEntry(self,groups,role)
        return cache[groups]

    def clear_graphs(self):
        """Release graph references after outstanding work has completed."""
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
        self._graphs={'encode':{},'decode':{}}
        self._graph_pool=None

    @torch.inference_mode()
    def prepare_graphs(self,lengths,roles=('encode','decode')):
        """Prepare shapes using zero placeholders, without access to signals."""
        for role in roles:
            if role not in self._graphs:raise ValueError('unknown graph role')
            for length in lengths:
                if length<1:raise ValueError('empty record length')
                self._entry((int(length)+7)//8,role)

    @torch.inference_mode()
    def predict(self,record):
        entry=self._entry(len(record.c),'encode')
        if entry is None:return super().predict(record)
        return entry.predict(record.q,record.c,record.profile)

    @torch.inference_mode()
    def decode(self,payload):
        if len(payload)<HEADER.size:raise ValueError('truncated header')
        magic,version,pi,flags,bands,length,step,maximum,*lengths=HEADER.unpack_from(payload)
        expected=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        if magic!=MAGIC or version!=1 or flags!=expected or bands!=self.model.bands:raise ValueError('model/header mismatch')
        if length<1 or pi>=len(self._steps) or not np.isclose(step,float(self._steps[pi])):raise ValueError('invalid profile or length')
        if HEADER.size+sum(lengths)!=len(payload) or any(n%4 for n in lengths):raise ValueError('invalid lengths')
        groups=(length+7)//8;entry=self._entry(groups,'decode')
        if entry is None:return super().decode(payload)
        streams=[];offset=HEADER.size
        for n in lengths:
            streams.append(constriction.stream.stack.AnsCoder(np.frombuffer(payload[offset:offset+n],dtype=np.uint32)));offset+=n
        cc,bp,bg,bs=streams
        q=np.zeros((groups,8),np.int32);c=np.zeros(groups,np.int32)
        index=np.arange(groups);masks=[index%8==0,index%8==4,index%4==2,index%2==1]
        bernoulli=constriction.stream.model.Bernoulli(perfect=False)
        laplace=constriction.stream.model.QuantizedLaplace(0,maximum)
        for stage,mask in enumerate(masks):
            ids=np.flatnonzero(mask);n=len(ids)
            if n==0:continue
            packed=entry.predict(q,c,pi,stage)
            if self.model.quantizer=='e8':c[ids]=cc.decode(bernoulli,np.ascontiguousarray(packed[:,0]))
            self._decode_stage(packed,c,q,ids,bp,bg,bs,bernoulli,laplace)
        return self._restore(q,c,step,length)

    def _decode_stage(self,packed,c,q,ids,bp,bg,bs,bernoulli,laplace):
        n=len(ids);fields=[packed[:,i:i+32].reshape(n,4,8) for i in [1,33,65]]
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

    def _restore(self,q,c,step,length):
        y=restore_doubled(c,q)*.5 if self.model.quantizer=='e8' else q.ravel()
        return idct(y*step,norm='ortho')[:length]
