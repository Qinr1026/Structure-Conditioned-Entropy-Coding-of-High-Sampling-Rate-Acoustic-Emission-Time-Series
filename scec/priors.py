"""Estimate pooled and structural priors from caller-supplied training blocks."""
import numpy as np
from .quantization import symbols

def state(q,c,bands,conditional):
 b=np.minimum(np.arange(len(q))*bands//len(q),bands-1)
 states=np.zeros(q.shape,dtype=np.int64)
 if conditional:
  states+=c[:,None]*2;states[:,7]+=q[:,:7].sum(1)%2
 return b,states


def fit(blocks,steps,quantizer,bands=64):
 shape=(len(steps),bands,4,8);tot=np.zeros(shape);nz=np.zeros(shape);mag=np.zeros(shape);neg=np.zeros(shape)
 cos=np.zeros((len(steps),bands));ct=np.zeros_like(cos)
 for pi,step in enumerate(steps):
  for x in blocks:
   q,c=symbols(x,step,quantizer);b,st=state(q,c,bands,True);idx=(pi,b[:,None],st,np.arange(8)[None,:])
   np.add.at(tot,idx,1);np.add.at(nz,idx,q!=0);np.add.at(mag,idx,np.maximum(abs(q)-1,0));np.add.at(neg,idx,q<0)
   np.add.at(cos,(pi,b),c);np.add.at(ct,(pi,b),1)
 # Hierarchical shrinkage to pooled probabilities (16 pseudo observations).
 pooled=lambda a:a.sum(axis=2,keepdims=True)
 pp=(pooled(nz)+.5)/(pooled(tot)+1);ss=(pooled(mag)+.5)/(pooled(nz)+1)+.2;sg=(pooled(neg)+.5)/(pooled(nz)+1)
 p=(nz+16*pp)/(tot+16);s=(mag+16*ss)/(nz+16)+.2;g=(neg+16*sg)/(nz+16)
 return dict(support=np.clip(p,1e-5,1-1e-5),scale=np.maximum(s,.08),sign=np.clip(g,1e-5,1-1e-5),pooled_support=pp,pooled_scale=ss,pooled_sign=sg,coset=(cos+.5)/(ct+1),steps=steps,bands=bands)
