"""Exact CPU symbol packing plus bounded CUDA-graph probability execution."""
import numpy as np
import constriction
from scipy.fft import idct
from scec.deployment.quantized_fast import njit
from scec.deployment.cuda_graph import SCECCUDAGraphCodec
from scec.entropy import ProgressiveNeuralE8ANS
from scec.reference import HEADER,MAGIC


@njit(cache=True,nogil=True)
def pack_symbols(q,c,fields):
    groups=len(c);total=8*groups
    cs=np.empty(groups,np.int32);cp=np.empty(groups,np.float32)
    nz=np.empty(total,np.int32);pp=np.empty(total,np.float32)
    signs=np.empty(total,np.int32);mags=np.empty(total,np.int32)
    sp=np.empty(total,np.float32);gp=np.empty(total,np.float32)
    cursor=0;nonzero=0;offset=0;maximum=1
    for start,stride in ((0,8),(4,8),(2,4),(1,2)):
        n=max(0,(groups-start+stride-1)//stride)
        for k in range(n):
            cs[offset+k]=c[start+k*stride];cp[offset+k]=fields[offset+k,0]
        for first,last in ((0,7),(7,8)):
            for k in range(n):
                i=start+k*stride
                for j in range(first,last):
                    value=q[i,j];present=value!=0
                    nz[cursor]=int(present);pp[cursor]=fields[offset+k,1+j];cursor+=1
                    if present:
                        magnitude=max(abs(value)-1,0)
                        signs[nonzero]=int(value<0);mags[nonzero]=magnitude
                        sp[nonzero]=fields[offset+k,9+j];gp[nonzero]=fields[offset+k,17+j]
                        nonzero+=1;maximum=max(maximum,magnitude)
        offset+=n
    return cs,cp,nz,pp,signs[:nonzero],mags[:nonzero],sp[:nonzero],gp[:nonzero],maximum


@njit(cache=True,nogil=True)
def select_probabilities(fields,c,q,ids,last):
    width=1 if last else 7;total=len(ids)*width
    p=np.empty(total,np.float32);s=np.empty(total,np.float32);g=np.empty(total,np.float32)
    for k in range(len(ids)):
        i=ids[k];state=2*c[i]
        if last:
            parity=0
            for j in range(7):parity+=q[i,j]
            state+=parity%2
        for jj in range(width):
            j=7 if last else jj;source=8*state+j;target=k*width+jj
            p[target]=fields[k,1+source];s[target]=fields[k,33+source];g[target]=fields[k,65+source]
    return p,s,g


@njit(cache=True,nogil=True)
def compact_probabilities(nz,s,g):
    size=int(nz.sum());sp=np.empty(size,np.float32);gp=np.empty(size,np.float32)
    count=0
    for i in range(len(nz)):
        if nz[i]:sp[count]=s[i];gp[count]=g[i];count+=1
    return sp,gp


@njit(cache=True,nogil=True)
def scatter_coordinates(q,ids,nz,signs,mags,last):
    width=1 if last else 7;cursor=0;nonzero=0
    for k in range(len(ids)):
        for jj in range(width):
            j=7 if last else jj;value=0
            if nz[cursor]:
                value=(mags[nonzero]+1)*(-1 if signs[nonzero] else 1);nonzero+=1
            q[ids[k],j]=value;cursor+=1


@njit(cache=True,nogil=True)
def restore_coefficients(q,c,step):
    values=np.empty(q.size,np.float64)
    for i in range(len(c)):
        parity=0
        for j in range(7):parity+=q[i,j]
        parity%=2
        for j in range(8):
            integer=np.int64(q[i,j])
            if j==7:integer=2*integer+parity
            doubled=2*integer+np.int64(c[i])
            # Preserve the two original floating-point operations, no fastmath.
            values[8*i+j]=(np.float64(doubled)*.5)*step
    return values


class SCECNativeGraphCodec(SCECCUDAGraphCodec):
    def finish(self,record,packed):
        cs,cp,nz,pp,signs,mags,sp,gp,maximum=pack_symbols(record.q,record.c,packed)
        bernoulli=constriction.stream.model.Bernoulli(perfect=False)
        cc=constriction.stream.stack.AnsCoder();bp=constriction.stream.stack.AnsCoder()
        bg=constriction.stream.stack.AnsCoder();bs=constriction.stream.stack.AnsCoder()
        if self.model.quantizer=='e8':cc.encode_reverse(cs,bernoulli,cp)
        bp.encode_reverse(nz,bernoulli,pp)
        if len(mags):
            bg.encode_reverse(signs,bernoulli,gp)
            bs.encode_reverse(mags,constriction.stream.model.QuantizedLaplace(0,maximum),np.zeros(len(mags),np.float32),sp)
        streams=[ProgressiveNeuralE8ANS._ans_bytes(coder) for coder in [cc,bp,bg,bs]]
        flags=int(self.model.quantizer=='e8')+2*int(self.model.neural)+4*int(self.model.conditioned)
        return HEADER.pack(MAGIC,1,record.profile,flags,self.model.bands,record.length,record.step,maximum,*map(len,streams))+b''.join(streams)

    def _decode_stage(self,packed,c,q,ids,bp,bg,bs,bernoulli,laplace):
        for last in (False,True):
            pp,sp,gp=select_probabilities(packed,c,q,ids,last)
            nz=bp.decode(bernoulli,pp)
            sp,gp=compact_probabilities(nz,sp,gp)
            signs=np.empty(0,np.int32);mags=np.empty(0,np.int32)
            if len(sp):
                signs=bg.decode(bernoulli,gp)
                mags=bs.decode(laplace,np.zeros(len(sp),np.float32),sp)
            scatter_coordinates(q,ids,nz,signs,mags,last)

    def _restore(self,q,c,step,length):
        if self.model.quantizer!='e8':return super()._restore(q,c,step,length)
        return idct(restore_coefficients(q,c,step),norm='ortho')[:length]
