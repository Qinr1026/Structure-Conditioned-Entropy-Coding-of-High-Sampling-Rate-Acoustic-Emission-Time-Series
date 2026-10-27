"""Compiled execution of the unchanged nearest-E8 rule, without fastmath."""
import numpy as np
from scipy.fft import dct
from numba import njit
from scec.deployment.scripted import SCECScriptedCodec


@njit(cache=True,nogil=True)
def nearest_coordinates(values):
    x=values.reshape(-1,8);n=len(x)
    q=np.empty((n,8),np.int32);c=np.empty(n,np.int32)
    a=np.empty(8,np.int32);b=np.empty(8,np.int32)
    ea=np.empty(8,np.float64);eb=np.empty(8,np.float64)
    for i in range(n):
        sa=0;sb=0;ja=0;jb=0;ra=-1.;rb=-1.
        for j in range(8):
            a[j]=int(np.rint(x[i,j]));b[j]=int(np.rint(x[i,j]-.5))
            sa+=a[j];sb+=b[j]
            da=abs(x[i,j]-a[j]);db=abs((x[i,j]-.5)-b[j])
            if da>ra:ra=da;ja=j
            if db>rb:rb=db;jb=j
        if sa%2:a[ja]+=1 if x[i,ja]-a[ja]>=0 else -1
        if sb%2:b[jb]+=1 if (x[i,jb]-.5)-b[jb]>=0 else -1
        for j in range(8):
            da=x[i,j]-a[j];db=x[i,j]-(float(b[j])+.5)
            ea[j]=da*da;eb[j]=db*db
        # NumPy's contiguous eight-element pairwise reduction order.
        aa=((ea[0]+ea[1])+(ea[2]+ea[3]))+((ea[4]+ea[5])+(ea[6]+ea[7]))
        bb=((eb[0]+eb[1])+(eb[2]+eb[3]))+((eb[4]+eb[5])+(eb[6]+eb[7]))
        use=bb<aa;c[i]=int(use);s=0
        for j in range(8):
            q[i,j]=b[j] if use else a[j]
            if j<7:s+=q[i,j]
        q[i,7]=(q[i,7]-(s%2))//2
    return q,c


class SCECQuantizedFastCodec(SCECScriptedCodec):
    def quantize(self,padded,step):
        if self.model.quantizer!='e8':return super().quantize(padded,step)
        return nearest_coordinates(dct(np.asarray(padded,dtype=np.float64),norm='ortho')/step)
