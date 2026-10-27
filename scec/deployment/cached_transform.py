"""Optional bounded PocketFFT plan cache for unchanged SCEC transforms.

This is a deployment optimization, not a new transform or codec model.
The pinned upstream header and reproducible build script are distributed with it.
"""
from pathlib import Path
import ctypes
import os
import sys
import numpy as np
from scipy.fft import dct,idct
from scec.deployment.pinned_graph import SCECPinnedGraphCodec
from scec.deployment.native_graph import restore_coefficients
from scec.deployment.quantized_fast import nearest_coordinates


class CachedDCT:
    def __init__(self, library=None):
        if library is None:
            suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
            library = os.environ.get('SCEC_NATIVE_LIBRARY') or Path(__file__).resolve().parents[1]/'native'/('cached_dct'+suffix)
        if not Path(library).is_file():
            raise FileNotFoundError('Build the optional native adapter with python tools/build_native.py, or use the reference backend.')
        self.library=ctypes.CDLL(str(library))
        self.function=self.library.scec_dct
        self.function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
        self.function.restype=ctypes.c_int

    def transform(self,x,inverse=False):
        x=np.ascontiguousarray(x,dtype=np.float64)
        if x.ndim!=1 or not len(x):raise ValueError('expected a nonempty 1D array')
        # Bound per-plan size as well as plan count. Long streams retain SciPy.
        if len(x)>262144:return (idct if inverse else dct)(x,norm='ortho')
        y=np.empty_like(x)
        result=self.function(x.ctypes.data,y.ctypes.data,len(x),3 if inverse else 2)
        if result:raise RuntimeError(f'PocketFFT transform failed: {result}')
        return y


class SCECCachedTransformCodec(SCECPinnedGraphCodec):
    def __init__(self,model,**kwargs):
        super().__init__(model,**kwargs)
        self.transform=CachedDCT()

    def prepare_transforms(self,lengths):
        """Prepare length-only FFT plans using zero placeholders, no signals."""
        for length in lengths:
            if length<1:raise ValueError('empty record length')
            padded=(int(length)+7)//8*8
            if padded<=262144:self.transform.transform(np.zeros(padded,np.float64))

    def quantize(self,padded,step):
        if self.model.quantizer!='e8':return super().quantize(padded,step)
        return nearest_coordinates(self.transform.transform(padded)/step)

    def _restore(self,q,c,step,length):
        if self.model.quantizer!='e8':return super()._restore(q,c,step,length)
        return self.transform.transform(restore_coefficients(q,c,step),inverse=True)[:length]
