"""Public, single-record SCEC API with explicit execution backends."""
import numpy as np
from .reference import HEADER, MAGIC


class SCEC:
    """Encode/decode using an externally shared, frozen probability model.

    Use separate encoder and decoder instances on the same numerical backend.
    Each instance is single-caller; do not mutate its model after construction.
    """
    def __init__(self, model, backend='reference', **backend_options):
        if backend == 'reference':
            from .deployment.inference import SCECCodec
            cls = SCECCodec
        elif backend == 'compiled':
            from .deployment.fast_io import SCECFastIOCodec
            cls = SCECFastIOCodec
        elif backend == 'graph':
            from .deployment.native_graph import SCECNativeGraphCodec
            cls = SCECNativeGraphCodec
        elif backend == 'native':
            from .deployment.compact_graph import SCECCompactGraphCodec
            cls = SCECCompactGraphCodec
        else:
            raise ValueError('backend must be reference, compiled, graph, or native')
        if backend in ('graph', 'native') and next(model.parameters()).device.type != 'cuda':
            raise ValueError('graph/native backends require a CUDA model; use reference on CPU')
        self.implementation = cls(model, **backend_options)
        self.model = model
        self.backend = backend

    def encode(self, samples, step):
        x = np.asarray(samples)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError('samples must be a nonempty, finite 1D array')
        if x.min() < -32768 or x.max() > 32767 or not np.array_equal(x, np.rint(x)):
            raise ValueError('the public API expects signed 16-bit integer ADC values')
        steps = self.model.steps.detach().cpu().numpy()
        pi = int(np.argmin(abs(steps - step)))
        if not np.isfinite(step) or not np.isclose(step, float(steps[pi])):
            raise ValueError('step is not supported by the shared model')
        # Use the shared float32 profile value on both sides of the stream.
        return self.implementation.encode_bytes(x, float(steps[pi]))

    def decode(self, payload):
        if len(payload) < HEADER.size:
            raise ValueError('truncated SCEC header')
        magic, version, pi, flags, bands, length, step, maximum, *lengths = HEADER.unpack_from(payload)
        expected = int(self.model.quantizer == 'e8') + 2*int(self.model.neural) + 4*int(self.model.conditioned)
        if magic != MAGIC or version != 1 or flags != expected or bands != self.model.bands:
            raise ValueError('model/header mismatch')
        if length < 1 or pi >= self.model.profiles or maximum < 1:
            raise ValueError('invalid sample count, profile, or magnitude bound')
        if not np.isfinite(step) or not np.isclose(step, float(self.model.steps[pi])):
            raise ValueError('profile/step mismatch')
        if HEADER.size + sum(lengths) != len(payload) or any(n % 4 for n in lengths):
            raise ValueError('invalid ANS stream lengths')
        return self.implementation.decode(payload)

    def prepare(self, lengths, role):
        """Optional length-only graph/FFT preparation, excluded from encoding."""
        if role not in ('encode', 'decode'):
            raise ValueError('role must be encode or decode')
        lengths = tuple(int(n) for n in lengths)
        if any(n < 1 for n in lengths):
            raise ValueError('lengths must be positive')
        impl = self.implementation
        if hasattr(impl, 'prepare_graphs'):
            impl.prepare_graphs(lengths, roles=(role,))
        if hasattr(impl, 'prepare_transforms'):
            impl.prepare_transforms(lengths)
