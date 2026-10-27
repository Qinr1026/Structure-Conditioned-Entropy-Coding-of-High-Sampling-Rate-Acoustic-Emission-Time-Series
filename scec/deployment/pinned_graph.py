"""Reuse pinned output staging, preserving every probability and stream byte."""
import numpy as np
import torch
from scipy.fft import dct, idct
from scec.deployment.cuda_graph import GraphEntry
from scec.deployment.native_graph import SCECNativeGraphCodec, restore_coefficients
from scec.deployment.quantized_fast import nearest_coordinates


class PinnedOutputEntry(GraphEntry):
    def __init__(self, codec, groups, role):
        super().__init__(codec, groups, role)
        self.host_outputs = [None if x is None else torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
                             for x in self.outputs]

    def predict(self, q, c, pi, stage=0):
        self.host_array[:8*self.groups] = q.ravel()
        self.host_array[8*self.groups:] = c
        self.symbols.copy_(self.host, non_blocking=True)
        self.profile.copy_(self.codec._profile_tensor(pi))
        self.graphs[stage].replay()
        target = self.host_outputs[stage]
        target.copy_(self.outputs[stage], non_blocking=True)
        torch.cuda.current_stream(self.codec.device).synchronize()
        result = target.numpy()
        # Encoder results may be consumed by an asynchronous finish() worker.
        return result.copy() if self.role == 'encode' else result


class SCECPinnedGraphCodec(SCECNativeGraphCodec):
    def _entry(self, groups, role):
        if self.device.type != 'cuda' or groups > self.max_graph_groups:
            return None
        cache = self._graphs[role]
        if groups not in cache:
            if len(cache) >= self.max_graph_shapes:
                return None
            cache[groups] = PinnedOutputEntry(self, groups, role)
        return cache[groups]
