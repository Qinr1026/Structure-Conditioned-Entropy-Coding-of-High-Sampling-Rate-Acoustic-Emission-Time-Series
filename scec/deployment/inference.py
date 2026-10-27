"""SCEC deployment path. Same weights, probabilities, symbols and SCE8 stream.

The reference path also computes two training-rate diagnostics and an encoder
IDCT. Neither is required to emit a complete unprotected compressed stream.
No waveform, probability output or content-dependent result is cached.
"""
import time
import numpy as np
import torch
import constriction
from torch.nn import functional as F
from scec.context import E8DecoderContextPrior
from scec.entropy import ProgressiveNeuralE8ANS
from scec.reference import StateCodec, HEADER, MAGIC


class SCECCodec(StateCodec):
    def __init__(self, model):
        super().__init__(model)
        self._steps = model.steps.detach().cpu().numpy().copy()
        self._frame_key = None
        self._frame = None

    def _shared_fields(self, profile, groups, key):
        # One bounded cache, containing only model/profile/length quantities.
        # New lengths/profiles are prepared inside the measured encode/decode.
        if self._frame_key == key:
            return self._frame
        m = self.model
        p, s, pc = m.base_fields(profile, groups)
        g = m.base_sign_fields(profile, groups)
        pl = torch.logit(p.clamp(1e-4, 1-1e-4))
        sl = torch.log(s.clamp_min(1e-4))
        cl = torch.logit(pc.clamp(1e-4, 1-1e-4))
        gl = torch.logit(g.clamp(1e-4, 1-1e-4))
        if groups == m.groups:
            frequency, resonance = m.normalized_frequency, m.resonance_features
        else:
            frequency = (torch.arange(groups, dtype=torch.float32, device=self.device)*8.0+3.5)/(groups*8.0)
            centers = torch.tensor(m.resonance_centers_normalized, dtype=torch.float32, device=self.device)
            resonance = torch.exp(-0.5*((frequency[:,None]-centers[None,:])/0.018)**2).view(1,groups,5)
            frequency = frequency.view(1,groups,1)
        step = ((torch.log(m.steps[profile])-torch.log(m.steps).mean()) /
                torch.log(m.steps).std(unbiased=False).clamp_min(1e-6)).view(1,1,1).expand(-1,groups,-1)
        self._frame_key = key
        self._frame = (p,s,pc,g,pl,sl,cl,gl,frequency,resonance,step)
        return self._frame

    def _common(self, q, c, profile, known, key):
        m=self.model;groups=q.shape[1]
        p,s,pc,g,pl,sl,cl,gl,freq,res,step=self._shared_fields(profile,groups,key)
        if not m.neural or not known.any():return p,s,pc,g
        mask=torch.from_numpy(known).to(self.device).view(1,groups,1)
        mask_values=mask.to(q.dtype)
        normalized=torch.sign(q.float())*torch.log1p(q.abs().float())/4.0
        features=torch.cat([normalized*mask_values,(q!=0).float()*mask_values,
            c.float().unsqueeze(-1)*mask_values,mask_values,pl,sl,cl.unsqueeze(-1),
            freq,res,step],dim=-1)
        h=F.gelu(m.input_projection(features.transpose(1,2)))
        for block in m.blocks:h=block(h)
        correction=m.output_projection(h).transpose(1,2)
        nonanchor=(~mask).float()
        outp=torch.sigmoid(pl+nonanchor*2.5*torch.tanh(correction[...,:8])).clamp(1e-4,1-1e-4)
        outs=s*torch.exp(nonanchor*2.2*torch.tanh(correction[...,8:16]))
        outc=torch.sigmoid(cl+(~mask[...,0]).float()*2.5*torch.tanh(correction[...,16])).clamp(1e-4,1-1e-4)
        outg=torch.sigmoid(gl+nonanchor*2.5*torch.tanh(correction[...,17:25])).clamp(1e-4,1-1e-4)
        return outp,outs.clamp_min(.03),outc,outg

    @torch.inference_mode()
    def fields(self, q, c, pi, known, common=None):
        qt = torch.from_numpy(q).unsqueeze(0).to(self.device)
        ct = torch.from_numpy(c).unsqueeze(0).to(self.device)
        pt = torch.tensor([pi], device=self.device)
        if common is None:
            common = self._common(qt, ct, pt, known, (len(c),pi))
        fields = self.model.condition(common, qt, ct, pt)
        return [v[0].cpu().numpy() for v in fields], common

    @torch.inference_mode()
    def encode_bytes(self, x, step):
        """Encode an in-memory waveform into the complete stream (the probability model is shared separately)."""
        x = np.asarray(x, dtype=np.float64)
        length = len(x)
        padded = np.pad(x, (0, (-length) % 8), mode='reflect' if length > 1 else 'edge')
        pi = int(np.argmin(abs(self._steps - step)))
        if not np.isclose(float(self._steps[pi]), step):
            raise ValueError('unknown step')
        q, c = self.quantize(padded, step)
        known = np.zeros(len(c), bool)
        cs, cps, qs, ps, ss, gs = [], [], [], [], [], []
        for mask in self.masks(len(c)):
            (p, s, pc, g), _ = self.fields(q, c, pi, known)
            cs.append(c[mask]); cps.append(pc[mask])
            for jj in [slice(0, 7), slice(7, 8)]:
                qs.append(q[mask, jj].ravel()); ps.append(p[mask, jj].ravel())
                ss.append(s[mask, jj].ravel()); gs.append(g[mask, jj].ravel())
            known |= mask
        coder = constriction.stream.stack.AnsCoder()
        if self.model.quantizer == 'e8':
            coder.encode_reverse(np.concatenate(cs),
                constriction.stream.model.Bernoulli(perfect=False),
                np.ascontiguousarray(np.concatenate(cps), dtype=np.float32))
        cb = ProgressiveNeuralE8ANS._ans_bytes(coder)
        maximum, bp, bg, bs = ProgressiveNeuralE8ANS._encode_coordinates(
            np.concatenate(qs), np.concatenate(ps), np.concatenate(ss), np.concatenate(gs))
        streams = [cb, bp, bg, bs]
        flags = int(self.model.quantizer == 'e8') + 2 * int(self.model.neural) + 4 * int(self.model.conditioned)
        header = HEADER.pack(MAGIC, 1, pi, flags, self.model.bands, length,
                             float(step), maximum, *map(len, streams))
        return header + b''.join(streams)

    def roundtrip(self, x, step):
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        payload = self.encode_bytes(x, step)
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        enc = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        y = self.decode(payload)
        if self.device.type == 'cuda': torch.cuda.synchronize(self.device)
        dec = (time.perf_counter() - start) * 1000
        return dict(payload=payload, reconstructed=y, bps=8*len(payload)/len(x),
                    encode_ms=enc, decode_ms=dec)
