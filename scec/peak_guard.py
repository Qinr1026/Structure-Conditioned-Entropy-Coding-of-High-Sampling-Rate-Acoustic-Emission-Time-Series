"""Optional pointwise error protection for integer ADC waveforms."""
import struct
import numpy as np
import zstandard as zstd

HEADER = struct.Struct('<4sB3xdQIIII')


class PeakGuard:
    def __init__(self, codec, compression_level=9):
        self.codec = codec
        self.compressor = zstd.ZstdCompressor(level=compression_level)

    def encode(self, samples, step, threshold):
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError('threshold must be finite and nonnegative')
        # The base API validates integer range and shape before conversion.
        base = self.codec.encode(samples, step)
        x = np.asarray(samples, dtype=np.int16)
        reconstructed = self.codec.decode(base)
        ids = np.flatnonzero(np.abs(x.astype(np.float64)-reconstructed) > threshold)
        indices = values = b''
        if ids.size:
            gaps = np.diff(np.concatenate(([0], ids))).astype('<u4')
            indices = self.compressor.compress(gaps.tobytes())
            values = self.compressor.compress(x[ids].astype('<i2').tobytes())
        return HEADER.pack(b'E8PG', 1, threshold, len(x), len(base), len(ids), len(indices), len(values))+base+indices+values

    def decode(self, payload):
        if len(payload) < HEADER.size:
            raise ValueError('truncated protected stream')
        magic, version, threshold, n, nb, count, ni, nv = HEADER.unpack_from(payload)
        if magic != b'E8PG' or version != 1 or not np.isfinite(threshold) or threshold < 0 or n < 1 or count > n:
            raise ValueError('invalid protected stream header')
        if HEADER.size+nb+ni+nv != len(payload):
            raise ValueError('invalid protected stream lengths')
        offset = HEADER.size
        y = self.codec.decode(payload[offset:offset+nb]); offset += nb
        if len(y) != n:
            raise ValueError('sample-count mismatch')
        if count:
            decompress = zstd.ZstdDecompressor().decompress
            ib = decompress(payload[offset:offset+ni], max_output_size=4*count); offset += ni
            vb = decompress(payload[offset:offset+nv], max_output_size=2*count)
            if len(ib) != 4*count or len(vb) != 2*count:
                raise ValueError('invalid correction payload size')
            ids = np.cumsum(np.frombuffer(ib, dtype='<u4'), dtype=np.uint64)
            if ids[-1] >= n or np.any(ids[1:] <= ids[:-1]):
                raise ValueError('invalid correction indices')
            y[ids.astype(np.int64)] = np.frombuffer(vb, dtype='<i2')
        elif ni or nv:
            raise ValueError('unexpected correction data')
        return y
