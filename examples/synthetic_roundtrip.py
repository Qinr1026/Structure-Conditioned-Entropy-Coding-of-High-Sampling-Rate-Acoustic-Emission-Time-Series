"""Small synthetic training and roundtrip example; not a performance benchmark."""
import argparse
import copy
import numpy as np
import torch
from scipy.fft import idct
from scec import SCEC
from scec.training import train_model
from scec.quantization import symbols, restore_doubled
from scec.peak_guard import PeakGuard


def synthetic_blocks(count, length, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    result = []
    for _ in range(count):
        f = rng.uniform(.02, .23)
        phase = rng.uniform(0, 2*np.pi)
        x = 900*np.sin(2*np.pi*f*t+phase) + 100*rng.normal(size=length)
        result.append(np.rint(x).astype(np.int16))
    return np.stack(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    parser.add_argument('--backend', default='reference', choices=['reference','compiled','graph','native'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    train = synthetic_blocks(8, 256, 71)
    val = synthetic_blocks(3, 256, 72)
    model, _ = train_model(train, val, [80., 240.], bands=8, epochs=1, batch_size=4, device=args.device)
    encoder = SCEC(model, args.backend)
    decoder = SCEC(copy.deepcopy(model), args.backend)
    x = synthetic_blocks(1, 513, 73)[0]
    encoder.prepare([len(x)], 'encode')
    decoder.prepare([len(x)], 'decode')
    payload = encoder.encode(x, 240.)
    y = decoder.decode(payload)
    padded = np.pad(x, (0, (-len(x)) % 8), mode='reflect')
    q, c = symbols(padded, 240., 'e8')
    expected = idct(restore_doubled(c, q)*.5*240., norm='ortho')[:len(x)]
    np.testing.assert_allclose(y, expected, rtol=0, atol=1e-10)
    protected = PeakGuard(encoder).encode(x, 240., 0.)
    np.testing.assert_array_equal(PeakGuard(decoder).decode(protected), x)
    print('Synthetic roundtrip passed; exact quantized reconstruction and zero-threshold protection checked.')
    print('No input signals, trained weights, or compressed payloads were saved.')


if __name__ == '__main__':
    main()
