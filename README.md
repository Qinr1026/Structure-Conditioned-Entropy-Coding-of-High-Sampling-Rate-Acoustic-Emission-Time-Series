# SCEC

Source code for **Structure-Conditioned Entropy Coding** of complete acoustic emission waveforms.

SCEC applies a global orthonormal DCT and nearest-E8 quantization, then codes the resulting symbols with a four-stage neural context model and ANS. Within each E8 vector, the decoded coset conditions the first seven coordinates; their recovered parity additionally conditions the eighth-coordinate quotient. The decoder obtains each conditioning variable before using it.

This repository contains algorithm source only. It includes no experimental signals, reconstructed signals, compressed signal payloads, pretrained weights, fitted probability tables, experiment results, plotting scripts, figures, manuscripts, or reviewer correspondence. The example generates mathematical inputs in memory. Users must train a model on their own data or supply a compatible model separately. Paper performance is not reproducible from this source-only release alone.

## Installation

Use Python 3.10 or newer in an isolated environment. From this repository root:

```shell
python -m pip install -e .
python -m examples.synthetic_roundtrip
python -m unittest discover -s tests -v
```

The pinned NumPy, SciPy, PyTorch and constriction versions match the research environment. The default example runs on CPU. For CUDA, install a PyTorch build suitable for your GPU before installing this package. No dependency, dataset or model is downloaded by the algorithm at import time.

## Train on your own data

Prepare two separate local `.npy` arrays, each of shape `(number_of_blocks, samples_per_block)`, with integer ADC values in the signed 16-bit range. Split by acquisition condition before creating blocks. Choose a block length divisible by eight times the number of frequency bands. Do not use test records for prior fitting, neural training, or strength selection.

```shell
python -m examples.train_own_data --train /path/to/train.npy --validation /path/to/validation.npy --output /path/to/new_model --bands 64 --epochs 12 --device cpu
```

The generic trainer fits priors on training blocks, trains the context network with AdamW, selects the neural checkpoint by estimated validation rate, and then chooses the structural-offset strength on validation blocks. It is a configurable algorithm example, not a replay of the paper's archived multi-stage curriculum. Block caches and model outputs must stay outside the source release.

The context model includes five fixed Gaussian frequency features. Their defaults are implementation hyperparameters, not proof of physical resonances or universally suitable frequencies. Training on a new acquisition setup requires evaluating the feature configuration. Estimated training rate and emitted file length are distinct quantities.

## Encode and decode

```python
import numpy as np
from scec import SCEC, load_model

encoder_model, _ = load_model('/path/to/new_model', device='cpu')
decoder_model, _ = load_model('/path/to/new_model', device='cpu')
encoder = SCEC(encoder_model, backend='reference')
decoder = SCEC(decoder_model, backend='reference')

# Placeholder input, generated locally; replace with your own ADC samples.
x = np.zeros(1024, dtype=np.int16)
payload = encoder.encode(x, step=900.)  # step must exist in the model
reconstructed = decoder.decode(payload)
```

The result is lossy unless a zero-threshold protection layer is requested. Encoder and decoder must share the same model, profile configuration and compatible floating-point execution environment. The 37-byte stream header does not embed model weights or their identity. `load_model` verifies the external checkpoint checksum; users must retain the corresponding model configuration with their archives. Cross-hardware/precision bitstream portability has not been established.

Optional pointwise protection stores original int16 values only at locations whose base reconstruction error exceeds a chosen threshold:

```python
from scec.peak_guard import PeakGuard
protected = PeakGuard(encoder).encode(x, step=900., threshold=10.)
reconstructed = PeakGuard(decoder).decode(protected)
```

## Accelerated execution

```shell
python -m pip install -e ".[fast]"
python -m examples.synthetic_roundtrip --device cuda --backend graph
```

| Backend | Implementation | Requirements |
|---|---|---|
| `reference` | Direct neural inference and ANS | CPU or CUDA |
| `compiled` | TorchScript fields and compiled nearest-E8 kernel | Numba; CPU or CUDA |
| `graph` | CUDA graphs and compiled symbol processing | CUDA and Numba |
| `native` | Compact probability transfers, pinned buffers and cached DCT/IDCT plans | CUDA, Numba and the native adapter |

Build the optional native adapter from source using a GCC/Clang C++ compiler:

```shell
python tools/build_native.py --cxx g++
python -m examples.synthetic_roundtrip --device cuda --backend native
```

Windows was used for the research implementation. The build script also provides Unix compiler flags, but other platforms require their own validation. No DLL, executable, compiler or runtime binary is distributed. An alternate adapter location can be supplied through `SCEC_NATIVE_LIBRARY`.

For graph/native backends, call `encoder.prepare(lengths, 'encode')` and `decoder.prepare(lengths, 'decode')` before repeated execution. Preparation uses zero placeholders and length-dependent constants. It has startup and memory costs. Graphs have a bounded shape cache; long or uncached shapes use the implementation's fallback path. The native transform cache has 64 plans and a 262,144-sample limit. Use separate encoder/decoder instances, freeze the model before construction, and call each instance from one thread. TorchScript fusion settings are process-global, so use a dedicated codec process. Performance depends on hardware, record lengths, preparation and warmup; this release makes no generic throughput guarantee.

## Source map

| Location | Purpose |
|---|---|
| `scec/quantization.py` | DCT, nearest E8 points and reversible integer coordinates |
| `scec/context.py`, `scec/model.py` | Neural context and structural state probabilities |
| `scec/priors.py`, `scec/training.py` | Training-only prior fitting and generic neural training |
| `scec/schedule.py`, `scec/entropy.py`, `scec/reference.py` | Causal schedule, ANS and stream format |
| `scec/codec.py`, `scec/bundle.py` | Public API and caller-owned model serialization |
| `scec/peak_guard.py` | Optional maximum-error protection |
| `scec/deployment/`, `scec/native/` | Compiled inference, graph execution and transform cache |
| `examples/`, `tests/` | Generated-input examples and correctness checks |

## Release and licensing

Only this repository's source files should be uploaded. `.gitignore` excludes common signal, model, figure, document, result and binary formats; review any future additions before committing them.

No project-wide license has been selected for the authors' code. The bundled PocketFFT header retains its own BSD-3-Clause notice; see `THIRD_PARTY_NOTICES.txt`. Third-party licensing does not assign a license to the SCEC authors' code.
