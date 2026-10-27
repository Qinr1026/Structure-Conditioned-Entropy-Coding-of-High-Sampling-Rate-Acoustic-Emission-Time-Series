"""DCT, nearest E8 quantization, and reversible coordinate representation."""
import numpy as np
from scipy.fft import dct

def nearest_d8(values: np.ndarray) -> np.ndarray:
    """Nearest point of D8 = integer vectors with an even coordinate sum."""
    rounded = np.rint(values).astype(np.int32)
    odd = (rounded.sum(axis=-1) & 1).astype(bool)
    if np.any(odd):
        residual = values[odd] - rounded[odd]
        coordinate = np.argmax(np.abs(residual), axis=-1)
        row = np.arange(coordinate.size)
        correction = np.where(residual[row, coordinate] >= 0.0, 1, -1)
        rounded[odd, coordinate] += correction.astype(np.int32)
    return rounded


def quantize_e8(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest E8 point, represented exactly in doubled integer coordinates."""
    groups = values.reshape(-1, 8)
    integer = nearest_d8(groups)
    shifted_integer = nearest_d8(groups - 0.5)
    shifted = shifted_integer.astype(np.float64) + 0.5
    integer_error = np.sum((groups - integer) ** 2, axis=-1)
    shifted_error = np.sum((groups - shifted) ** 2, axis=-1)
    use_shifted = shifted_error < integer_error
    doubled = integer * 2
    doubled[use_shifted] = shifted_integer[use_shifted] * 2 + 1
    reconstructed = doubled.astype(np.float64) * 0.5
    return doubled, reconstructed.reshape(values.shape)

def coordinate_representation(doubled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(doubled, dtype=np.int64).reshape(-1, 8)
    coset = np.mod(groups[:, 0], 2).astype(np.int32)
    integer = (groups - coset[:, None]) // 2
    parity = np.mod(integer[:, :7].sum(axis=1), 2)
    if not np.array_equal(np.mod(integer[:, 7], 2), parity):
        raise RuntimeError("invalid E8 vector parity")
    coded = integer.copy()
    coded[:, 7] = (integer[:, 7] - parity) // 2
    return coset, coded.astype(np.int32)


def restore_doubled(coset: np.ndarray, coded: np.ndarray) -> np.ndarray:
    coded = np.asarray(coded, dtype=np.int64).reshape(-1, 8)
    integer = coded.copy()
    integer[:, 7] = 2 * coded[:, 7] + np.mod(coded[:, :7].sum(axis=1), 2)
    return (2 * integer + np.asarray(coset, dtype=np.int64)[:, None]).reshape(-1)

def symbols(x,step,quantizer):
 y=dct(np.asarray(x,dtype=np.float64),norm='ortho')/step
 if quantizer=='e8':
  doubled,_=quantize_e8(y);c,q=coordinate_representation(doubled)
 else:q=np.rint(y).astype(np.int32).reshape(-1,8);c=np.zeros(len(q),np.int32)
 return q,c
