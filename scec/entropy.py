"""Shared ANS support, sign and magnitude stream encoding."""
import numpy as np
import constriction

class ProgressiveNeuralE8ANS:
    @staticmethod
    def _ans_bytes(coder: object) -> bytes:
        return np.ascontiguousarray(coder.get_compressed(), dtype=np.uint32).tobytes()


    @staticmethod
    def _encode_coordinates(
        symbols: np.ndarray, probability: np.ndarray, scale: np.ndarray,
        sign_probability: np.ndarray,
    ) -> tuple[int, bytes, bytes, bytes]:
        symbols = np.asarray(symbols, dtype=np.int32).ravel()
        probability = np.asarray(probability, dtype=np.float32).ravel()
        scale = np.asarray(scale, dtype=np.float32).ravel()
        sign_probability = np.asarray(sign_probability, dtype=np.float32).ravel()
        support = symbols != 0
        support_coder = constriction.stream.stack.AnsCoder()
        support_coder.encode_reverse(
            support.astype(np.int32), constriction.stream.model.Bernoulli(perfect=False),
            np.ascontiguousarray(probability),
        )
        sign_coder = constriction.stream.stack.AnsCoder()
        if bool(support.any()):
            sign_coder.encode_reverse(
                (symbols[support] < 0).astype(np.int32),
                constriction.stream.model.Bernoulli(perfect=False),
                np.ascontiguousarray(sign_probability[support]),
            )
        magnitudes = np.maximum(np.abs(symbols[support]) - 1, 0).astype(np.int32)
        maximum = max(int(magnitudes.max(initial=0)), 1)
        magnitude_coder = constriction.stream.stack.AnsCoder()
        if magnitudes.size:
            magnitude_coder.encode_reverse(
                magnitudes, constriction.stream.model.QuantizedLaplace(0, maximum),
                np.zeros(magnitudes.size, dtype=np.float32),
                np.ascontiguousarray(scale[support]),
            )
        return (
            maximum, ProgressiveNeuralE8ANS._ans_bytes(support_coder),
            ProgressiveNeuralE8ANS._ans_bytes(sign_coder),
            ProgressiveNeuralE8ANS._ans_bytes(magnitude_coder),
        )
