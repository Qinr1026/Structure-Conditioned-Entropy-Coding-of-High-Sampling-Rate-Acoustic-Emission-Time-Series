// Minimal C ABI around the exact PocketFFT revision used by SciPy 1.15.3.
// The upstream BSD-3-Clause copyright and license are in pocketfft_hdronly.h.
#define POCKETFFT_CACHE_SIZE 64
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"
#include <cmath>
#include <cstddef>

#ifdef _WIN32
#define SCEC_EXPORT __declspec(dllexport)
#else
#define SCEC_EXPORT __attribute__((visibility("default")))
#endif

extern "C" SCEC_EXPORT int scec_dct(
    const double* input, double* output, size_t length, int type) noexcept {
  if (!input || !output || length == 0 || (type != 2 && type != 3)) return 1;
  try {
    // SciPy norm_fct<double>(1, 2*N), including long-double sqrt and rounding.
    double factor = double(1 / std::sqrt((long double)(2*length)));
    pocketfft::dct<double>({length}, {sizeof(double)}, {sizeof(double)}, {0},
                           type, input, output, factor, true, 1);
    return 0;
  } catch (...) { return 2; }
}
