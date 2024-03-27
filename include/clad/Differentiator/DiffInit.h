#ifndef CLAD_DIFFINIT_H
#define CLAD_DIFFINIT_H

namespace clad {
/// The purpose of this function is to initialize adjoints
/// (or all of its differentiable fields) with 0.
// FIXME: Add support for objects.
template <typename T> CUDA_HOST_DEVICE void zero_init(T& x) { new (&x) T(); }
template <typename T> CUDA_HOST_DEVICE void zero_init(T* x, std::size_t N) {
  for (std::size_t i = 0; i < N; ++i)
    zero_init(x[i]);
}
template <typename T, std::size_t N>
CUDA_HOST_DEVICE void zero_init(T (&arr)[N]) {
  zero_init((T*)arr, N);
}
} // namespace clad

#endif // CLAD_DIFFINIT_H
