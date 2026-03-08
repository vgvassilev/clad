// This file defines configuration that is shared by various files

#ifndef CLAD_CONFIG_H
#define CLAD_CONFIG_H

#if __cplusplus >= 201402L
#define CLAD_CONSTEXPR_CXX14 constexpr
#else
#define CLAD_CONSTEXPR_CXX14 inline
#endif

#include <cstdlib>
#include <memory>

namespace clad {
// The aim is to have a single unsigned integer for storing both, the
// differentiation order and the options. The differentiation order is
// stored in the lower 8 bits, and the options are stored in the upper
// 24 bits. The differentiation order is stored in the lower 8 bits
// thus allowing for a maximum differentiation order of 2^8 - 1 = 255.
constexpr unsigned ORDER_BITS = 8;
constexpr unsigned ORDER_MASK = (1 << ORDER_BITS) - 1;

enum order {
  first = 1,
  second = 2,
  third = 3,
}; // enum order

enum opts : unsigned {
  use_enzyme = 1 << ORDER_BITS,
  vector_mode = 1 << (ORDER_BITS + 1),

  // Storing two bits for tbr analysis.
  // 00 - default, 01 - enable, 10 - disable, 11 - not used / invalid
  enable_tbr = 1 << (ORDER_BITS + 2),
  disable_tbr = 1 << (ORDER_BITS + 3),
  enable_va = 1 << (ORDER_BITS + 5),
  disable_va = 1 << (ORDER_BITS + 6),
  enable_ua = 1 << (ORDER_BITS + 9),
  disable_ua = 1 << (ORDER_BITS + 10),

  // Specifying whether we only want the diagonal of the hessian.
  diagonal_only = 1 << (ORDER_BITS + 4),

  // Specify that we need a constexpr-enabled CladFunction
  immediate_mode = 1 << (ORDER_BITS + 7),
}; // enum opts

constexpr unsigned GetDerivativeOrder(const unsigned bitmasked_opts) {
  return bitmasked_opts & ORDER_MASK;
}

constexpr bool HasOption(const unsigned bitmasked_opts, const unsigned option) {
  return (bitmasked_opts & option) == option;
}

constexpr unsigned GetBitmaskedOpts() { return 0; }
constexpr unsigned GetBitmaskedOpts(const unsigned first) { return first; }
template <typename... Opts>
constexpr unsigned GetBitmaskedOpts(const unsigned first, Opts... opts) {
  return first | GetBitmaskedOpts(opts...);
}

} // namespace clad

// Define CUDA_HOST_DEVICE attribute for adding CUDA support to
// clad functions
#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// Define trap function that is a CUDA compatible replacement for
// exit(int code) function
#ifdef __CUDACC__
inline __host__ __device__ void trap(int code) {
#ifdef __CUDA_ARCH__
  // Device code path
  asm("trap;");
#else
  // Host code path
  exit(code);
#endif
}
#else
inline void trap(int code) { exit(code); }
#endif

#ifdef  __CUDACC__
template <typename T> __host__ __device__ T* clad_addressof(T& r) {
#ifdef __CUDA_ARCH__
  return __builtin_addressof(r);
#else
  return std::addressof(r);
#endif
}
#else
template<typename T>
T* clad_addressof(T& r) {
  return std::addressof(r);
}
#endif

#endif // CLAD_CONFIG_H
