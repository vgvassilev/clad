//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_TAPE_AND_PULLBACK_H
#define CLAD_TAPE_AND_PULLBACK_H

#include "clad/Differentiator/BuiltinDerivatives.h"
#ifdef __CUDACC__
#include "clad/Differentiator/BuiltinDerivativesCUDA.cuh"
#endif
#include "clad/Differentiator/Tape.h"

#include <algorithm>
#include <cstddef>
#include <utility>
#ifndef __CUDACC__
#include <mutex>
#endif

namespace clad {

/// Tape type used for storing values in reverse-mode AD inside loops.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool is_multithread = false, bool DiskOffload = false>
using tape = tape_impl<T, SBO_SIZE, SLAB_SIZE, is_multithread, DiskOffload>;

/// Add value to the end of the tape, return the same value.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false, typename... ArgsT>
CUDA_HOST_DEVICE T&
push(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& to,
     ArgsT... val) {
  to.emplace_back(std::forward<ArgsT>(val)...);
  return to.back();
}

/// A specialization for C arrays
template <typename T, typename U, size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
CUDA_HOST_DEVICE void
push(tape<T[N], SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& to,
     const U& val) {
  to.emplace_back();
  std::copy(std::begin(val), std::end(val), std::begin(to.back()));
}

/// Remove the last value from the tape, return it.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
CUDA_HOST_DEVICE T
pop(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& to) {
  T val = std::move(to.back());
  to.pop_back();
  return val;
}

/// A specialization for C arrays
template <typename T, std::size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
CUDA_HOST_DEVICE void pop(tape<T[N], SBO_SIZE, SLAB_SIZE,
                               /*is_multithread=*/false, DiskOffload>& to) {
  to.pop_back();
}

/// Access return the last value in the tape.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
CUDA_HOST_DEVICE T&
back(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithread=*/false, DiskOffload>& of) {
  return of.back();
}

/// Thread safe tape access functions with mutex locking mechanism
#ifndef __CUDACC__
/// Add value to the end of the tape, return the same value.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false, typename... ArgsT>
T push(tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& to,
       ArgsT&&... val) {
  std::lock_guard<std::mutex> lock(to.mutex());
  to.emplace_back(std::forward<ArgsT>(val)...);
  return to.back();
}

/// A specialization for C arrays
template <typename T, typename U, size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
void push(
    tape<T[N], SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& to,
    const U& val) {
  std::lock_guard<std::mutex> lock(to.mutex());
  to.emplace_back();
  std::copy(std::begin(val), std::end(val), std::begin(to.back()));
}

/// Remove the last value from the tape, return it.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
T pop(
    tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& to) {
  std::lock_guard<std::mutex> lock(to.mutex());
  T val = std::move(to.back());
  to.pop_back();
  return val;
}

/// A specialization for C arrays
template <typename T, std::size_t N, std::size_t SBO_SIZE = 64,
          std::size_t SLAB_SIZE = 1024, bool DiskOffload = false>
void pop(tape<T[N], SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true,
              DiskOffload>& to) {
  std::lock_guard<std::mutex> lock(to.mutex());
  to.pop_back();
}

/// Access return the last value in the tape.
template <typename T, std::size_t SBO_SIZE = 64, std::size_t SLAB_SIZE = 1024,
          bool DiskOffload = false>
T& back(
    tape<T, SBO_SIZE, SLAB_SIZE, /*is_multithreaded=*/true, DiskOffload>& of) {
  std::lock_guard<std::mutex> lock(of.mutex());
  return of.back();
}
#endif

} // namespace clad

#endif // CLAD_TAPE_AND_PULLBACK_H
