#ifndef CLAD_TAPE_H
#define CLAD_TAPE_H

#include "clad/Differentiator/ArrayRef.h"
#include "clad/Differentiator/CladConfig.h"
#include <cassert>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace clad {
namespace vector_tape {
/// Dynamically-sized array (std::vector-like), primarily used for storing
/// values in reverse-mode AD inside loops.
template <typename T> class tape_impl {
  std::vector<T> data;

public:
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  tape_impl() {
    data.reserve(64);
  }

  /// Add new value of type T constructed from args to the end of the tape.
  template <typename... ArgsT>
  CUDA_HOST_DEVICE void emplace_back(ArgsT&&... args) {
    data.emplace_back(std::forward<ArgsT>(args)...);
  }

  CUDA_HOST_DEVICE std::size_t size() const { return data.size(); }

  CUDA_HOST_DEVICE iterator begin() { return data.begin(); }

  CUDA_HOST_DEVICE const_iterator begin() const { return data.cbegin(); }

  CUDA_HOST_DEVICE iterator end() { return data.end(); }

  CUDA_HOST_DEVICE const_iterator end() const { return data.cend(); }

  /// Access last value (must not be empty).
  CUDA_HOST_DEVICE reference back() {
    assert(!data.empty());
    return data.back();
  }

  CUDA_HOST_DEVICE const_reference back() const {
    assert(!data.empty());
    return data.back();
  }

  CUDA_HOST_DEVICE reference operator[](std::size_t i) {
    assert(i < data.size());
    return data[i];
  }

  CUDA_HOST_DEVICE const_reference operator[](std::size_t i) const {
    assert(i < data.size());
    return data[i];
  }

  /// Remove the last value from the tape.
  CUDA_HOST_DEVICE void pop_back() {
    assert(!data.empty());
    data.pop_back();
  }
};

/// Tape type used for storing values in reverse-mode AD inside loops.
template <typename T> using tape = tape_impl<T>;

/// Add value to the end of the tape, return the same value.
template <typename T, typename... ArgsT>
CUDA_HOST_DEVICE T push(tape<T>& to, ArgsT&&... val) {
  to.emplace_back(std::forward<ArgsT>(val)...);
  return to.back();
}

/// Add value to the end of the tape, return the same value.
/// A specialization for clad::array_ref types to use in reverse mode.
template <typename T, typename U>
CUDA_HOST_DEVICE clad::array_ref<T> push(tape<clad::array_ref<T>>& to, U val) {
  to.emplace_back(val);
  return val;
}

/// Remove the last value from the tape, return it.
template <typename T> CUDA_HOST_DEVICE T pop(tape<T>& to) {
  T val = std::move(to.back());
  to.pop_back();
  return val;
}

/// Access return the last value in the tape.
template <typename T> CUDA_HOST_DEVICE T& back(tape<T>& of) {
  return of.back();
}
} // namespace vector_tape
} // namespace clad
#endif // CLAD_TAPE_H
