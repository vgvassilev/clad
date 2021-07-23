#ifndef CLANG_ARRAY_H
#define CLANG_ARRAY_H

#include "clad/Differentiator/CladConfig.h"

#include <type_traits>

namespace clad {
  /// This class is not meant to be used by user. It is used by clad internally
  /// only
  template <typename T> class array {
  private:
    /// The pointer to the underlying array
    T* m_arr = nullptr;
    /// The size of the array
    std::size_t m_size = 0;

  public:
    /// Delete default constructor
    array() = delete;
    /// Constructor to create an array of the specified size
    CUDA_HOST_DEVICE array(std::size_t size)
        : m_arr(new T[size]{static_cast<T>(0)}), m_size(size) {}

    /// Destructor to delete the array if it was created by array_ref
    CUDA_HOST_DEVICE ~array() { delete[] m_arr; }

    /// Returns the size of the underlying array
    CUDA_HOST_DEVICE std::size_t size() { return m_size; }
    /// Returns the ptr of the underlying array
    CUDA_HOST_DEVICE T* ptr() { return m_arr; }
    /// Returns the reference to the location at the index of the underlying
    /// array
    CUDA_HOST_DEVICE T& operator[](std::size_t i) { return m_arr[i]; }
    /// Returns the reference to the underlying array
    CUDA_HOST_DEVICE T& operator*() { return *m_arr; }

    // Arithmetic overloads
    /// Divides the number from every element in the array
    CUDA_HOST_DEVICE array<T>& operator/=(T n) {
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] /= n;
      return *this;
    }
    /// Multiplies the number to every element in the array
    CUDA_HOST_DEVICE array<T>& operator*=(T n) {
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] *= n;
      return *this;
    }
    /// Adds the number to every element in the array
    CUDA_HOST_DEVICE array<T>& operator+=(T n) {
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] += n;
      return *this;
    }
    /// Subtracts the number from every element in the array
    CUDA_HOST_DEVICE array<T>& operator-=(T n) {
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] -= n;
      return *this;
    }
  };
} // namespace clad

#endif // CLANG_ARRAY_H
