#ifndef CLAD_ARRAY_REF_H
#define CLAD_ARRAY_REF_H

#include "clad/Differentiator/Array.h"
#include "clad/Differentiator/CladConfig.h"

#include <assert.h>
#include <type_traits>

namespace clad {
  /// Stores the pointer to and the size of an array and provides some helper
  /// functions for it. The array is supplied should have a life greater than
  /// that of the array_ref
  template <typename T> class array_ref {
  private:
    /// The pointer to the underlying array
    T* m_arr = nullptr;
    /// The size of the array
    std::size_t m_size = 0;

  public:
    /// Delete default constructor
    array_ref() = delete;
    /// Constructor to store the pointer to and size of an array supplied by the
    /// user
    CUDA_HOST_DEVICE array_ref(T* arr, std::size_t size)
        : m_arr(arr), m_size(size) {}
    /// Constructor for arrays having size equal to 1 or non pointer types to
    /// store their addresses
    CUDA_HOST_DEVICE array_ref(T* a) : m_arr(a), m_size(1) {}
    /// Constructor for clad::array types
    CUDA_HOST_DEVICE array_ref(array<T>& a)
        : m_arr(a.ptr()), m_size(a.size()) {}

    /// Returns the size of the underlying array
    CUDA_HOST_DEVICE std::size_t size() { return m_size; }
    /// Returns an array_ref to a part of the underlying array starting at
    /// offset and having the specified size
    CUDA_HOST_DEVICE array_ref<T> slice(std::size_t offset, std::size_t size) {
      assert((offset >= 0) && (offset + size <= m_size) &&
             "Window is outside array. Please provide an offset and size "
             "inside the array size.");
      return array_ref<T>(&m_arr[offset], size);
    }
    /// Returns the reference to the location at the index of the underlying
    /// array
    CUDA_HOST_DEVICE T& operator[](std::size_t i) { return m_arr[i]; }
    /// Returns the reference to the underlying array
    CUDA_HOST_DEVICE T& operator*() { return *m_arr; }

    // Arithmetic overloads
    /// Divides the arrays element wise
    CUDA_HOST_DEVICE array_ref<T>& operator/=(array_ref<T>& Ar) {
      assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                    "for carrying out addition assignment");
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] /= Ar[i];
      return *this;
    }
    /// Multiplies the arrays element wise
    CUDA_HOST_DEVICE array_ref<T>& operator*=(array_ref<T>& Ar) {
      assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                    "for carrying out addition assignment");
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] *= Ar[i];
      return *this;
    }
    /// Adds the arrays element wise
    CUDA_HOST_DEVICE array_ref<T>& operator+=(array_ref<T>& Ar) {
      assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                    "for carrying out addition assignment");
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] += Ar[i];
      return *this;
    }
    /// Subtracts the arrays element wise
    CUDA_HOST_DEVICE array_ref<T>& operator-=(array_ref<T>& Ar) {
      assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                    "for carrying out addition assignment");
      for (std::size_t i = 0; i < m_size; i++)
        m_arr[i] -= Ar[i];
      return *this;
    }
  };
} // namespace clad

#endif // CLAD_ARRAY_REF_H
