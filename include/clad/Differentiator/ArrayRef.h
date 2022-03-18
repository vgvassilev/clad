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

    template <typename U>
    CUDA_HOST_DEVICE array_ref<T>& operator=(array<U>& a) {
      assert(m_size == a.size());
      for (std::size_t i = 0; i < m_size; ++i)
        m_arr[i] = a[i];
      return *this;
    }
    /// Returns the size of the underlying array
    CUDA_HOST_DEVICE std::size_t size() const { return m_size; }
    CUDA_HOST_DEVICE T* ptr() const { return m_arr; }
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

  /// `array_ref<void>` specialisation is created to be used as a placeholder
  /// type in the overloaded derived function. All `array_ref<T>` types are
  /// implicitly convertible to `array_ref<void>` type.
  ///
  /// `array_ref<void>` variables should be converted to the correct
  /// `array_ref<T>` type before being used. To make this process easier and
  /// more convenient, `array_ref<void>` provides implicit converter operators
  /// that facilitates convertion to `array_ref<T>` type using `static_cast`.
  template <> class array_ref<void> {
  private:
    /// The pointer to the underlying array
    void* m_arr = nullptr;
    /// The size of the array
    std::size_t m_size = 0;

  public:
    // delete the default constructor
    array_ref() = delete;
    // Here we are using C-style cast instead of `static_cast` because
    // we may also need to remove qualifiers (`const`, `volatile`, etc) while
    // converting to `void*` type.
    // We cannot create specialisation of `array_ref<void>` with qualifiers
    // (such as `array_ref<const void>`, `array_ref<volatile void>` etc) because
    // each derivative parameter has to be of the same type in the overloaded
    // gradient for the overloaded gradient mechanism to work and this class is
    // used as the placeholder type for the common derivative parameter type.
    template <typename T, class = typename std::enable_if<
                              std::is_pointer<T>::value ||
                              std::is_same<T, std::nullptr_t>::value>::type>
    CUDA_HOST_DEVICE array_ref(T arr, std::size_t size = 1)
        : m_arr((void*)arr), m_size(size) {}
    template <typename T>
    CUDA_HOST_DEVICE array_ref(const array_ref<T>& other)
        : m_arr(other.ptr()), m_size(other.size()) {}
    template <typename T> CUDA_HOST_DEVICE operator array_ref<T>() {
      return array_ref<T>((T*)(m_arr), m_size);
    }
    CUDA_HOST_DEVICE void* ptr() const { return m_arr; }
    CUDA_HOST_DEVICE std::size_t size() const { return m_size; }
  };
} // namespace clad

#endif // CLAD_ARRAY_REF_H
