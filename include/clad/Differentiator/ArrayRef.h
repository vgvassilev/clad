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

// NOLINTBEGIN(*-pointer-arithmetic)
template <typename T> class array_ref {
private:
  /// The pointer to the underlying array
  T* m_arr = nullptr;
  /// The size of the array
  std::size_t m_size = 0;

public:
  /// Delete default constructor
  array_ref() = default;
  /// Constructor to store the pointer to and size of an array supplied by the
  /// user
  constexpr CUDA_HOST_DEVICE array_ref(T* arr, std::size_t size)
      : m_arr(arr), m_size(size) {}
  /// Constructor for arrays having size equal to 1 or non pointer types to
  /// store their addresses
  constexpr CUDA_HOST_DEVICE array_ref(T* a) : m_arr(a), m_size(1) {}
  /// Constructor for clad::array types
  constexpr CUDA_HOST_DEVICE array_ref(array<T>& a)
      : m_arr(a.ptr()), m_size(a.size()) {}

  /// Operator for conversion from array_ref<T> to T*.
  constexpr CUDA_HOST_DEVICE operator T*() { return m_arr; }
  /// Operator for conversion from array_ref<T> to const T*.
  constexpr CUDA_HOST_DEVICE operator const T*() const { return m_arr; }

  template <typename U>
  CUDA_HOST_DEVICE array_ref<T>& operator=(const array<U>& a) {
    assert(m_size == a.size());
    for (std::size_t i = 0; i < m_size; ++i)
      m_arr[i] = a[i];
    return *this;
  }
  template <typename U>
  constexpr CUDA_HOST_DEVICE array_ref<T>& operator=(const array_ref<T>& a) {
    m_arr = a.ptr();
    m_size = a.size();
    return *this;
  }
  /// Returns the size of the underlying array
  constexpr CUDA_HOST_DEVICE std::size_t size() const { return m_size; }
  constexpr CUDA_HOST_DEVICE PUREFUNC T* ptr() const { return m_arr; }
  constexpr CUDA_HOST_DEVICE PUREFUNC T*& ptr_ref() { return m_arr; }
  /// Returns an array_ref to a part of the underlying array starting at
  /// offset and having the specified size
  constexpr CUDA_HOST_DEVICE array_ref<T> slice(std::size_t offset,
                                                std::size_t size) {
    assert((offset >= 0) && (offset + size <= m_size) &&
           "Window is outside array. Please provide an offset and size "
           "inside the array size.");
    return array_ref<T>(&m_arr[offset], size);
  }
  /// Returns the reference to the underlying array
  constexpr CUDA_HOST_DEVICE PUREFUNC T& operator*() { return *m_arr; }

  // Arithmetic overloads
  /// Divides the arrays element wise
  template <typename U>
  CUDA_HOST_DEVICE array_ref<T>& operator/=(array_ref<U>& Ar) {
    assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                  "for carrying out addition assignment");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= Ar[i];
    return *this;
  }
  /// Multiplies the arrays element wise
  template <typename U>
  CUDA_HOST_DEVICE array_ref<T>& operator*=(array_ref<U>& Ar) {
    assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                  "for carrying out addition assignment");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= Ar[i];
    return *this;
  }
  /// Adds the arrays element wise
  template <typename U>
  CUDA_HOST_DEVICE array_ref<T>& operator+=(array_ref<U>& Ar) {
    assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                  "for carrying out addition assignment");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += Ar[i];
    return *this;
  }
  /// Subtracts the arrays element wise
  template <typename U>
  CUDA_HOST_DEVICE array_ref<T>& operator-=(array_ref<U>& Ar) {
    assert(m_size == Ar.size() && "Size of both the array_refs must be equal "
                                  "for carrying out addition assignment");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= Ar[i];
    return *this;
  }
  /// Divides the elements of the array_ref by elements of the array
  template <typename U> CUDA_HOST_DEVICE array_ref<T>& operator/=(array<U>& A) {
    assert(m_size == A.size() && "Size of arrays must be equal");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= A[i];
    return *this;
  }
  /// Multiplies the elements of the array_ref by elements of the array
  template <typename U> CUDA_HOST_DEVICE array_ref<T>& operator*=(array<U>& A) {
    assert(m_size == A.size() && "Size of arrays must be equal");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= A[i];
    return *this;
  }
  /// Adds the elements of the array_ref by elements of the array
  template <typename U> CUDA_HOST_DEVICE array_ref<T>& operator+=(array<U>& A) {
    assert(m_size == A.size() && "Size of arrays must be equal");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += A[i];
    return *this;
  }
  /// Subtracts the elements of the array_ref by elements of the array
  template <typename U> CUDA_HOST_DEVICE array_ref<T>& operator-=(array<U>& A) {
    assert(m_size == A.size() && "Size of arrays must be equal");
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= A[i];
    return *this;
  }
  /// Divides the array by a scalar
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array_ref<T>& operator/=(U a) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= a;
    return *this;
  }
  /// Multiplies the array by a scalar
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array_ref<T>& operator*=(U a) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= a;
    return *this;
  }
  /// Adds the array by a scalar
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array_ref<T>& operator+=(U a) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += a;
    return *this;
  }

  /// Subtracts the array by a scalar
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array_ref<T>& operator-=(U a) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= a;
    return *this;
  }
};

/// Overloaded operators for clad::array_ref which returns an array
/// expression.

/// Multiplies the arrays element wise
template <typename T, typename U>
constexpr CUDA_HOST_DEVICE
    array_expression<const array_ref<T>&, BinaryMul, const array_ref<U>&>
    operator*(const array_ref<T>& Ar, const array_ref<U>& Br) {
  assert(Ar.size() == Br.size() &&
         "Size of both the array_refs must be equal for carrying out "
         "multiplication assignment");
  return array_expression<const array_ref<T>&, BinaryMul, const array_ref<U>&>(
      Ar, Br);
}

/// Adds the arrays element wise
template <typename T, typename U>
constexpr CUDA_HOST_DEVICE
    array_expression<const array_ref<T>&, BinaryAdd, const array_ref<U>&>
    operator+(const array_ref<T>& Ar, const array_ref<U>& Br) {
  assert(Ar.size() == Br.size() &&
         "Size of both the array_refs must be equal for carrying out addition "
         "assignment");
  return array_expression<const array_ref<T>&, BinaryAdd, const array_ref<U>&>(
      Ar, Br);
}

/// Subtracts the arrays element wise
template <typename T, typename U>
constexpr CUDA_HOST_DEVICE
    array_expression<const array_ref<T>&, BinarySub, const array_ref<U>&>
    operator-(const array_ref<T>& Ar, const array_ref<U>& Br) {
  assert(
      Ar.size() == Br.size() &&
      "Size of both the array_refs must be equal for carrying out subtraction "
      "assignment");
  return array_expression<const array_ref<T>&, BinarySub, const array_ref<U>&>(
      Ar, Br);
}

/// Divides the arrays element wise
template <typename T, typename U>
constexpr CUDA_HOST_DEVICE
    array_expression<const array_ref<T>&, BinaryDiv, const array_ref<U>&>
    operator/(const array_ref<T>& Ar, const array_ref<U>& Br) {
  assert(Ar.size() == Br.size() &&
         "Size of both the array_refs must be equal for carrying out division "
         "assignment");
  return array_expression<const array_ref<T>&, BinaryDiv, const array_ref<U>&>(
      Ar, Br);
}

/// Multiplies array_ref by a scalar
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<const array_ref<T>&, BinaryMul, U>
operator*(const array_ref<T>& Ar, U a) {
  return array_expression<const array_ref<T>&, BinaryMul, U>(Ar, a);
}

/// Multiplies array_ref by a scalar (reverse order)
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<const array_ref<T>&, BinaryMul, U>
operator*(U a, const array_ref<T>& Ar) {
  return array_expression<const array_ref<T>&, BinaryMul, U>(Ar, a);
}

/// Divides array_ref by a scalar
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<const array_ref<T>&, BinaryDiv, U>
operator/(const array_ref<T>& Ar, U a) {
  return array_expression<const array_ref<T>&, BinaryDiv, U>(Ar, a);
}

/// Adds array_ref by a scalar
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<const array_ref<T>&, BinaryAdd, U>
operator+(const array_ref<T>& Ar, U a) {
  return array_expression<const array_ref<T>&, BinaryAdd, U>(Ar, a);
}

/// Adds array_ref by a scalar (reverse order)
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<const array_ref<T>&, BinaryAdd, U>
operator+(U a, const array_ref<T>& Ar) {
  return array_expression<const array_ref<T>&, BinaryAdd, U>(Ar, a);
}

/// Subtracts array_ref by a scalar
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<const array_ref<T>&, BinarySub, U>
operator-(const array_ref<T>& Ar, U a) {
  return array_expression<const array_ref<T>&, BinarySub, U>(Ar, a);
}

/// Subtracts array_ref by a scalar (reverse order)
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
constexpr CUDA_HOST_DEVICE array_expression<U, BinarySub, const array_ref<T>&>
operator-(U a, const array_ref<T>& Ar) {
  return array_expression<U, BinarySub, const array_ref<T>&>(a, Ar);
}

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
    constexpr CUDA_HOST_DEVICE array_ref(T arr, std::size_t size = 1)
        : m_arr((void*)arr), m_size(size) {}
    template <typename T>
    constexpr CUDA_HOST_DEVICE array_ref(const array_ref<T>& other)
        : m_arr(other.ptr()), m_size(other.size()) {}
    template <typename T> constexpr CUDA_HOST_DEVICE operator array_ref<T>() {
      return array_ref<T>((T*)(m_arr), m_size);
    }
    [[nodiscard]] constexpr CUDA_HOST_DEVICE void* ptr() const { return m_arr; }
    [[nodiscard]] constexpr CUDA_HOST_DEVICE std::size_t size() const {
      return m_size;
    }
  };
  // NOLINTEND(*-pointer-arithmetic)
} // namespace clad

#endif // CLAD_ARRAY_REF_H
