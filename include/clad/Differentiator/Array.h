#ifndef CLAD_ARRAY_H
#define CLAD_ARRAY_H

#include "clad/Differentiator/CladConfig.h"

#include <assert.h>
#include <initializer_list>
#include <type_traits>

namespace clad {
template <typename T> class array_ref;

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

  template <typename U>
  CUDA_HOST_DEVICE array(clad::array_ref<U> arr)
      : m_arr(new T[arr.size()]{static_cast<T>(T())}), m_size(arr.size()) {
    (*this) = arr;
  }

  template <typename U>
  CUDA_HOST_DEVICE array(U* a, std::size_t size)
      : m_arr(new T[size]{static_cast<T>(T())}), m_size(size) {
    for (std::size_t i = 0; i < size; ++i)
      m_arr[i] = a[i];
  }

  CUDA_HOST_DEVICE array(const array<T>& arr) : array(arr.m_arr, arr.m_size) {}

  CUDA_HOST_DEVICE array(std::size_t size, const clad::array<T>& arr)
      : m_arr(new T[size]{static_cast<T>(T())}), m_size(size) {
    for (std::size_t i = 0; i < size; ++i)
      m_arr[i] = arr[i];
  }

  // initializing all entries using the same value
  template <typename U>
  CUDA_HOST_DEVICE array(std::size_t size, U val)
      : m_arr(new T[size]{static_cast<T>(val)}), m_size(size) {}

  CUDA_HOST_DEVICE array(std::initializer_list<T> arr)
      : m_arr(new T[arr.size()]{static_cast<T>(T())}), m_size(arr.size()) {
    std::size_t i = 0;
    for (const auto& e : arr)
      m_arr[i++] = e;
  }

  CUDA_HOST_DEVICE array<T>& operator=(const array<T>& arr) {
    (*this) = arr.m_arr;
    return *this;
  }

  /// Destructor to delete the array if it was created by array_ref
  CUDA_HOST_DEVICE ~array() { delete[] m_arr; }

  /// Returns the size of the underlying array
  CUDA_HOST_DEVICE std::size_t size() const { return m_size; }
  /// Returns the ptr of the underlying array
  CUDA_HOST_DEVICE T* ptr() { return m_arr; }
  /// Returns the reference to the location at the index of the underlying
  /// array
  CUDA_HOST_DEVICE T& operator[](std::ptrdiff_t i) { return m_arr[i]; }
  /// Returns the reference to the underlying array
  CUDA_HOST_DEVICE T& operator*() { return *m_arr; }

  // Arithmetic overloads
  /// Divides the number from every element in the array
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array<T>& operator/=(U n) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= n;
    return *this;
  }
  /// Multiplies the number to every element in the array
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array<T>& operator*=(U n) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= n;
    return *this;
  }
  /// Adds the number to every element in the array
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array<T>& operator+=(U n) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += n;
    return *this;
  }
  /// Subtracts the number from every element in the array
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  CUDA_HOST_DEVICE array<T>& operator-=(U n) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= n;
    return *this;
  }
  /// Initializes the clad::array to the given array
  CUDA_HOST_DEVICE array<T>& operator=(T* arr) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr ? arr[i] : 0;
    return *this;
  }
  /// Performs element wise addition
  CUDA_HOST_DEVICE array<T>& operator+=(T* arr) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += arr[i];
    return *this;
  }
  /// Performs element wise subtraction
  CUDA_HOST_DEVICE array<T>& operator-=(T* arr) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= arr[i];
    return *this;
  }
  /// Performs element wise multiplication
  CUDA_HOST_DEVICE array<T>& operator*=(T* arr) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= arr[i];
    return *this;
  }
  /// Performs element wise division
  CUDA_HOST_DEVICE array<T>& operator/=(T* arr) {
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= arr[i];
    return *this;
  }
  /// Initializes the clad::array to the given clad::array_ref
  CUDA_HOST_DEVICE array<T>& operator=(array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr[i];
    return *this;
  }

  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator=(array_ref<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr[i];
    return *this;
  }

  /// Performs element wise addition
  CUDA_HOST_DEVICE array<T>& operator+=(array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += arr[i];
    return *this;
  }
  /// Performs element wise subtraction
  CUDA_HOST_DEVICE array<T>& operator-=(array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= arr[i];
    return *this;
  }
  /// Performs element wise multiplication
  CUDA_HOST_DEVICE array<T>& operator*=(array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= arr[i];
    return *this;
  }
  /// Performs element wise division
  CUDA_HOST_DEVICE array<T>& operator/=(array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= arr[i];
    return *this;
  }
  /// Initializes the clad::array to the given clad::array
  CUDA_HOST_DEVICE array<T>& operator=(array<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr[i];
    return *this;
  }
  /// Performs element wise addition
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator+=(array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise subtraction
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator-=(array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise multiplication
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator*=(array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise division
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator/=(array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= static_cast<T>(arr[i]);
    return *this;
  }

  /// Negate the array and return a new array.
  CUDA_HOST_DEVICE array<T> operator-() const {
    array<T> arr2(m_size);
    for (std::size_t i = 0; i < m_size; i++)
      arr2[i] = -m_arr[i];
    return arr2;
  }

  /// Subtracts the number from every element in the array and returns a new
  /// array, when the number is on the left side.
  CUDA_HOST_DEVICE friend array<T> operator-(T n, const array<T>& arr) {
    size_t size = arr.size();
    array<T> arr2(size);
    for (std::size_t i = 0; i < size; i++)
      arr2[i] = n - arr[i];
    return arr2;
  }

  /// Implicitly converts from clad::array to pointer to an array of type T
  CUDA_HOST_DEVICE operator T*() const { return m_arr; }
}; // class array

/// Overloaded operators for clad::array which return a new array.

/// Multiplies the number to every element in the array and returns a new
/// array.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE array<T> operator*(const array<T>& arr, U n) {
  array<T> arr2(arr);
  arr2 *= n;
  return arr2;
}

/// Multiplies the number to every element in the array and returns a new
/// array, when the number is on the left side.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE array<T> operator*(U n, const array<T>& arr) {
  return arr * n;
}

/// Divides the number from every element in the array and returns a new
/// array
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE array<T> operator/(const array<T>& arr, U n) {
  array<T> arr2(arr);
  arr2 /= n;
  return arr2;
}

/// Adds the number to every element in the array and returns a new array
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE array<T> operator+(const array<T>& arr, U n) {
  array<T> arr2(arr);
  arr2 += n;
  return arr2;
}

/// Adds the number to every element in the array and returns a new array,
/// when the number is on the left side.
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE array<T> operator+(U n, const array<T>& arr) {
  return arr + n;
}

/// Subtracts the number from every element in the array and returns a new
/// array
template <typename T, typename U,
          typename std::enable_if<std::is_arithmetic<U>::value, int>::type = 0>
CUDA_HOST_DEVICE array<T> operator-(const array<T>& arr, U n) {
  array<T> arr2(arr);
  arr2 -= n;
  return arr2;
}

/// Function to define element wise adding of two arrays.
template <typename T>
CUDA_HOST_DEVICE array<T> operator+(const array<T>& arr1,
                                    const array<T>& arr2) {
  assert(arr1.size() == arr2.size());
  array<T> arr(arr1);
  arr += arr2;
  return arr;
}

/// Function to define element wise subtraction of two arrays.
template <typename T>
CUDA_HOST_DEVICE array<T> operator-(const array<T>& arr1,
                                    const array<T>& arr2) {
  assert(arr1.size() == arr2.size());
  array<T> arr(arr1);
  arr -= arr2;
  return arr;
}

} // namespace clad

#endif // CLANG_ARRAY_H
