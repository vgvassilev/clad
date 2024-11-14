#ifndef CLAD_ARRAY_H
#define CLAD_ARRAY_H

#include "clad/Differentiator/ArrayExpression.h"
#include "clad/Differentiator/CladConfig.h"

#include <assert.h>
#include <initializer_list>
#include <type_traits>

namespace clad {
template <typename T> class array_ref;

// For MSVC, __attribute__((pure)) is not supported.
#if defined(_MSC_VER)
#define PUREFUNC
#else
#define PUREFUNC __attribute__((pure))
#endif

/// This class is not meant to be used by user. It is used by clad internally
/// only

// NOLINTBEGIN(*-pointer-arithmetic)
template <typename T> class array {
private:
  /// The pointer to the underlying array
  T* m_arr = nullptr;
  /// The size of the array
  std::size_t m_size = 0;

public:
  /// Default constructor
  array() = default;
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
      : m_arr(new T[size]), m_size(size) {
    for (std::size_t i = 0; i < size; ++i)
      m_arr[i] = static_cast<T>(a[i]);
  }

  CUDA_HOST_DEVICE array(const array<T>& arr) : array(arr.m_arr, arr.m_size) {}

  template <typename U>
  CUDA_HOST_DEVICE array(const array<U>& arr)
      : m_arr(new T[arr.size()]), m_size(arr.size()) {
    (*this) = arr;
  }

  CUDA_HOST_DEVICE array(std::size_t size, const clad::array<T>& arr)
      : m_arr(new T[size]), m_size(size) {
    for (std::size_t i = 0; i < size; ++i)
      m_arr[i] = arr[i];
  }

  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array(std::size_t size,
                         const array_expression<L, BinaryOp, R>& expression)
      : m_arr(new T[size]), m_size(size) {
    for (std::size_t i = 0; i < size; ++i)
      m_arr[i] = expression[i];
  }

  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array(const array_expression<L, BinaryOp, R>& expression)
      : m_arr(new T[expression.size()]), m_size(expression.size()) {
    for (std::size_t i = 0; i < expression.size(); ++i)
      m_arr[i] = expression[i];
  }

  // initializing all entries using the same value
  template <typename U>
  CUDA_HOST_DEVICE array(std::size_t size, U val)
      : m_arr(new T[size]), m_size(size) {
    for (std::size_t i = 0; i < size; ++i)
      m_arr[i] = static_cast<T>(val);
  }

  CUDA_HOST_DEVICE array(std::initializer_list<T> arr)
      : m_arr(new T[arr.size()]{static_cast<T>(T())}), m_size(arr.size()) {
    std::size_t i = 0;
    for (const auto& e : arr)
      m_arr[i++] = e;
  }

  CUDA_HOST_DEVICE array<T>& operator=(const array<T>& arr) {
    if (m_size < arr.m_size) {
      delete[] m_arr;
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      m_arr = new T[arr.m_size];
      m_size = arr.m_size;
    }
    (*this) = arr.m_arr;
    return *this;
  }

  CUDA_HOST_DEVICE array<T> slice(std::size_t offset, std::size_t size) const {
    assert(offset + size <= m_size);
    return array<T>(&m_arr[offset], size);
  }

  /// Destructor to delete the array.
  CUDA_HOST_DEVICE ~array() { delete[] m_arr; }

  /// Returns the size of the underlying array
  CUDA_HOST_DEVICE std::size_t size() const { return m_size; }
  /// Iterator functions
  CUDA_HOST_DEVICE T* begin() { return m_arr; }
  CUDA_HOST_DEVICE const T* begin() const { return m_arr; }
  CUDA_HOST_DEVICE T* end() { return m_arr + m_size; }
  CUDA_HOST_DEVICE const T* end() const { return m_arr + m_size; }
  /// Returns the ptr of the underlying array
  CUDA_HOST_DEVICE PUREFUNC T* ptr() const { return m_arr; }
  CUDA_HOST_DEVICE PUREFUNC T*& ptr_ref() { return m_arr; }
  /// Returns the reference to the location at the index of the underlying
  /// array
  CUDA_HOST_DEVICE PUREFUNC T& operator[](std::ptrdiff_t i) { return m_arr[i]; }
  CUDA_HOST_DEVICE PUREFUNC const T& operator[](std::ptrdiff_t i) const {
    return m_arr[i];
  }
  /// Returns the reference to the underlying array
  CUDA_HOST_DEVICE PUREFUNC T& operator*() { return *m_arr; }

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
  CUDA_HOST_DEVICE array<T>& operator=(const array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr[i];
    return *this;
  }

  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator=(const array_ref<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr[i];
    return *this;
  }

  /// Performs element wise addition
  CUDA_HOST_DEVICE array<T>& operator+=(const array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += arr[i];
    return *this;
  }
  /// Performs element wise subtraction
  CUDA_HOST_DEVICE array<T>& operator-=(const array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= arr[i];
    return *this;
  }
  /// Performs element wise multiplication
  CUDA_HOST_DEVICE array<T>& operator*=(const array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= arr[i];
    return *this;
  }
  /// Performs element wise division
  CUDA_HOST_DEVICE array<T>& operator/=(const array_ref<T>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= arr[i];
    return *this;
  }
  /// Initializes the clad::array to the given clad::array
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator=(const array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise addition
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator+=(const array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise subtraction
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator-=(const array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise multiplication
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator*=(const array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= static_cast<T>(arr[i]);
    return *this;
  }
  /// Initializes the clad::array from the given clad::array_expression
  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array<T>&
  operator=(const array_expression<L, BinaryOp, R>& arr_exp) {
    assert(arr_exp.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] = arr_exp[i];
    return *this;
  }
  /// Performs element wise division
  template <typename U>
  CUDA_HOST_DEVICE array<T>& operator/=(const array<U>& arr) {
    assert(arr.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= static_cast<T>(arr[i]);
    return *this;
  }
  /// Performs element wise addition with array_expression
  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array<T>&
  operator+=(const array_expression<L, BinaryOp, R>& arr_exp) {
    assert(arr_exp.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] += arr_exp[i];
    return *this;
  }
  /// Performs element wise subtraction with array_expression
  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array<T>&
  operator-=(const array_expression<L, BinaryOp, R>& arr_exp) {
    assert(arr_exp.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] -= arr_exp[i];
    return *this;
  }
  /// Performs element wise multiplication with array_expression
  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array<T>&
  operator*=(const array_expression<L, BinaryOp, R>& arr_exp) {
    assert(arr_exp.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] *= arr_exp[i];
    return *this;
  }
  /// Performs element wise division with array_expression
  template <typename L, typename BinaryOp, typename R>
  CUDA_HOST_DEVICE array<T>&
  operator/=(const array_expression<L, BinaryOp, R>& arr_exp) {
    assert(arr_exp.size() == m_size);
    for (std::size_t i = 0; i < m_size; i++)
      m_arr[i] /= arr_exp[i];
    return *this;
  }

  /// Negate the array and return a new array.
  CUDA_HOST_DEVICE array_expression<T, BinarySub, const array<T>&>
  operator-() const {
    return array_expression<T, BinarySub, const array<T>&>(static_cast<T>(0),
                                                           *this);
  }

  /// Implicitly converts from clad::array to pointer to an array of type T
  CUDA_HOST_DEVICE operator T*() const { return m_arr; }
}; // class array
// NOLINTEND(*-pointer-arithmetic)

// Function to instantiate a one-hot array of size n with 1 at index i.
// A one-hot vector is a vector with all elements set to 0 except for one
// element which is set to 1.
// For example, if n=4 and i=2, the returned array is: {0, 0, 1, 0}
template <typename T>
CUDA_HOST_DEVICE array<T> one_hot_vector(std::size_t n, std::size_t i) {
  array<T> arr(n);
  arr[i] = 1;
  return arr;
}

// Function to instantiate a zero vector of size n
template <typename T> CUDA_HOST_DEVICE array<T> zero_vector(std::size_t n) {
  return array<T>(n);
}

} // namespace clad

#endif // CLAD_ARRAY_H
