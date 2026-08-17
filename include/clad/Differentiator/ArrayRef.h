#ifndef CLAD_ARRAY_REF_H
#define CLAD_ARRAY_REF_H

#include "clad/Differentiator/Array.h"
#include "clad/Differentiator/CladConfig.h"

#include <assert.h>
#include <type_traits>

namespace clad {

// As in clad::array, the element-wise assignments of clad::array_ref come in
// families that differ only in the operator token, passed to these generators
// as `op`: one of `+=`, `-=`, `*=`, `/=`, or `=` where plain assignment has the
// same shape.
//
// The generators below suppress the same three checks as the ones in Array.h;
// see the comment there for why none of them can be followed.
// NOLINTBEGIN(cppcoreguidelines-macro-usage, bugprone-macro-parentheses,
// modernize-type-traits)

/// Applies `op` between every element and a scalar.
#define CLAD_ARRAY_REF_OP_SCALAR(op)                                           \
  template <typename U,                                                        \
            std::enable_if_t<std::is_arithmetic<U>::value, int> = 0>           \
  CUDA_HOST_DEVICE array_ref<T>& operator op(U a) {                            \
    for (std::size_t i = 0; i < m_size; i++)                                   \
      m_arr[i] op a;                                                           \
    return *this;                                                              \
  }

/// Applies `op` element-wise with another array_ref.
#define CLAD_ARRAY_REF_OP_ARRAY_REF(op)                                        \
  template <typename U>                                                        \
  CUDA_HOST_DEVICE array_ref<T>& operator op(const array_ref<U>& Ar) {         \
    assert(m_size == Ar.size() && "Size of both the array_refs must be equal " \
                                  "for carrying out compound assignment");     \
    for (std::size_t i = 0; i < m_size; i++)                                   \
      m_arr[i] op Ar[i];                                                       \
    return *this;                                                              \
  }

/// Applies `op` element-wise with a clad::array.
#define CLAD_ARRAY_REF_OP_ARRAY(op)                                            \
  template <typename U>                                                        \
  CUDA_HOST_DEVICE array_ref<T>& operator op(const array<U>& A) {              \
    assert(m_size == A.size() && "Size of arrays must be equal");              \
    for (std::size_t i = 0; i < m_size; i++)                                   \
      m_arr[i] op A[i];                                                        \
    return *this;                                                              \
  }

/// Applies `op` element-wise with an unevaluated array expression.
#define CLAD_ARRAY_REF_OP_EXPR(op)                                             \
  template <typename L, typename BinaryOp, typename R>                         \
  CUDA_HOST_DEVICE array_ref<T>& operator op(                                  \
      const array_expression<L, BinaryOp, R>& arr_exp) {                       \
    assert(arr_exp.size() == m_size);                                          \
    for (std::size_t i = 0; i < m_size; i++)                                   \
      m_arr[i] op arr_exp[i];                                                  \
    return *this;                                                              \
  }
// NOLINTEND(cppcoreguidelines-macro-usage, bugprone-macro-parentheses,
// modernize-type-traits)

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
  CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE operator T*() { return m_arr; }
  /// Operator for conversion from array_ref<T> to const T*.
  constexpr CUDA_HOST_DEVICE operator const T*() const { return m_arr; }

  CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE array_ref<T>&
  operator=(const array_ref<T>& a) {
    if (this == &a)
      return *this;
    assert(m_size == a.size());
    for (std::size_t i = 0; i < m_size; ++i)
      m_arr[i] = a[i];
    return *this;
  }

  /// Returns the size of the underlying array
  constexpr CUDA_HOST_DEVICE std::size_t size() const { return m_size; }
  constexpr CUDA_HOST_DEVICE PUREFUNC T* ptr() const { return m_arr; }
  CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE PUREFUNC T*& ptr_ref() { return m_arr; }
  /// Returns an array_ref to a part of the underlying array starting at
  /// offset and having the specified size
  CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE array_ref<T> slice(std::size_t offset,
                                                           std::size_t size) {
    assert((offset >= 0) && (offset + size <= m_size) &&
           "Window is outside array. Please provide an offset and size "
           "inside the array size.");
    return array_ref<T>(&m_arr[offset], size);
  }
  /// Returns the reference to the underlying array
  CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE PUREFUNC T& operator*() {
    return *m_arr;
  }

  // Arithmetic overloads
  CLAD_ARRAY_REF_OP_SCALAR(+=)
  CLAD_ARRAY_REF_OP_SCALAR(-=)
  CLAD_ARRAY_REF_OP_SCALAR(*=)
  CLAD_ARRAY_REF_OP_SCALAR(/=)

  /// Assignment from an array_ref is not generated here: the array_ref<T>
  /// overload above has to stay non-templated to catch self-assignment.
  CLAD_ARRAY_REF_OP_ARRAY_REF(+=)
  CLAD_ARRAY_REF_OP_ARRAY_REF(-=)
  CLAD_ARRAY_REF_OP_ARRAY_REF(*=)
  CLAD_ARRAY_REF_OP_ARRAY_REF(/=)

  CLAD_ARRAY_REF_OP_ARRAY(=)
  CLAD_ARRAY_REF_OP_ARRAY(+=)
  CLAD_ARRAY_REF_OP_ARRAY(-=)
  CLAD_ARRAY_REF_OP_ARRAY(*=)
  CLAD_ARRAY_REF_OP_ARRAY(/=)

  CLAD_ARRAY_REF_OP_EXPR(=)
  CLAD_ARRAY_REF_OP_EXPR(+=)
  CLAD_ARRAY_REF_OP_EXPR(-=)
  CLAD_ARRAY_REF_OP_EXPR(*=)
  CLAD_ARRAY_REF_OP_EXPR(/=)
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
    constexpr CUDA_HOST_DEVICE array_ref(T arr, std::size_t size = 1)
        : m_arr((void*)arr), m_size(size) {}
    template <typename T>
    constexpr CUDA_HOST_DEVICE array_ref(const array_ref<T>& other)
        : m_arr(other.ptr()), m_size(other.size()) {}
    template <typename T>
    CLAD_CONSTEXPR_CXX14 CUDA_HOST_DEVICE operator array_ref<T>() {
      return array_ref<T>((T*)(m_arr), m_size);
    }
    [[nodiscard]] constexpr CUDA_HOST_DEVICE void* ptr() const { return m_arr; }
    [[nodiscard]] constexpr CUDA_HOST_DEVICE std::size_t size() const {
      return m_size;
    }
  };
  // NOLINTEND(*-pointer-arithmetic)
} // namespace clad

#undef CLAD_ARRAY_REF_OP_SCALAR
#undef CLAD_ARRAY_REF_OP_ARRAY_REF
#undef CLAD_ARRAY_REF_OP_ARRAY
#undef CLAD_ARRAY_REF_OP_EXPR

#endif // CLAD_ARRAY_REF_H
