#ifndef CLAD_DIFFERENTIATOR_ARRAYEXPRESSION_H
#define CLAD_DIFFERENTIATOR_ARRAYEXPRESSION_H

#include <algorithm>
#include <type_traits>

// This is a helper class to implement expression templates for clad::array.

// NOLINTBEGIN(*-pointer-arithmetic)
namespace clad {

// Operator to add two elements.
struct BinaryAdd {
  template <typename T, typename U>
  static auto apply(const T& t, const U& u) -> decltype(t + u) {
    return t + u;
  }
};

// Operator to add two elements.
struct BinaryMul {
  template <typename T, typename U>
  static auto apply(const T& t, const U& u) -> decltype(t * u) {
    return t * u;
  }
};

// Operator to divide two elements.
struct BinaryDiv {
  template <typename T, typename U>
  static auto apply(const T& t, const U& u) -> decltype(t / u) {
    return t / u;
  }
};

// Operator to subtract two elements.
struct BinarySub {
  template <typename T, typename U>
  static auto apply(const T& t, const U& u) -> decltype(t - u) {
    return t - u;
  }
};

// Class to represent an array expression using templates.
template <typename LeftExp, typename BinaryOp, typename RightExp>
class array_expression {
  LeftExp l;
  RightExp r;

public:
  array_expression(LeftExp l, RightExp r) : l(l), r(r) {}

  // for scalars
  template <typename T, typename std::enable_if<std::is_arithmetic<T>::value,
                                                int>::type = 0>
  std::size_t get_size(const T& t) const {
    return 1;
  }
  template <typename T, typename std::enable_if<std::is_arithmetic<T>::value,
                                                int>::type = 0>
  T get(const T& t, std::size_t i) const {
    return t;
  }

  // for vectors
  template <typename T, typename std::enable_if<!std::is_arithmetic<T>::value,
                                                int>::type = 0>
  std::size_t get_size(const T& t) const {
    return t.size();
  }
  template <typename T, typename std::enable_if<!std::is_arithmetic<T>::value,
                                                int>::type = 0>
  auto get(const T& t, std::size_t i) const -> decltype(t[i]) {
    return t[i];
  }

  // We also need to handle the case when any of the operands is a scalar.
  auto operator[](std::size_t i) const
      -> decltype(BinaryOp::apply(get(l, i), get(r, i))) {
    return BinaryOp::apply(get(l, i), get(r, i));
  }

  std::size_t size() const { return std::max(get_size(l), get_size(r)); }
};

// A template class to determine whether a given type is array_expression, array
// or array_ref.
template <typename T> class array;
template <typename T> class array_ref;

template <typename T> struct is_clad_type : std::false_type {};

template <typename LeftExp, typename BinaryOp, typename RightExp>
struct is_clad_type<array_expression<LeftExp, BinaryOp, RightExp>>
    : std::true_type {};

template <typename T> struct is_clad_type<array<T>> : std::true_type {};

template <typename T> struct is_clad_type<array_ref<T>> : std::true_type {};

// Operator overload for addition, when one of the operands is array_expression,
// array or array_ref.
template <
    typename T1, typename T2,
    typename std::enable_if<is_clad_type<T1>::value || is_clad_type<T2>::value,
                            int>::type = 0>
array_expression<const T1&, BinaryAdd, const T2&> operator+(const T1& l,
                                                            const T2& r) {
  return {l, r};
}

// Operator overload for multiplication, when one of the operands is
// array_expression, array or array_ref.
template <
    typename T1, typename T2,
    typename std::enable_if<is_clad_type<T1>::value || is_clad_type<T2>::value,
                            int>::type = 0>
array_expression<const T1&, BinaryMul, const T2&> operator*(const T1& l,
                                                            const T2& r) {
  return {l, r};
}

// Operator overload for subtraction, when one of the operands is
// array_expression, array or array_ref.
template <
    typename T1, typename T2,
    typename std::enable_if<is_clad_type<T1>::value || is_clad_type<T2>::value,
                            int>::type = 0>
array_expression<const T1&, BinarySub, const T2&> operator-(const T1& l,
                                                            const T2& r) {
  return {l, r};
}

// Operator overload for division, when one of the operands is array_expression,
// array or array_ref.
template <
    typename T1, typename T2,
    typename std::enable_if<is_clad_type<T1>::value || is_clad_type<T2>::value,
                            int>::type = 0>
array_expression<const T1&, BinaryDiv, const T2&> operator/(const T1& l,
                                                            const T2& r) {
  return {l, r};
}
} // namespace clad
// NOLINTEND(*-pointer-arithmetic)

#endif // CLAD_DIFFERENTIATOR_ARRAYEXPRESSION_H