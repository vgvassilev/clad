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

  // Operator overload for addition.
  template <typename RE>
  array_expression<const array_expression<LeftExp, BinaryOp, RightExp>&,
                   BinaryAdd, RE>
  operator+(const RE& r) const {
    return array_expression<
        const array_expression<LeftExp, BinaryOp, RightExp>&, BinaryAdd, RE>(
        *this, r);
  }

  // Operator overload for multiplication.
  template <typename RE>
  array_expression<const array_expression<LeftExp, BinaryOp, RightExp>&,
                   BinaryMul, RE>
  operator*(const RE& r) const {
    return array_expression<
        const array_expression<LeftExp, BinaryOp, RightExp>&, BinaryMul, RE>(
        *this, r);
  }

  // Operator overload for subtraction.
  template <typename RE>
  array_expression<const array_expression<LeftExp, BinaryOp, RightExp>&,
                   BinarySub, RE>
  operator-(const RE& r) const {
    return array_expression<
        const array_expression<LeftExp, BinaryOp, RightExp>&, BinarySub, RE>(
        *this, r);
  }

  // Operator overload for division.
  template <typename RE>
  array_expression<const array_expression<LeftExp, BinaryOp, RightExp>&,
                   BinaryDiv, RE>
  operator/(const RE& r) const {
    return array_expression<
        const array_expression<LeftExp, BinaryOp, RightExp>&, BinaryDiv, RE>(
        *this, r);
  }
};

// Operator overload for addition, when the right operand is an array_expression
// and the left operand is a scalar.
template <typename T, typename LeftExp, typename BinaryOp, typename RightExp,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
array_expression<T, BinaryAdd,
                 const array_expression<LeftExp, BinaryOp, RightExp>&>
operator+(const T& l, const array_expression<LeftExp, BinaryOp, RightExp>& r) {
  return array_expression<T, BinaryAdd,
                          const array_expression<LeftExp, BinaryOp, RightExp>&>(
      l, r);
}

// Operator overload for multiplication, when the right operand is an
// array_expression and the left operand is a scalar.
template <typename T, typename LeftExp, typename BinaryOp, typename RightExp,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
array_expression<T, BinaryMul,
                 const array_expression<LeftExp, BinaryOp, RightExp>&>
operator*(const T& l, const array_expression<LeftExp, BinaryOp, RightExp>& r) {
  return array_expression<T, BinaryMul,
                          const array_expression<LeftExp, BinaryOp, RightExp>&>(
      l, r);
}

// Operator overload for subtraction, when the right operand is an
// array_expression and the left operand is a scalar.
template <typename T, typename LeftExp, typename BinaryOp, typename RightExp,
          typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
array_expression<T, BinarySub,
                 const array_expression<LeftExp, BinaryOp, RightExp>&>
operator-(const T& l, const array_expression<LeftExp, BinaryOp, RightExp>& r) {
  return array_expression<T, BinarySub,
                          const array_expression<LeftExp, BinaryOp, RightExp>&>(
      l, r);
}
} // namespace clad
// NOLINTEND(*-pointer-arithmetic)

#endif // CLAD_DIFFERENTIATOR_ARRAYEXPRESSION_H