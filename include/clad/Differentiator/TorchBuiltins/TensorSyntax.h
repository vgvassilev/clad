#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSYNTAX_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSYNTAX_H

#include "clad/Differentiator/TorchBuiltins/BasicOps.h" // IWYU pragma: export

// Keep the derivative formulas in BasicOps.h. These adapters only map the
// operator and Tensor method spellings onto the corresponding ATen functions.
namespace clad::custom_derivatives::at {

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
operator_plus_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                           const ::at::Tensor& d_lhs,
                           const ::at::Tensor& d_rhs) {
  return add_reverse_forw(lhs, rhs, /*alpha=*/1, d_lhs, d_rhs,
                          /*d_alpha=*/0);
}

inline void operator_plus_pullback(const ::at::Tensor& lhs,
                                   const ::at::Tensor& rhs,
                                   ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                   ::at::Tensor* d_rhs) {
  ::at::Scalar d_alpha = 0;
  add_pullback(lhs, rhs, /*alpha=*/1, ::std::move(d_output), d_lhs, d_rhs,
               &d_alpha);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
operator_minus_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                            const ::at::Tensor& d_lhs,
                            const ::at::Tensor& d_rhs) {
  return sub_reverse_forw(lhs, rhs, /*alpha=*/1, d_lhs, d_rhs,
                          /*d_alpha=*/0);
}

inline void operator_minus_pullback(const ::at::Tensor& lhs,
                                    const ::at::Tensor& rhs,
                                    ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                    ::at::Tensor* d_rhs) {
  ::at::Scalar d_alpha = 0;
  sub_pullback(lhs, rhs, /*alpha=*/1, ::std::move(d_output), d_lhs, d_rhs,
               &d_alpha);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
operator_star_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                           const ::at::Tensor& d_lhs,
                           const ::at::Tensor& d_rhs) {
  return mul_reverse_forw(lhs, rhs, d_lhs, d_rhs);
}

inline void operator_star_pullback(const ::at::Tensor& lhs,
                                   const ::at::Tensor& rhs,
                                   ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                   ::at::Tensor* d_rhs) {
  mul_pullback(lhs, rhs, ::std::move(d_output), d_lhs, d_rhs);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
operator_slash_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                            const ::at::Tensor& d_lhs,
                            const ::at::Tensor& d_rhs) {
  return div_reverse_forw(lhs, rhs, d_lhs, d_rhs);
}

inline void operator_slash_pullback(const ::at::Tensor& lhs,
                                    const ::at::Tensor& rhs,
                                    ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                    ::at::Tensor* d_rhs) {
  div_pullback(lhs, rhs, ::std::move(d_output), d_lhs, d_rhs);
}

} // namespace clad::custom_derivatives::at

namespace clad::custom_derivatives::class_functions {

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
add_reverse_forw(const ::at::Tensor* self, const ::at::Tensor& other,
                 const ::at::Scalar& alpha, const ::at::Tensor* d_self,
                 const ::at::Tensor& d_other, const ::at::Scalar& d_alpha) {
  return ::clad::custom_derivatives::at::add_reverse_forw(
      *self, other, alpha, *d_self, d_other, d_alpha);
}

inline void add_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_self, ::at::Tensor* d_other,
                         ::at::Scalar* d_alpha) {
  ::clad::custom_derivatives::at::add_pullback(
      *self, other, alpha, ::std::move(d_output), d_self, d_other, d_alpha);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
sub_reverse_forw(const ::at::Tensor* self, const ::at::Tensor& other,
                 const ::at::Scalar& alpha, const ::at::Tensor* d_self,
                 const ::at::Tensor& d_other, const ::at::Scalar& d_alpha) {
  return ::clad::custom_derivatives::at::sub_reverse_forw(
      *self, other, alpha, *d_self, d_other, d_alpha);
}

inline void sub_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_self, ::at::Tensor* d_other,
                         ::at::Scalar* d_alpha) {
  ::clad::custom_derivatives::at::sub_pullback(
      *self, other, alpha, ::std::move(d_output), d_self, d_other, d_alpha);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
mul_reverse_forw(const ::at::Tensor* self, const ::at::Tensor& other,
                 const ::at::Tensor* d_self, const ::at::Tensor& d_other) {
  return ::clad::custom_derivatives::at::mul_reverse_forw(*self, other, *d_self,
                                                          d_other);
}

inline void mul_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::Tensor* d_other) {
  ::clad::custom_derivatives::at::mul_pullback(
      *self, other, ::std::move(d_output), d_self, d_other);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
div_reverse_forw(const ::at::Tensor* self, const ::at::Tensor& other,
                 const ::at::Tensor* d_self, const ::at::Tensor& d_other) {
  return ::clad::custom_derivatives::at::div_reverse_forw(*self, other, *d_self,
                                                          d_other);
}

inline void div_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::Tensor* d_other) {
  ::clad::custom_derivatives::at::div_pullback(
      *self, other, ::std::move(d_output), d_self, d_other);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
relu_reverse_forw(const ::at::Tensor* self, const ::at::Tensor* d_self) {
  return ::clad::custom_derivatives::at::relu_reverse_forw(*self, *d_self);
}

inline void relu_pullback(const ::at::Tensor* self, ::at::Tensor d_output,
                          ::at::Tensor* d_self) {
  ::clad::custom_derivatives::at::relu_pullback(*self, ::std::move(d_output),
                                                d_self);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
dot_reverse_forw(const ::at::Tensor* self, const ::at::Tensor& other,
                 const ::at::Tensor* d_self, const ::at::Tensor& d_other) {
  return ::clad::custom_derivatives::at::dot_reverse_forw(*self, other, *d_self,
                                                          d_other);
}

inline void dot_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::Tensor* d_other) {
  ::clad::custom_derivatives::at::dot_pullback(
      *self, other, ::std::move(d_output), d_self, d_other);
}

} // namespace clad::custom_derivatives::class_functions

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSYNTAX_H
