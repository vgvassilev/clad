#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSYNTAX_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSYNTAX_H

#include "clad/Differentiator/TorchBuiltins/BasicOps.h" // IWYU pragma: export

// Keep the derivative formulas in BasicOps.h. These adapters only map the
// operator and Tensor method spellings onto the corresponding ATen functions.
namespace clad::custom_derivatives::at {

inline void operator_plus_pullback(const ::at::Tensor& lhs,
                                   const ::at::Tensor& rhs,
                                   ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                   ::at::Tensor* d_rhs) {
  ::at::Scalar d_alpha = 0;
  add_pullback(lhs, rhs, /*alpha=*/1, ::std::move(d_output), d_lhs, d_rhs,
               &d_alpha);
}

inline void operator_minus_pullback(const ::at::Tensor& lhs,
                                    const ::at::Tensor& rhs,
                                    ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                    ::at::Tensor* d_rhs) {
  ::at::Scalar d_alpha = 0;
  sub_pullback(lhs, rhs, /*alpha=*/1, ::std::move(d_output), d_lhs, d_rhs,
               &d_alpha);
}

inline void operator_star_pullback(const ::at::Tensor& lhs,
                                   const ::at::Tensor& rhs,
                                   ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                   ::at::Tensor* d_rhs) {
  mul_pullback(lhs, rhs, ::std::move(d_output), d_lhs, d_rhs);
}

inline void operator_slash_pullback(const ::at::Tensor& lhs,
                                    const ::at::Tensor& rhs,
                                    ::at::Tensor d_output, ::at::Tensor* d_lhs,
                                    ::at::Tensor* d_rhs) {
  div_pullback(lhs, rhs, ::std::move(d_output), d_lhs, d_rhs);
}

} // namespace clad::custom_derivatives::at

namespace clad::custom_derivatives::class_functions {

inline void add_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_self, ::at::Tensor* d_other,
                         ::at::Scalar* d_alpha) {
  ::clad::custom_derivatives::at::add_pullback(
      *self, other, alpha, ::std::move(d_output), d_self, d_other, d_alpha);
}

inline void sub_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_self, ::at::Tensor* d_other,
                         ::at::Scalar* d_alpha) {
  ::clad::custom_derivatives::at::sub_pullback(
      *self, other, alpha, ::std::move(d_output), d_self, d_other, d_alpha);
}

inline void mul_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::Tensor* d_other) {
  ::clad::custom_derivatives::at::mul_pullback(
      *self, other, ::std::move(d_output), d_self, d_other);
}

inline void div_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::Tensor* d_other) {
  ::clad::custom_derivatives::at::div_pullback(
      *self, other, ::std::move(d_output), d_self, d_other);
}

inline void relu_pullback(const ::at::Tensor* self, ::at::Tensor d_output,
                          ::at::Tensor* d_self) {
  ::clad::custom_derivatives::at::relu_pullback(*self, ::std::move(d_output),
                                                d_self);
}

inline void sum_pullback(const ::at::Tensor* self,
                         ::std::optional<::at::ScalarType> dtype,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::std::optional<::at::ScalarType>* d_dtype) {
  ::clad::custom_derivatives::at::sum_pullback(
      *self, dtype, ::std::move(d_output), d_self, d_dtype);
}

inline void sum_pullback(const ::at::Tensor* self,
                         ::at::OptionalIntArrayRef dims, bool keepdim,
                         ::std::optional<::at::ScalarType> dtype,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::OptionalIntArrayRef* d_dims, bool* d_keepdim,
                         ::std::optional<::at::ScalarType>* d_dtype) {
  ::clad::custom_derivatives::at::sum_pullback(*self, dims, keepdim, dtype,
                                               ::std::move(d_output), d_self,
                                               d_dims, d_keepdim, d_dtype);
}

inline void dot_pullback(const ::at::Tensor* self, const ::at::Tensor& other,
                         ::at::Tensor d_output, ::at::Tensor* d_self,
                         ::at::Tensor* d_other) {
  ::clad::custom_derivatives::at::dot_pullback(
      *self, other, ::std::move(d_output), d_self, d_other);
}

} // namespace clad::custom_derivatives::class_functions

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSYNTAX_H
