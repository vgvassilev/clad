#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_BASICOPS_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_BASICOPS_H

#include "clad/Differentiator/TorchBuiltins/TensorLifecycle.h" // IWYU pragma: export

#include <ATen/ops/threshold_backward.h>

// First supported operator set: add, sub, mul, div, relu, sum, dot, and
// item<float>. Scalar options such as add/sub alpha are non-differentiable.
namespace clad::custom_derivatives::at {

inline void add_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_lhs, ::at::Tensor* d_rhs,
                         ::at::Scalar* /*d_alpha*/) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  if (d_lhs)
    ::clad::torch::detail::accumulate_to_shape(d_lhs, d_output, lhs);
  if (d_rhs)
    ::clad::torch::detail::accumulate_to_shape(d_rhs,
                                               ::at::mul(d_output, alpha), rhs);
}

inline void sub_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_lhs, ::at::Tensor* d_rhs,
                         ::at::Scalar* /*d_alpha*/) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  if (d_lhs)
    ::clad::torch::detail::accumulate_to_shape(d_lhs, d_output, lhs);
  if (d_rhs)
    ::clad::torch::detail::accumulate_to_shape(
        d_rhs, ::at::mul(d_output, -alpha), rhs);
}

inline void mul_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         ::at::Tensor d_output, ::at::Tensor* d_lhs,
                         ::at::Tensor* d_rhs) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  if (d_lhs)
    ::clad::torch::detail::accumulate_to_shape(d_lhs, ::at::mul(d_output, rhs),
                                               lhs);
  if (d_rhs)
    ::clad::torch::detail::accumulate_to_shape(d_rhs, ::at::mul(d_output, lhs),
                                               rhs);
}

inline void div_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         ::at::Tensor d_output, ::at::Tensor* d_lhs,
                         ::at::Tensor* d_rhs) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  if (d_lhs)
    ::clad::torch::detail::accumulate_to_shape(d_lhs, ::at::div(d_output, rhs),
                                               lhs);
  if (d_rhs) {
    auto numerator = ::at::mul(d_output, lhs);
    auto denominator = ::at::mul(rhs, rhs);
    ::clad::torch::detail::accumulate_to_shape(
        d_rhs, ::at::neg(::at::div(numerator, denominator)), rhs);
  }
}

inline void relu_pullback(const ::at::Tensor& input, ::at::Tensor d_output,
                          ::at::Tensor* d_input) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_tensor(input);
  ::clad::torch::detail::validate_tensor(d_output);
  if (d_input)
    ::clad::torch::detail::accumulate(
        d_input, ::at::threshold_backward(d_output, input, 0));
}

inline void sum_pullback(const ::at::Tensor& input,
                         ::std::optional<::at::ScalarType> dtype,
                         ::at::Tensor d_output, ::at::Tensor* d_input,
                         ::std::optional<::at::ScalarType>* /*d_dtype*/) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_reduction(input, dtype);
  ::clad::torch::detail::validate_tensor(d_output);
  if (!d_input)
    return;
  auto contribution = ::clad::torch::detail::restore_reduced_adjoint(
      d_output, input, ::at::OptionalIntArrayRef(), /*keepdim=*/false);
  ::clad::torch::detail::accumulate(d_input, contribution);
}

inline void sum_pullback(const ::at::Tensor& input,
                         ::at::OptionalIntArrayRef dims, bool keepdim,
                         ::std::optional<::at::ScalarType> dtype,
                         ::at::Tensor d_output, ::at::Tensor* d_input,
                         ::at::OptionalIntArrayRef* /*d_dims*/,
                         bool* /*d_keepdim*/,
                         ::std::optional<::at::ScalarType>* /*d_dtype*/) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_reduction(input, dtype);
  ::clad::torch::detail::validate_tensor(d_output);
  if (!d_input)
    return;
  auto contribution = ::clad::torch::detail::restore_reduced_adjoint(
      d_output, input, dims, keepdim);
  ::clad::torch::detail::accumulate(d_input, contribution);
}

inline void dot_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         ::at::Tensor d_output, ::at::Tensor* d_lhs,
                         ::at::Tensor* d_rhs) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_dot_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  if (d_lhs)
    ::clad::torch::detail::accumulate(d_lhs, ::at::mul(rhs, d_output));
  if (d_rhs)
    ::clad::torch::detail::accumulate(d_rhs, ::at::mul(lhs, d_output));
}

} // namespace clad::custom_derivatives::at

namespace clad::custom_derivatives::class_functions {

inline void item_pullback(const ::at::Tensor* input, float d_output,
                          ::at::Tensor* d_input) {
  ::at::NoGradGuard guard;
  ::clad::torch::detail::validate_tensor(*input);
  TORCH_CHECK(input->numel() == 1,
              "Clad Torch item<float>() expects one tensor element");
  if (d_input)
    ::clad::torch::detail::accumulate(d_input,
                                      ::at::full_like(*input, d_output));
}

} // namespace clad::custom_derivatives::class_functions

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_BASICOPS_H
