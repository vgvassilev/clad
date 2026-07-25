#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_BASICOPS_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_BASICOPS_H

#include "clad/Differentiator/TorchBuiltins/TensorSupport.h" // IWYU pragma: export

// First supported operator set: add, sub, mul, relu, dot, and item<float>.
// Scalar options such as add/sub alpha are treated as non-differentiable.
namespace clad::custom_derivatives::at {

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
add_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                 const ::at::Scalar& alpha,
                 [[maybe_unused]] const ::at::Tensor& d_lhs,
                 [[maybe_unused]] const ::at::Tensor& d_rhs,
                 [[maybe_unused]] const ::at::Scalar& d_alpha) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::at::NoGradGuard guard;
  return ::clad::torch::detail::make_value_and_zero_adjoint(
      ::at::add(lhs, rhs, alpha));
}

inline void add_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_lhs, ::at::Tensor* d_rhs,
                         [[maybe_unused]] ::at::Scalar* d_alpha) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  ::at::NoGradGuard guard;
  ::clad::torch::detail::accumulate(d_lhs, d_output);
  ::clad::torch::detail::accumulate(d_rhs, ::at::mul(d_output, alpha));
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
sub_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                 const ::at::Scalar& alpha,
                 [[maybe_unused]] const ::at::Tensor& d_lhs,
                 [[maybe_unused]] const ::at::Tensor& d_rhs,
                 [[maybe_unused]] const ::at::Scalar& d_alpha) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::at::NoGradGuard guard;
  return ::clad::torch::detail::make_value_and_zero_adjoint(
      ::at::sub(lhs, rhs, alpha));
}

inline void sub_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         const ::at::Scalar& alpha, ::at::Tensor d_output,
                         ::at::Tensor* d_lhs, ::at::Tensor* d_rhs,
                         [[maybe_unused]] ::at::Scalar* d_alpha) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  ::at::NoGradGuard guard;
  ::clad::torch::detail::accumulate(d_lhs, d_output);
  ::clad::torch::detail::accumulate(d_rhs, ::at::mul(d_output, -alpha));
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
mul_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                 [[maybe_unused]] const ::at::Tensor& d_lhs,
                 [[maybe_unused]] const ::at::Tensor& d_rhs) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::at::NoGradGuard guard;
  return ::clad::torch::detail::make_value_and_zero_adjoint(
      ::at::mul(lhs, rhs));
}

inline void mul_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         ::at::Tensor d_output, ::at::Tensor* d_lhs,
                         ::at::Tensor* d_rhs) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  ::at::NoGradGuard guard;
  ::clad::torch::detail::accumulate(d_lhs, ::at::mul(d_output, rhs));
  ::clad::torch::detail::accumulate(d_rhs, ::at::mul(d_output, lhs));
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
relu_reverse_forw(const ::at::Tensor& input,
                  [[maybe_unused]] const ::at::Tensor& d_input) {
  ::clad::torch::detail::validate_tensor(input);
  ::at::NoGradGuard guard;
  return ::clad::torch::detail::make_value_and_zero_adjoint(::at::relu(input));
}

inline void relu_pullback(const ::at::Tensor& input, ::at::Tensor d_output,
                          ::at::Tensor* d_input) {
  ::clad::torch::detail::validate_tensor(input);
  ::clad::torch::detail::validate_tensor(d_output);
  ::at::NoGradGuard guard;
  auto mask = ::at::gt(input, 0).to(input.scalar_type());
  ::clad::torch::detail::accumulate(d_input, ::at::mul(d_output, mask));
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
dot_reverse_forw(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                 [[maybe_unused]] const ::at::Tensor& d_lhs,
                 [[maybe_unused]] const ::at::Tensor& d_rhs) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  TORCH_CHECK(lhs.dim() == 1,
              "Clad Torch dot derivatives expect one-dimensional tensors");
  ::at::NoGradGuard guard;
  return ::clad::torch::detail::make_value_and_zero_adjoint(
      ::at::dot(lhs, rhs));
}

inline void dot_pullback(const ::at::Tensor& lhs, const ::at::Tensor& rhs,
                         ::at::Tensor d_output, ::at::Tensor* d_lhs,
                         ::at::Tensor* d_rhs) {
  ::clad::torch::detail::validate_elementwise_inputs(lhs, rhs);
  ::clad::torch::detail::validate_tensor(d_output);
  TORCH_CHECK(lhs.dim() == 1,
              "Clad Torch dot derivatives expect one-dimensional tensors");
  ::at::NoGradGuard guard;
  ::clad::torch::detail::accumulate(d_lhs, ::at::mul(rhs, d_output));
  ::clad::torch::detail::accumulate(d_rhs, ::at::mul(lhs, d_output));
}

} // namespace clad::custom_derivatives::at

namespace clad::custom_derivatives::class_functions {

inline ::clad::ValueAndAdjoint<float, float>
item_reverse_forw(const ::at::Tensor* input,
                  [[maybe_unused]] const ::at::Tensor* d_input) {
  ::clad::torch::detail::validate_tensor(*input);
  TORCH_CHECK(input->numel() == 1,
              "Clad Torch item<float>() expects one tensor element");
  ::at::NoGradGuard guard;
  return {input->item<float>(), 0.0F};
}

inline void item_pullback(const ::at::Tensor* input, float d_output,
                          ::at::Tensor* d_input) {
  ::clad::torch::detail::validate_tensor(*input);
  TORCH_CHECK(input->numel() == 1,
              "Clad Torch item<float>() expects one tensor element");
  ::at::NoGradGuard guard;
  ::clad::torch::detail::accumulate(d_input, ::at::full_like(*input, d_output));
}

} // namespace clad::custom_derivatives::class_functions

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_BASICOPS_H
