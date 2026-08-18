#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORLIFECYCLE_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORLIFECYCLE_H

#include "clad/Differentiator/TorchBuiltins/TensorSupport.h" // IWYU pragma: export

#include <utility>

namespace clad {

inline ::at::Tensor zero_like(const ::at::Tensor& tensor) {
  if (!tensor.defined())
    return {};
  return ::clad::torch::detail::zero_adjoint_like(tensor);
}

inline void zero_init(::at::Tensor& tensor) {
  if (tensor.defined()) {
    ::at::NoGradGuard guard;
    tensor.zero_();
  }
}

namespace custom_derivatives::class_functions {

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
constructor_reverse_forw(ConstructorPushforwardTag<::at::Tensor>,
                         const ::at::Tensor& input,
                         [[maybe_unused]] const ::at::Tensor& d_input) {
  ::clad::torch::detail::validate_tensor(input);
  // A local Tensor is a new reverse-mode node even though its primal copy
  // shares storage. Its adjoint must start at zero and use independent storage.
  return {input, ::clad::torch::detail::zero_adjoint_like(input)};
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
constructor_reverse_forw(ConstructorPushforwardTag<::at::Tensor>,
                         ::at::Tensor&& input,
                         [[maybe_unused]] ::at::Tensor&& d_input) {
  ::clad::torch::detail::validate_tensor(input);
  auto adjoint = ::clad::torch::detail::zero_adjoint_like(input);
  return {::std::move(input), ::std::move(adjoint)};
}

inline void constructor_pullback([[maybe_unused]] const ::at::Tensor& input,
                                 ::at::Tensor* d_this, ::at::Tensor* d_input) {
  ::clad::torch::detail::propagate_adjoint(d_this, d_input);
}

inline void constructor_pullback([[maybe_unused]] ::at::Tensor&& input,
                                 ::at::Tensor* d_this, ::at::Tensor* d_input) {
  ::clad::torch::detail::propagate_adjoint(d_this, d_input);
}

} // namespace custom_derivatives::class_functions
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORLIFECYCLE_H
