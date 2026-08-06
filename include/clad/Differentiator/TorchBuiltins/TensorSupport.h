#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSUPPORT_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSUPPORT_H

#include "clad/Differentiator/BuiltinDerivatives.h" // IWYU pragma: export

#include <ATen/ATen.h>           // IWYU pragma: export
#include <ATen/core/grad_mode.h> // IWYU pragma: export

#include <utility>

namespace clad::torch::detail {

// The first operator set deliberately has a narrow, checked contract. Device,
// dtype, layout, and broadcasting support can be added independently later.
inline void validate_tensor(const ::at::Tensor& tensor) {
  TORCH_CHECK(tensor.defined(),
              "Clad Torch derivatives require a defined tensor");
  TORCH_CHECK(tensor.device().is_cpu(),
              "Clad Torch derivatives currently support CPU tensors only");
  TORCH_CHECK(tensor.scalar_type() == ::at::kFloat,
              "Clad Torch derivatives currently support float32 tensors only");
  TORCH_CHECK(tensor.is_contiguous(),
              "Clad Torch derivatives currently require contiguous tensors");
}

inline void validate_elementwise_inputs(const ::at::Tensor& lhs,
                                        const ::at::Tensor& rhs) {
  validate_tensor(lhs);
  validate_tensor(rhs);
  TORCH_CHECK(lhs.sizes() == rhs.sizes(),
              "Clad Torch elementwise derivatives do not support broadcasting");
}

inline ::at::Tensor zero_adjoint_like(const ::at::Tensor& primal) {
  ::at::NoGradGuard guard;
  return ::at::zeros_like(primal);
}

inline ::clad::ValueAndAdjoint<::at::Tensor, ::at::Tensor>
make_value_and_zero_adjoint(::at::Tensor value) {
  auto adjoint = zero_adjoint_like(value);
  return {::std::move(value), ::std::move(adjoint)};
}

inline void accumulate(::at::Tensor* destination,
                       const ::at::Tensor& contribution) {
  ::at::NoGradGuard guard;
  if (!destination->defined())
    *destination = ::at::zeros_like(contribution);
  destination->add_(contribution);
}

inline void propagate_adjoint(::at::Tensor* source, ::at::Tensor* destination) {
  if (!source->defined())
    return;
  if (destination->defined() && source->is_alias_of(*destination))
    return;

  // Tensor copies share storage, while local adjoints must remain independent.
  ::at::NoGradGuard guard;
  auto contribution = source->clone();
  accumulate(destination, contribution);
  source->zero_();
}

} // namespace clad::torch::detail

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSUPPORT_H
