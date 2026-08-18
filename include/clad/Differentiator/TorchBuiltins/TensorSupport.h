#ifndef CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSUPPORT_H
#define CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSUPPORT_H

#include "clad/Differentiator/BuiltinDerivatives.h" // IWYU pragma: export

#include <ATen/ATen.h>           // IWYU pragma: export
#include <ATen/core/grad_mode.h> // IWYU pragma: export
#include <c10/core/WrapDimMinimal.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace clad::torch::detail {

// The first operator set deliberately has a narrow, checked contract for
// device, dtype, and layout. Elementwise operations follow ATen broadcasting.
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
}

inline void validate_dot_inputs(const ::at::Tensor& lhs,
                                const ::at::Tensor& rhs) {
  validate_elementwise_inputs(lhs, rhs);
  TORCH_CHECK(lhs.sizes() == rhs.sizes(),
              "Clad Torch dot derivatives require matching shapes");
  TORCH_CHECK(lhs.dim() == 1,
              "Clad Torch dot derivatives expect one-dimensional tensors");
}

inline void validate_reduction(const ::at::Tensor& input,
                               ::std::optional<::at::ScalarType> dtype) {
  validate_tensor(input);
  TORCH_CHECK(!dtype || *dtype == input.scalar_type(),
              "Clad Torch reduction derivatives do not support dtype "
              "conversion");
}

inline ::std::vector<int64_t>
canonicalize_reduction_dims(::at::IntArrayRef dims, int64_t input_rank) {
  ::std::vector<int64_t> canonical_dims;
  canonical_dims.reserve(dims.size());
  for (int64_t dim : dims)
    canonical_dims.push_back(::c10::maybe_wrap_dim(dim, input_rank));
  ::std::sort(canonical_dims.begin(), canonical_dims.end());
  TORCH_CHECK(
      ::std::adjacent_find(canonical_dims.begin(), canonical_dims.end()) ==
          canonical_dims.end(),
      "Clad Torch reduction dimensions must be unique");
  return canonical_dims;
}

inline ::at::Tensor restore_reduced_adjoint(const ::at::Tensor& adjoint,
                                            const ::at::Tensor& input,
                                            ::at::OptionalIntArrayRef dims,
                                            bool keepdim) {
  if (keepdim || input.dim() == 0 || !dims || dims->empty())
    return adjoint.expand_as(input);

  auto restored = adjoint;
  for (int64_t dim : canonicalize_reduction_dims(*dims, input.dim()))
    restored = restored.unsqueeze(dim);
  return restored.expand_as(input);
}

inline ::at::Tensor zero_adjoint_like(const ::at::Tensor& primal) {
  ::at::NoGradGuard guard;
  return ::at::zeros_like(primal);
}

inline void accumulate(::at::Tensor* destination,
                       const ::at::Tensor& contribution) {
  ::at::NoGradGuard guard;
  if (!destination->defined())
    *destination = ::at::zeros_like(contribution);
  destination->add_(contribution);
}

inline void accumulate_to_shape(::at::Tensor* destination,
                                const ::at::Tensor& contribution,
                                const ::at::Tensor& primal) {
  if (contribution.sizes() == primal.sizes()) {
    accumulate(destination, contribution);
    return;
  }
  accumulate(destination, contribution.sum_to_size(primal.sizes()));
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
