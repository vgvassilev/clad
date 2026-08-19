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
// device and dtype. Elementwise operations follow ATen broadcasting, while
// ATen kernels preserve the logical semantics of strided Tensor views.
inline void validate_tensor(const ::at::Tensor& tensor) {
  TORCH_CHECK(tensor.defined(),
              "Clad Torch derivatives require a defined tensor");
  TORCH_CHECK(tensor.device().is_cpu(),
              "Clad Torch derivatives currently support CPU tensors only");
  TORCH_CHECK(tensor.scalar_type() == ::at::kFloat,
              "Clad Torch derivatives currently support float32 tensors only");
  TORCH_CHECK(tensor.layout() == ::at::kStrided,
              "Clad Torch derivatives currently support strided tensors only");
}

inline bool should_propagate(const ::at::Tensor& adjoint) {
  if (!adjoint.defined())
    return false;
  validate_tensor(adjoint);
  return true;
}

inline bool can_combine_adjoint_contributions(const ::at::Tensor& lhs,
                                              const ::at::Tensor& rhs,
                                              const ::at::Tensor* d_lhs,
                                              const ::at::Tensor* d_rhs) {
  return d_lhs && d_lhs == d_rhs && lhs.sizes() == rhs.sizes();
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

inline ::std::vector<int64_t> inverse_permutation(::at::IntArrayRef dims,
                                                  int64_t input_rank) {
  TORCH_CHECK(static_cast<int64_t>(dims.size()) == input_rank,
              "Clad Torch permute derivatives require one dimension per input "
              "dimension");

  ::std::vector<int64_t> inverse(input_rank, -1);
  for (int64_t output_dim = 0; output_dim < input_rank; ++output_dim) {
    const int64_t input_dim =
        ::c10::maybe_wrap_dim(dims[output_dim], input_rank);
    TORCH_CHECK(inverse[input_dim] == -1,
                "Clad Torch permutation dimensions must be unique");
    inverse[input_dim] = output_dim;
  }
  return inverse;
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
  if (!destination->defined()) {
    // A borrowed contribution can be an expanded or strided view. Clone it so
    // later accumulation cannot mutate the upstream adjoint through an alias.
    *destination = contribution.clone();
    return;
  }
  destination->add_(contribution);
}

inline void accumulate(::at::Tensor* destination, ::at::Tensor&& contribution) {
  ::at::NoGradGuard guard;
  if (!destination->defined()) {
    // A unique, non-view temporary can become the adjoint directly. Clone
    // views and shared handles so later accumulation cannot mutate an alias.
    if (!contribution.is_view() && contribution.use_count() == 1 &&
        contribution.storage().unique())
      *destination = ::std::move(contribution);
    else
      *destination = contribution.clone();
    return;
  }
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

inline void accumulate_to_shape(::at::Tensor* destination,
                                ::at::Tensor&& contribution,
                                const ::at::Tensor& primal) {
  if (contribution.sizes() == primal.sizes()) {
    accumulate(destination, ::std::move(contribution));
    return;
  }
  accumulate(destination, contribution.sum_to_size(primal.sizes()));
}

inline void propagate_adjoint(::at::Tensor* source, ::at::Tensor* destination) {
  if (!source || !source->defined() || !destination)
    return;
  if (destination->defined() && source->is_alias_of(*destination))
    return;

  // Tensor copies share storage, while local adjoints must remain independent.
  ::at::NoGradGuard guard;
  accumulate(destination, source->clone());
  source->zero_();
}

} // namespace clad::torch::detail

#endif // CLAD_DIFFERENTIATOR_TORCHBUILTINS_TENSORSUPPORT_H
