#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/TorchBuiltins.h" // IWYU pragma: keep

#include <torch/torch.h> // IWYU pragma: keep

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

namespace torch_api = torch;

void require_true(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

void require_close(const at::Tensor& actual, const at::Tensor& expected,
                   const char* message) {
  if (!at::allclose(actual, expected, 1.0e-5, 1.0e-6)) {
    std::cerr << message << "\nactual: " << actual << "\nexpected: " << expected
              << '\n';
    throw std::runtime_error(message);
  }
}

template <typename Callback>
void require_c10_error(Callback callback, const char* message) {
  try {
    callback();
  } catch (const c10::Error&) {
    return;
  }
  throw std::runtime_error(message);
}

torch::Tensor torch_utility(const torch::Tensor& input,
                            const torch::Tensor& bias) {
  auto squared = torch::mul(input, input);
  auto shifted = torch::add(squared, bias);
  auto activated = torch::relu(shifted);
  return activated;
}

float torch_loss(const torch::Tensor& input, const torch::Tensor& bias) {
  auto output = torch_utility(input, bias);
  auto residual = torch::sub(output, bias);
  auto scalar = torch::dot(residual, input);
  return scalar.item<float>();
}

at::Tensor at_utility(const at::Tensor& input) {
  // Exercise at::Tensor's shallow copy constructor. Its adjoint must still use
  // independent storage so reverse accumulation cannot alias the input buffer.
  at::Tensor copied(input);
  auto doubled = at::add(copied, input);
  auto activated = at::relu(doubled);
  return activated;
}

float at_loss(const at::Tensor& input) {
  auto output = at_utility(input);
  auto scalar = at::dot(output, input);
  return scalar.item<float>();
}

torch::Tensor direct_utility(const torch::Tensor& input) {
  return torch::relu(input);
}

float direct_loss(const torch::Tensor& input) {
  auto output = direct_utility(input);
  auto scalar = torch::dot(output, input);
  return scalar.item<float>();
}

torch_api::Tensor namespace_alias_utility(const torch_api::Tensor& input) {
  auto squared = torch_api::mul(input, input);
  return squared;
}

float namespace_alias_loss(const torch_api::Tensor& input) {
  auto output = namespace_alias_utility(input);
  auto scalar = torch_api::dot(output, input);
  return scalar.item<float>();
}

torch::Tensor operator_utility(const torch::Tensor& lhs,
                               const torch::Tensor& rhs) {
  auto added = lhs + rhs;
  auto subtracted = lhs - rhs;
  auto multiplied = added * subtracted;
  return multiplied / rhs;
}

float operator_loss(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  auto output = operator_utility(lhs, rhs);
  auto scalar = torch::dot(output, lhs);
  return scalar.item<float>();
}

torch::Tensor method_utility(const torch::Tensor& lhs,
                             const torch::Tensor& rhs) {
  auto added = lhs.add(rhs, /*alpha=*/0.5);
  auto subtracted = lhs.sub(rhs, /*alpha=*/0.25);
  auto multiplied = added.mul(subtracted);
  auto divided = multiplied.div(rhs);
  return divided.relu();
}

float method_loss(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  auto output = method_utility(lhs, rhs);
  auto scalar = output.dot(lhs);
  return scalar.item<float>();
}

float partial_gradient_loss(const at::Tensor& lhs, const at::Tensor& rhs) {
  auto product = at::mul(lhs, rhs);
  return at::dot(product, lhs).item<float>();
}

float tensor_sum_loss(const at::Tensor& input) {
  auto scalar = input.sum();
  return scalar.item<float>();
}

float tensor_sum_dim_method_loss(const at::Tensor& input,
                                 const at::Tensor& weights) {
  auto reduced = input.sum(/*dim=*/1);
  auto weighted = reduced.mul(weights);
  return weighted.sum().item<float>();
}

float tensor_sum_dims_method_loss(const at::Tensor& input,
                                  const at::Tensor& weights) {
  auto reduced = input.sum({-3, -1}, /*keepdim=*/true);
  auto weighted = reduced.mul(weights);
  return weighted.sum().item<float>();
}

float tensor_sum_dims_loss(const at::Tensor& input, const at::Tensor& weights,
                           at::OptionalIntArrayRef dims, bool keepdim) {
  auto reduced = at::sum(input, dims, keepdim);
  auto weighted = at::mul(reduced, weights);
  return weighted.sum().item<float>();
}

float default_method_parameter_loss(const at::Tensor& input,
                                    const at::Tensor& bias,
                                    const at::Tensor& offset,
                                    const at::Tensor& weights, int64_t dim) {
  auto shifted = input.add(bias);
  auto centered = shifted.sub(offset);
  auto reduced = centered.sum(dim);
  auto weighted = reduced.mul(weights);
  auto scalar = weighted.sum();
  return scalar.item<float>();
}

float default_function_parameter_loss(const at::Tensor& input,
                                      const at::Tensor& weights,
                                      at::OptionalIntArrayRef dims) {
  auto reduced = at::sum(input, dims);
  auto weighted = at::mul(reduced, weights);
  auto scalar = at::sum(weighted);
  return scalar.item<float>();
}

float explicit_sum_options_loss(const at::Tensor& input,
                                const at::Tensor& weights,
                                at::OptionalIntArrayRef dims, bool keepdim) {
  auto reduced = input.sum(dims, keepdim, at::kFloat);
  auto weighted = reduced.mul(weights);
  auto scalar = weighted.sum(at::ScalarType::Float);
  return scalar.item<float>();
}

float unsupported_sum_dtype_loss(const at::Tensor& input) {
  auto scalar = input.sum(at::ScalarType::Double);
  return scalar.item<float>();
}

float broadcast_add_loss(const at::Tensor& lhs, const at::Tensor& rhs) {
  auto output = lhs + rhs;
  auto scalar = output.sum();
  return scalar.item<float>();
}

float broadcast_sub_loss(const at::Tensor& lhs, const at::Tensor& rhs) {
  auto output = at::sub(lhs, rhs);
  auto scalar = at::sum(output);
  return scalar.item<float>();
}

float broadcast_mul_loss(const at::Tensor& lhs, const at::Tensor& rhs) {
  auto output = lhs.mul(rhs);
  auto scalar = output.sum();
  return scalar.item<float>();
}

float broadcast_div_loss(const at::Tensor& lhs, const at::Tensor& rhs) {
  auto output = at::div(lhs, rhs);
  auto scalar = at::sum(output);
  return scalar.item<float>();
}

float strided_input_loss(const at::Tensor& input, const at::Tensor& weights) {
  auto scaled = input.mul(weights);
  auto shifted = scaled.add(input);
  return shifted.relu().sum().item<float>();
}

float transpose_method_loss(const at::Tensor& input, const at::Tensor& weights,
                            int64_t dim0, int64_t dim1) {
  auto transposed = input.transpose(dim0, dim1);
  return transposed.mul(weights).sum().item<float>();
}

float transpose_function_loss(const at::Tensor& input,
                              const at::Tensor& weights, int64_t dim0,
                              int64_t dim1) {
  auto transposed = at::transpose(input, dim0, dim1);
  return at::sum(at::mul(transposed, weights)).item<float>();
}

float permute_method_loss(const at::Tensor& input, const at::Tensor& weights,
                          at::IntArrayRef dims) {
  auto permuted = input.permute(dims);
  return permuted.mul(weights).sum().item<float>();
}

float permute_function_loss(const at::Tensor& input, const at::Tensor& weights,
                            at::IntArrayRef dims) {
  auto permuted = at::permute(input, dims);
  return at::sum(at::mul(permuted, weights)).item<float>();
}

float chained_view_loss(const at::Tensor& input, const at::Tensor& weights,
                        at::IntArrayRef dims, int64_t dim0, int64_t dim1) {
  auto permuted = input.permute(dims);
  auto transposed = at::transpose(permuted, dim0, dim1);
  auto activated = transposed.mul(weights).relu();
  return activated.sum().item<float>();
}

void check_torch_alias_composition() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto input = torch::linspace(-2.0, 2.0, 17, options);
  const auto bias = torch::linspace(0.5, -0.5, 17, options);
  auto input_gradient = torch::zeros_like(input);
  auto bias_gradient = torch::zeros_like(bias);

  auto clad_gradient = clad::gradient(torch_loss, "input, bias");
  clad_gradient.execute(input, bias, &input_gradient, &bias_gradient);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_bias = bias.detach().clone().set_requires_grad(true);
  auto native_output = torch_utility(native_input, native_bias);
  auto native_loss =
      torch::dot(torch::sub(native_output, native_bias), native_input);
  auto native_gradients =
      torch::autograd::grad({native_loss}, {native_input, native_bias});

  require_close(input_gradient, native_gradients[0],
                "torch::Tensor input gradient is incorrect");
  require_close(bias_gradient, native_gradients[1],
                "torch::Tensor bias gradient is incorrect");
}

void check_at_namespace_composition() {
  const auto input =
      at::linspace(-2.0, 2.0, 17, at::TensorOptions().dtype(at::kFloat));
  auto gradient = at::zeros_like(input);
  auto clad_gradient = clad::gradient(at_loss, "input");
  clad_gradient.execute(input, &gradient);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_loss = at::dot(at_utility(native_input), native_input);
  auto native_gradient =
      torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(gradient, native_gradient,
                "at::Tensor composition gradient is incorrect");
}

void check_direct_return_composition() {
  const auto input = torch::linspace(
      -2.0, 2.0, 17, torch::TensorOptions().dtype(torch::kFloat32));
  auto gradient = torch::zeros_like(input);
  auto clad_gradient = clad::gradient(direct_loss, "input");
  clad_gradient.execute(input, &gradient);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_loss = torch::dot(direct_utility(native_input), native_input);
  auto native_gradient =
      torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(gradient, native_gradient,
                "direct Tensor return gradient is incorrect");
}

void check_tensor_adjoint_lifecycle() {
  const auto input =
      at::linspace(-2.0, 2.0, 17, at::TensorOptions().dtype(at::kFloat));

  auto lazy_adjoint = clad::zero_like(input);
  require_true(!lazy_adjoint.defined(),
               "Tensor zero_like did not create a lazy zero adjoint");

  const at::Tensor d_input;
  auto copied =
      clad::custom_derivatives::class_functions::constructor_reverse_forw(
          clad::Tag<at::Tensor>{}, input, d_input);
  require_true(copied.adjoint.defined(),
               "Tensor copy adjoint storage was not materialized");
  require_true(copied.adjoint.sizes() == input.sizes(),
               "Tensor copy adjoint has the wrong shape");
  require_close(copied.adjoint, at::zeros_like(input),
                "Tensor copy adjoint is not zero");
  require_true(!copied.adjoint.is_alias_of(input),
               "Tensor copy adjoint aliases the primal input");
}

void check_lazy_accumulation() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  auto seed = at::scalar_tensor(2.0, options);
  auto expanded = seed.expand({2, 3});
  at::Tensor borrowed_destination;

  clad::torch::detail::accumulate(&borrowed_destination, expanded);
  require_true(borrowed_destination.defined(),
               "borrowed contribution did not materialize its destination");
  require_close(borrowed_destination, at::full({2, 3}, 2.0, options),
                "borrowed contribution was accumulated incorrectly");
  require_true(!borrowed_destination.is_alias_of(seed),
               "borrowed contribution aliases its upstream adjoint");
  borrowed_destination.add_(1.0);
  require_close(seed, at::scalar_tensor(2.0, options),
                "mutating a materialized adjoint changed its source view");

  at::Tensor temporary_view_destination;
  clad::torch::detail::accumulate(&temporary_view_destination,
                                  seed.expand({2, 3}));
  require_true(temporary_view_destination.is_contiguous(),
               "temporary view contribution was not materialized");
  require_true(!temporary_view_destination.is_alias_of(seed),
               "temporary view contribution aliases its upstream adjoint");
  temporary_view_destination.add_(1.0);
  require_close(seed, at::scalar_tensor(2.0, options),
                "mutating a temporary view contribution changed its source");

  auto shared_contribution = at::linspace(1.0, 4.0, 4, options);
  auto shared_alias = shared_contribution;
  at::Tensor shared_destination;
  clad::torch::detail::accumulate(&shared_destination,
                                  ::std::move(shared_contribution));
  require_true(!shared_destination.is_alias_of(shared_alias),
               "shared Tensor handle was reused as an adjoint");

  auto detached_source = at::linspace(1.0, 4.0, 4, options);
  auto detached_contribution = detached_source.detach();
  at::Tensor detached_destination;
  clad::torch::detail::accumulate(&detached_destination,
                                  ::std::move(detached_contribution));
  require_true(!detached_destination.is_alias_of(detached_source),
               "shared Tensor storage was reused as an adjoint");

  auto owned_contribution = at::linspace(1.0, 4.0, 4, options);
  const auto* owned_storage = owned_contribution.data_ptr<float>();
  at::Tensor owned_destination;
  clad::torch::detail::accumulate(&owned_destination,
                                  ::std::move(owned_contribution));
  require_true(owned_destination.data_ptr<float>() == owned_storage,
               "unique Tensor contribution storage was not reused");

  auto accumulated = at::full({4}, 3.0, options);
  clad::torch::detail::accumulate(&accumulated, at::ones({4}, options));
  require_close(accumulated, at::full({4}, 4.0, options),
                "defined destination accumulation is incorrect");
}

void check_undefined_output_adjoint() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto lhs = at::linspace(1.0, 4.0, 4, options);
  const auto rhs = at::linspace(2.0, 5.0, 4, options);
  const at::Tensor d_output;
  auto d_lhs = at::full_like(lhs, 3.0);
  auto d_rhs = at::full_like(rhs, 4.0);
  const auto expected_lhs = d_lhs.clone();
  const auto expected_rhs = d_rhs.clone();
  at::Scalar d_alpha = 0;

  clad::custom_derivatives::at::add_pullback(lhs, rhs, /*alpha=*/1, d_output,
                                             &d_lhs, &d_rhs, &d_alpha);
  clad::custom_derivatives::at::sub_pullback(lhs, rhs, /*alpha=*/1, d_output,
                                             &d_lhs, &d_rhs, &d_alpha);
  clad::custom_derivatives::at::mul_pullback(lhs, rhs, d_output, &d_lhs,
                                             &d_rhs);
  clad::custom_derivatives::at::div_pullback(lhs, rhs, d_output, &d_lhs,
                                             &d_rhs);
  clad::custom_derivatives::at::relu_pullback(lhs, d_output, &d_lhs);
  clad::custom_derivatives::at::dot_pullback(lhs, rhs, d_output, &d_lhs,
                                             &d_rhs);

  ::std::optional<at::ScalarType> dtype;
  ::std::optional<at::ScalarType> d_dtype;
  clad::custom_derivatives::at::sum_pullback(lhs, dtype, d_output, &d_lhs,
                                             &d_dtype);

  const ::std::vector<int64_t> dims{0};
  at::OptionalIntArrayRef d_dims;
  bool d_keepdim = false;
  clad::custom_derivatives::at::sum_pullback(lhs, dims, /*keepdim=*/false,
                                             dtype, d_output, &d_lhs, &d_dims,
                                             &d_keepdim, &d_dtype);

  require_close(d_lhs, expected_lhs,
                "undefined output adjoint changed the lhs destination");
  require_close(d_rhs, expected_rhs,
                "undefined output adjoint changed the rhs destination");
}

void check_shared_pullback_destination() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto lhs = at::linspace(1.0, 4.0, 4, options);
  const auto rhs = at::linspace(2.0, 5.0, 4, options);
  const auto d_output = at::linspace(0.5, 2.0, 4, options);
  const auto initial = at::full({4}, 0.25, options);
  at::Scalar d_alpha = 0;

  auto actual = initial.clone();
  clad::custom_derivatives::at::add_pullback(lhs, rhs, /*alpha=*/2, d_output,
                                             &actual, &actual, &d_alpha);
  require_close(actual, initial + 3 * d_output,
                "shared add destination was accumulated incorrectly");

  actual = initial.clone();
  clad::custom_derivatives::at::sub_pullback(lhs, rhs, /*alpha=*/2, d_output,
                                             &actual, &actual, &d_alpha);
  require_close(actual, initial - d_output,
                "shared sub destination was accumulated incorrectly");

  actual = initial.clone();
  clad::custom_derivatives::at::mul_pullback(lhs, rhs, d_output, &actual,
                                             &actual);
  require_close(actual, initial + d_output * (lhs + rhs),
                "shared mul destination was accumulated incorrectly");

  actual = initial.clone();
  clad::custom_derivatives::at::div_pullback(lhs, rhs, d_output, &actual,
                                             &actual);
  require_close(actual, initial + d_output / rhs - d_output * lhs / (rhs * rhs),
                "shared div destination was accumulated incorrectly");

  const auto d_dot = at::scalar_tensor(1.5, options);
  actual = initial.clone();
  clad::custom_derivatives::at::dot_pullback(lhs, rhs, d_dot, &actual, &actual);
  require_close(actual, initial + 1.5 * (lhs + rhs),
                "shared dot destination was accumulated incorrectly");

  at::Tensor lazy_actual;
  clad::custom_derivatives::at::dot_pullback(lhs, lhs, d_dot, &lazy_actual,
                                             &lazy_actual);
  require_close(lazy_actual, 3 * lhs,
                "shared dot destination was not materialized correctly");
}

void check_lazy_generated_gradient() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(0.5, 6.0, 6, options).reshape({2, 3});
  const auto original = input.clone();
  at::Tensor actual;

  auto gradient = clad::gradient(tensor_sum_loss, "input");
  gradient.execute(input, &actual);

  require_true(actual.defined(),
               "generated gradient did not materialize its output");
  require_true(actual.is_contiguous(),
               "generated gradient did not materialize an expanded view");
  require_true(!actual.is_alias_of(input),
               "generated gradient aliases its primal input");
  require_close(actual, at::ones_like(input),
                "lazy generated Tensor gradient is incorrect");
  actual.add_(1.0);
  require_close(input, original,
                "mutating a lazy generated gradient changed its input");

  const auto lhs = at::linspace(0.5, 2.0, 8, options).reshape({2, 1, 4});
  const auto rhs = at::linspace(1.0, 3.0, 3, options).reshape({1, 3, 1});
  at::Tensor d_lhs;
  at::Tensor d_rhs;
  auto broadcast_gradient = clad::gradient(broadcast_mul_loss, "lhs, rhs");
  broadcast_gradient.execute(lhs, rhs, &d_lhs, &d_rhs);

  const auto output_sizes = ::std::vector<int64_t>{2, 3, 4};
  require_close(d_lhs, rhs.expand(output_sizes).sum_to_size(lhs.sizes()),
                "lazy broadcast lhs gradient is incorrect");
  require_close(d_rhs, lhs.expand(output_sizes).sum_to_size(rhs.sizes()),
                "lazy broadcast rhs gradient is incorrect");
  require_true(!d_lhs.is_alias_of(rhs) && !d_rhs.is_alias_of(lhs),
               "lazy broadcast gradients alias their primal inputs");
}

void check_native_relu_backward() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::tensor({-2.0, -0.0, 0.0, 1.5}, options);
  const auto d_output = at::tensor({0.5, 1.0, 1.5, 2.0}, options);
  auto actual = at::full_like(input, 3.0);
  const auto expected =
      actual + at::threshold_backward(d_output, input, /*threshold=*/0);

  clad::custom_derivatives::at::relu_pullback(input, d_output, &actual);
  require_close(actual, expected,
                "native ReLU backward accumulation is incorrect");
}

void check_partial_pullback_inputs() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto lhs = at::linspace(0.5, 2.0, 8, options).reshape({2, 1, 4});
  const auto rhs = at::linspace(1.0, 3.0, 3, options).reshape({1, 3, 1});
  const auto d_output = at::ones({2, 3, 4}, options);

  auto lhs_gradient = at::zeros_like(lhs);
  clad::custom_derivatives::at::mul_pullback(lhs, rhs, d_output, &lhs_gradient,
                                             /*d_rhs=*/nullptr);
  require_close(lhs_gradient, rhs.expand({2, 3, 4}).sum_to_size(lhs.sizes()),
                "masked lhs pullback gradient is incorrect");

  auto rhs_gradient = at::zeros_like(rhs);
  clad::custom_derivatives::at::mul_pullback(lhs, rhs, d_output,
                                             /*d_lhs=*/nullptr, &rhs_gradient);
  require_close(rhs_gradient, lhs.expand({2, 3, 4}).sum_to_size(rhs.sizes()),
                "masked rhs pullback gradient is incorrect");
}

void check_namespace_alias_composition() {
  const auto input = torch_api::linspace(
      -2.0, 2.0, 17, torch_api::TensorOptions().dtype(torch_api::kFloat32));
  auto gradient = torch_api::zeros_like(input);
  auto clad_gradient = clad::gradient(namespace_alias_loss, "input");
  clad_gradient.execute(input, &gradient);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_loss =
      torch_api::dot(namespace_alias_utility(native_input), native_input);
  auto native_gradient =
      torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(gradient, native_gradient,
                "namespace-aliased Torch gradient is incorrect");
}

void check_operator_composition() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto lhs = torch::linspace(0.5, 2.0, 17, options);
  const auto rhs = torch::linspace(1.0, 3.0, 17, options);
  auto lhs_gradient = torch::zeros_like(lhs);
  auto rhs_gradient = torch::zeros_like(rhs);
  auto clad_gradient = clad::gradient(operator_loss, "lhs, rhs");
  clad_gradient.execute(lhs, rhs, &lhs_gradient, &rhs_gradient);

  auto native_lhs = lhs.detach().clone().set_requires_grad(true);
  auto native_rhs = rhs.detach().clone().set_requires_grad(true);
  auto native_loss =
      torch::dot(operator_utility(native_lhs, native_rhs), native_lhs);
  auto native_gradients =
      torch::autograd::grad({native_loss}, {native_lhs, native_rhs});
  require_close(lhs_gradient, native_gradients[0],
                "Tensor operator lhs gradient is incorrect");
  require_close(rhs_gradient, native_gradients[1],
                "Tensor operator rhs gradient is incorrect");
}

void check_method_composition() {
  const auto options = torch::TensorOptions().dtype(torch::kFloat32);
  const auto lhs = torch::linspace(0.5, 2.0, 17, options);
  const auto rhs = torch::linspace(1.0, 3.0, 17, options);
  auto lhs_gradient = torch::zeros_like(lhs);
  auto rhs_gradient = torch::zeros_like(rhs);
  auto clad_gradient = clad::gradient(method_loss, "lhs, rhs");
  clad_gradient.execute(lhs, rhs, &lhs_gradient, &rhs_gradient);

  auto native_lhs = lhs.detach().clone().set_requires_grad(true);
  auto native_rhs = rhs.detach().clone().set_requires_grad(true);
  auto native_loss = method_utility(native_lhs, native_rhs).dot(native_lhs);
  auto native_gradients =
      torch::autograd::grad({native_loss}, {native_lhs, native_rhs});
  require_close(lhs_gradient, native_gradients[0],
                "Tensor method lhs gradient is incorrect");
  require_close(rhs_gradient, native_gradients[1],
                "Tensor method rhs gradient is incorrect");
}

void check_partial_gradients_do_not_alias_inputs() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto lhs = at::linspace(0.5, 2.0, 17, options);
  const auto rhs = at::linspace(1.0, 3.0, 17, options);
  const auto original_lhs = lhs.clone();
  const auto original_rhs = rhs.clone();

  auto lhs_gradient = at::zeros_like(lhs);
  auto lhs_only = clad::gradient(partial_gradient_loss, "lhs");
  lhs_only.execute(lhs, rhs, &lhs_gradient);
  require_close(lhs_gradient, 2 * original_lhs * original_rhs,
                "partial lhs gradient is incorrect");
  require_close(lhs, original_lhs,
                "partial lhs gradient modified the lhs input");
  require_close(rhs, original_rhs,
                "partial lhs gradient modified the rhs input");

  auto rhs_gradient = at::zeros_like(rhs);
  auto rhs_only = clad::gradient(partial_gradient_loss, "rhs");
  rhs_only.execute(lhs, rhs, &rhs_gradient);
  require_close(rhs_gradient, original_lhs * original_lhs,
                "partial rhs gradient is incorrect");
  require_close(lhs, original_lhs,
                "partial rhs gradient modified the lhs input");
  require_close(rhs, original_rhs,
                "partial rhs gradient modified the rhs input");
}

void check_sum_shape_support() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  auto gradient = clad::gradient(tensor_sum_loss, "input");
  auto check = [&](const at::Tensor& input) {
    const auto original = input.clone();
    auto actual = at::zeros_like(input);
    gradient.execute(input, &actual);
    require_close(actual, at::ones_like(input),
                  "Tensor sum gradient is incorrect");
    require_close(input, original, "Tensor sum modified its input");
  };

  check(at::scalar_tensor(2.0, options));
  check(at::linspace(0.5, 3.0, 6, options).reshape({2, 3}));
  check(at::linspace(0.5, 12.0, 24, options).reshape({2, 3, 4}));
  check(at::empty({2, 0, 3}, options));
}

void check_sum_dim_method() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(0.5, 12.0, 24, options).reshape({2, 3, 4});
  const auto weights = at::linspace(0.5, 4.0, 8, options).reshape({2, 4});
  auto actual = at::zeros_like(input);
  auto gradient = clad::gradient(tensor_sum_dim_method_loss, "input");
  gradient.execute(input, weights, &actual);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_loss = native_input.sum(/*dim=*/1).mul(weights).sum();
  auto expected = torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(actual, expected,
                "Tensor method sum(dim) gradient is incorrect");

  const auto keepdim_weights =
      at::linspace(0.5, 1.5, 3, options).reshape({1, 3, 1});
  actual.zero_();
  auto dims_gradient = clad::gradient(tensor_sum_dims_method_loss, "input");
  dims_gradient.execute(input, keepdim_weights, &actual);

  native_input = input.detach().clone().set_requires_grad(true);
  native_loss =
      native_input.sum({-3, -1}, /*keepdim=*/true).mul(keepdim_weights).sum();
  expected = torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(actual, expected,
                "Tensor method sum(dims, keepdim) gradient is incorrect");
}

void check_sum_dims_case(const at::Tensor& input, const at::Tensor& weights,
                         at::OptionalIntArrayRef dims, bool keepdim) {
  const auto original = input.clone();
  auto actual = at::zeros_like(input);
  auto gradient = clad::gradient(tensor_sum_dims_loss, "input");
  gradient.execute(input, weights, dims, keepdim, &actual);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_loss =
      at::mul(at::sum(native_input, dims, keepdim), weights).sum();
  auto expected = torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(actual, expected, "Tensor sum(dim) gradient is incorrect");
  require_close(input, original, "Tensor sum(dim) modified its input");
}

void check_sum_dims() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(0.5, 12.0, 24, options).reshape({2, 3, 4});

  const std::vector<int64_t> middle_dim{1};
  check_sum_dims_case(input, at::linspace(0.5, 4.0, 8, options).reshape({2, 4}),
                      middle_dim,
                      /*keepdim=*/false);

  const std::vector<int64_t> outer_dims{-1, -3};
  check_sum_dims_case(input, at::linspace(0.5, 1.5, 3, options), outer_dims,
                      /*keepdim=*/false);
  check_sum_dims_case(
      input, at::linspace(0.5, 1.5, 3, options).reshape({1, 3, 1}), outer_dims,
      /*keepdim=*/true);

  const std::vector<int64_t> all_dims;
  check_sum_dims_case(input, at::scalar_tensor(2.0, options), all_dims,
                      /*keepdim=*/false);
  check_sum_dims_case(input, at::ones({1, 1, 1}, options), all_dims,
                      /*keepdim=*/true);
  check_sum_dims_case(input, at::scalar_tensor(3.0, options),
                      at::OptionalIntArrayRef(), /*keepdim=*/false);

  const auto scalar = at::scalar_tensor(2.0, options);
  const std::vector<int64_t> scalar_dim{-1};
  check_sum_dims_case(scalar, at::scalar_tensor(4.0, options), scalar_dim,
                      /*keepdim=*/false);
  check_sum_dims_case(scalar, at::scalar_tensor(5.0, options), scalar_dim,
                      /*keepdim=*/true);

  const auto empty = at::empty({2, 0, 3}, options);
  check_sum_dims_case(empty, at::empty({0}, options), outer_dims,
                      /*keepdim=*/false);
  check_sum_dims_case(empty, at::ones({2, 3}, options), middle_dim,
                      /*keepdim=*/false);
}

void check_sum_dim_contract() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::ones({2, 3, 4}, options);
  const auto weights = at::scalar_tensor(1.0, options);
  auto gradient = clad::gradient(tensor_sum_dims_loss, "input");

  require_c10_error(
      [&] {
        const std::vector<int64_t> dims{3};
        auto actual = at::zeros_like(input);
        gradient.execute(input, weights, dims, false, &actual);
      },
      "out-of-range sum dimension was not rejected");

  require_c10_error(
      [&] {
        const std::vector<int64_t> dims{1, -2};
        auto actual = at::zeros_like(input);
        gradient.execute(input, weights, dims, false, &actual);
      },
      "duplicate sum dimensions were not rejected");
}

void check_default_method_parameters() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input =
      at::linspace(-2.0, 3.0, 120, options).reshape({2, 3, 4, 5});
  const auto bias = at::linspace(0.2, 1.0, 15, options).reshape({1, 3, 1, 5});
  const auto offset = at::linspace(-0.5, 0.5, 8, options).reshape({2, 1, 4, 1});
  const auto weights = at::linspace(0.5, 1.5, 15, options).reshape({1, 3, 5});
  constexpr int64_t dim = -2;

  auto input_gradient = at::zeros_like(input);
  auto bias_gradient = at::zeros_like(bias);
  auto offset_gradient = at::zeros_like(offset);
  auto weights_gradient = at::zeros_like(weights);
  auto gradient = clad::gradient(default_method_parameter_loss,
                                 "input, bias, offset, weights");
  gradient.execute(input, bias, offset, weights, dim, &input_gradient,
                   &bias_gradient, &offset_gradient, &weights_gradient);

  auto native_input = input.detach().clone().set_requires_grad(true);
  auto native_bias = bias.detach().clone().set_requires_grad(true);
  auto native_offset = offset.detach().clone().set_requires_grad(true);
  auto native_weights = weights.detach().clone().set_requires_grad(true);
  auto native_loss = native_input.add(native_bias)
                         .sub(native_offset)
                         .sum(dim)
                         .mul(native_weights)
                         .sum();
  auto native_gradients =
      torch::autograd::grad({native_loss}, {native_input, native_bias,
                                            native_offset, native_weights});

  require_close(input_gradient, native_gradients[0],
                "default method input gradient is incorrect");
  require_close(bias_gradient, native_gradients[1],
                "default method bias gradient is incorrect");
  require_close(offset_gradient, native_gradients[2],
                "default method offset gradient is incorrect");
  require_close(weights_gradient, native_gradients[3],
                "default method weights gradient is incorrect");

  require_c10_error(
      [&] {
        input_gradient.zero_();
        bias_gradient.zero_();
        offset_gradient.zero_();
        weights_gradient.zero_();
        gradient.execute(input, bias, offset, weights, /*dim=*/-5,
                         &input_gradient, &bias_gradient, &offset_gradient,
                         &weights_gradient);
      },
      "out-of-range scalar sum dimension was not rejected");
}

void check_default_function_parameters() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(-1.5, 2.5, 24, options).reshape({2, 3, 4});
  auto gradient =
      clad::gradient(default_function_parameter_loss, "input, weights");

  auto check = [&](const at::Tensor& weights, at::OptionalIntArrayRef dims) {
    auto input_gradient = at::zeros_like(input);
    auto weights_gradient = at::zeros_like(weights);
    gradient.execute(input, weights, dims, &input_gradient, &weights_gradient);

    auto native_input = input.detach().clone().set_requires_grad(true);
    auto native_weights = weights.detach().clone().set_requires_grad(true);
    auto native_loss =
        at::mul(at::sum(native_input, dims), native_weights).sum();
    auto native_gradients =
        torch::autograd::grad({native_loss}, {native_input, native_weights});
    require_close(input_gradient, native_gradients[0],
                  "default function input gradient is incorrect");
    require_close(weights_gradient, native_gradients[1],
                  "default function weights gradient is incorrect");
  };

  const std::vector<int64_t> mixed_dims{-1, -3};
  check(at::linspace(0.5, 1.5, 3, options), mixed_dims);

  const std::vector<int64_t> empty_dims;
  check(at::scalar_tensor(2.0, options), empty_dims);
}

void check_explicit_sum_options() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(-1.0, 2.0, 24, options).reshape({2, 3, 4});
  const std::vector<int64_t> dims{0, -1};
  auto gradient = clad::gradient(explicit_sum_options_loss, "input, weights");

  auto check = [&](const at::Tensor& weights, bool keepdim) {
    auto input_gradient = at::zeros_like(input);
    auto weights_gradient = at::zeros_like(weights);
    gradient.execute(input, weights, dims, keepdim, &input_gradient,
                     &weights_gradient);

    auto native_input = input.detach().clone().set_requires_grad(true);
    auto native_weights = weights.detach().clone().set_requires_grad(true);
    auto native_loss = native_input.sum(dims, keepdim, at::kFloat)
                           .mul(native_weights)
                           .sum(at::ScalarType::Float);
    auto native_gradients =
        torch::autograd::grad({native_loss}, {native_input, native_weights});
    require_close(input_gradient, native_gradients[0],
                  "explicit sum options input gradient is incorrect");
    require_close(weights_gradient, native_gradients[1],
                  "explicit sum options weights gradient is incorrect");
  };

  check(at::linspace(0.5, 1.5, 3, options), /*keepdim=*/false);
  check(at::linspace(0.5, 1.5, 3, options).reshape({1, 3, 1}),
        /*keepdim=*/true);

  require_c10_error(
      [&] {
        auto input_gradient = at::zeros_like(input);
        auto dtype_gradient =
            clad::gradient(unsupported_sum_dtype_loss, "input");
        dtype_gradient.execute(input, &input_gradient);
      },
      "sum dtype conversion was not rejected");
}

template <typename Gradient, typename Operation>
void check_binary_gradients(Gradient& gradient, Operation operation,
                            const at::Tensor& lhs, const at::Tensor& rhs,
                            const char* lhs_message, const char* rhs_message) {
  const auto original_lhs = lhs.clone();
  const auto original_rhs = rhs.clone();
  auto lhs_gradient = at::zeros_like(lhs);
  auto rhs_gradient = at::zeros_like(rhs);
  gradient.execute(lhs, rhs, &lhs_gradient, &rhs_gradient);

  auto native_lhs = lhs.detach().clone().set_requires_grad(true);
  auto native_rhs = rhs.detach().clone().set_requires_grad(true);
  auto native_loss = operation(native_lhs, native_rhs).sum();
  auto native_gradients =
      torch::autograd::grad({native_loss}, {native_lhs, native_rhs});

  require_close(lhs_gradient, native_gradients[0], lhs_message);
  require_close(rhs_gradient, native_gradients[1], rhs_message);
  require_close(lhs, original_lhs, "broadcasting modified the lhs input");
  require_close(rhs, original_rhs, "broadcasting modified the rhs input");
}

void check_broadcast_case(const at::Tensor& lhs, const at::Tensor& rhs) {
  auto add_gradient = clad::gradient(broadcast_add_loss, "lhs, rhs");
  check_binary_gradients(
      add_gradient,
      [](const at::Tensor& x, const at::Tensor& y) { return at::add(x, y); },
      lhs, rhs, "broadcast add lhs gradient is incorrect",
      "broadcast add rhs gradient is incorrect");

  auto sub_gradient = clad::gradient(broadcast_sub_loss, "lhs, rhs");
  check_binary_gradients(
      sub_gradient,
      [](const at::Tensor& x, const at::Tensor& y) { return at::sub(x, y); },
      lhs, rhs, "broadcast sub lhs gradient is incorrect",
      "broadcast sub rhs gradient is incorrect");

  auto mul_gradient = clad::gradient(broadcast_mul_loss, "lhs, rhs");
  check_binary_gradients(
      mul_gradient,
      [](const at::Tensor& x, const at::Tensor& y) { return at::mul(x, y); },
      lhs, rhs, "broadcast mul lhs gradient is incorrect",
      "broadcast mul rhs gradient is incorrect");

  auto div_gradient = clad::gradient(broadcast_div_loss, "lhs, rhs");
  check_binary_gradients(
      div_gradient,
      [](const at::Tensor& x, const at::Tensor& y) { return at::div(x, y); },
      lhs, rhs, "broadcast div lhs gradient is incorrect",
      "broadcast div rhs gradient is incorrect");
}

void check_broadcasting() {
  const auto options = at::TensorOptions().dtype(at::kFloat);

  const auto matrix_lhs = at::linspace(0.5, 3.0, 6, options).reshape({2, 3});
  const auto matrix_rhs = at::linspace(1.0, 2.0, 6, options).reshape({2, 3});
  check_broadcast_case(matrix_lhs, matrix_rhs);

  const auto lhs = at::linspace(0.5, 2.0, 8, options).reshape({2, 1, 4});
  const auto rhs = at::linspace(1.0, 3.0, 3, options).reshape({1, 3, 1});
  check_broadcast_case(lhs, rhs);
  check_broadcast_case(rhs, lhs);

  const auto scalar = at::scalar_tensor(1.5, options);
  check_broadcast_case(lhs, scalar);
  check_broadcast_case(scalar, lhs);

  const auto empty = at::empty({2, 0, 3}, options);
  const auto row = at::ones({1, 3}, options);
  check_broadcast_case(empty, row);
  check_broadcast_case(row, empty);
}

void check_partial_broadcast_gradients() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto lhs = at::linspace(0.5, 2.0, 8, options).reshape({2, 1, 4});
  const auto rhs = at::linspace(1.0, 3.0, 3, options).reshape({1, 3, 1});
  const auto original_lhs = lhs.clone();
  const auto original_rhs = rhs.clone();

  auto native_lhs = lhs.detach().clone().set_requires_grad(true);
  auto native_rhs = rhs.detach().clone().set_requires_grad(true);
  auto native_loss = at::mul(native_lhs, native_rhs).sum();
  auto native_gradients =
      torch::autograd::grad({native_loss}, {native_lhs, native_rhs});

  auto lhs_gradient = at::zeros_like(lhs);
  auto lhs_only = clad::gradient(broadcast_mul_loss, "lhs");
  lhs_only.execute(lhs, rhs, &lhs_gradient);
  require_close(lhs_gradient, native_gradients[0],
                "partial broadcast lhs gradient is incorrect");

  auto rhs_gradient = at::zeros_like(rhs);
  auto rhs_only = clad::gradient(broadcast_mul_loss, "rhs");
  rhs_only.execute(lhs, rhs, &rhs_gradient);
  require_close(rhs_gradient, native_gradients[1],
                "partial broadcast rhs gradient is incorrect");
  require_close(lhs, original_lhs,
                "partial broadcasting modified the lhs input");
  require_close(rhs, original_rhs,
                "partial broadcasting modified the rhs input");
}

template <typename Gradient, typename NativeLoss, typename... Args>
void check_view_gradient(Gradient& gradient, NativeLoss native_loss,
                         const at::Tensor& input, const char* message,
                         const Args&... args) {
  const auto original = input.clone();
  at::Tensor actual;
  gradient.execute(input, args..., &actual);

  auto native_input = input.detach().set_requires_grad(true);
  auto expected =
      torch::autograd::grad({native_loss(native_input)}, {native_input})[0];
  require_close(actual, expected, message);
  require_close(input, original, "view differentiation modified its input");
  require_true(!actual.is_alias_of(input),
               "view gradient aliases its primal input");
}

void check_strided_inputs() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto base = at::linspace(-2.0, 3.0, 24, options).reshape({2, 3, 4});
  const auto input = base.permute({2, 0, 1});
  const auto weights = at::linspace(0.5, 1.5, 6, options).reshape({1, 2, 3});
  require_true(!input.is_contiguous(),
               "strided input test unexpectedly became contiguous");

  auto gradient = clad::gradient(strided_input_loss, "input");
  check_view_gradient(
      gradient,
      [&](const at::Tensor& native_input) {
        return native_input.mul(weights).add(native_input).relu().sum();
      },
      input, "strided input gradient is incorrect", weights);

  auto strided_destination = at::zeros_like(input);
  require_true(!strided_destination.is_contiguous(),
               "strided gradient destination unexpectedly became contiguous");
  const auto destination_strides = strided_destination.strides().vec();
  gradient.execute(input, weights, &strided_destination);
  auto native_input = input.detach().set_requires_grad(true);
  auto native_loss = native_input.mul(weights).add(native_input).relu().sum();
  auto expected = torch::autograd::grad({native_loss}, {native_input})[0];
  require_close(strided_destination, expected,
                "preallocated strided gradient is incorrect");
  require_true(strided_destination.strides().vec() == destination_strides,
               "gradient accumulation changed destination strides");

  const auto vector =
      at::linspace(0.5, 8.0, 16, options).reshape({8, 2}).select(1, 0);
  require_true(!vector.is_contiguous(),
               "strided vector test unexpectedly became contiguous");
  auto actual = at::zeros_like(vector);
  auto dot_gradient = clad::gradient(at_loss, "input");
  dot_gradient.execute(vector, &actual);

  auto native_vector = vector.detach().set_requires_grad(true);
  native_loss = at::dot(at_utility(native_vector), native_vector);
  expected = torch::autograd::grad({native_loss}, {native_vector})[0];
  require_close(actual, expected, "strided dot input gradient is incorrect");
}

void check_view_pullbacks() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::zeros({2, 3, 4}, options);

  const auto transpose_output_adjoint =
      at::linspace(0.5, 12.0, 24, options).reshape({2, 3, 4}).transpose(0, 2);
  require_true(!transpose_output_adjoint.is_contiguous(),
               "transpose pullback seed unexpectedly became contiguous");
  int64_t d_dim0 = 0;
  int64_t d_dim1 = 0;
  at::Tensor transpose_input_adjoint;
  clad::custom_derivatives::at::transpose_pullback(
      input, /*dim0=*/0, /*dim1=*/2, transpose_output_adjoint,
      &transpose_input_adjoint, &d_dim0, &d_dim1);
  require_close(transpose_input_adjoint,
                transpose_output_adjoint.transpose(0, 2),
                "transpose pullback did not invert the view");
  require_true(!transpose_input_adjoint.is_alias_of(transpose_output_adjoint),
               "transpose pullback gradient aliases its output adjoint");

  const std::vector<int64_t> dims{-1, 0, 1};
  const auto permute_output_adjoint =
      at::linspace(0.5, 12.0, 24, options).reshape({2, 3, 4}).permute(dims);
  require_true(!permute_output_adjoint.is_contiguous(),
               "permute pullback seed unexpectedly became contiguous");
  at::IntArrayRef d_dims;
  at::Tensor permute_input_adjoint;
  clad::custom_derivatives::at::permute_pullback(
      input, dims, permute_output_adjoint, &permute_input_adjoint, &d_dims);
  require_close(permute_input_adjoint,
                permute_output_adjoint.permute({1, 2, 0}),
                "permute pullback did not apply the inverse permutation");
  require_true(!permute_input_adjoint.is_alias_of(permute_output_adjoint),
               "permute pullback gradient aliases its output adjoint");

  const at::Tensor zero_output_adjoint;
  const auto initial = at::full_like(input, 2.0);
  transpose_input_adjoint = initial.clone();
  permute_input_adjoint = initial.clone();
  clad::custom_derivatives::at::transpose_pullback(
      input, /*dim0=*/0, /*dim1=*/2, zero_output_adjoint,
      &transpose_input_adjoint, &d_dim0, &d_dim1);
  clad::custom_derivatives::at::permute_pullback(
      input, dims, zero_output_adjoint, &permute_input_adjoint, &d_dims);
  require_close(transpose_input_adjoint, initial,
                "undefined transpose adjoint was propagated");
  require_close(permute_input_adjoint, initial,
                "undefined permute adjoint was propagated");
}

void check_transpose_gradients() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(-2.0, 3.0, 24, options).reshape({2, 3, 4});
  const auto weights = at::linspace(0.5, 1.5, 24, options).reshape({4, 3, 2});
  constexpr int64_t dim0 = -3;
  constexpr int64_t dim1 = -1;

  auto method_gradient = clad::gradient(transpose_method_loss, "input");
  check_view_gradient(
      method_gradient,
      [&](const at::Tensor& native_input) {
        return native_input.transpose(dim0, dim1).mul(weights).sum();
      },
      input, "Tensor method transpose gradient is incorrect", weights, dim0,
      dim1);

  auto function_gradient = clad::gradient(transpose_function_loss, "input");
  check_view_gradient(
      function_gradient,
      [&](const at::Tensor& native_input) {
        return at::transpose(native_input, dim0, dim1).mul(weights).sum();
      },
      input, "at::transpose gradient is incorrect", weights, dim0, dim1);
}

void check_permute_gradients() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::linspace(-2.0, 3.0, 24, options).reshape({2, 3, 4});
  const auto weights = at::linspace(0.5, 1.5, 24, options).reshape({4, 2, 3});
  const std::vector<int64_t> dims{-1, 0, 1};

  auto method_gradient = clad::gradient(permute_method_loss, "input");
  check_view_gradient(
      method_gradient,
      [&](const at::Tensor& native_input) {
        return native_input.permute(dims).mul(weights).sum();
      },
      input, "Tensor method permute gradient is incorrect", weights,
      at::IntArrayRef(dims));

  auto function_gradient = clad::gradient(permute_function_loss, "input");
  check_view_gradient(
      function_gradient,
      [&](const at::Tensor& native_input) {
        return at::permute(native_input, dims).mul(weights).sum();
      },
      input, "at::permute gradient is incorrect", weights,
      at::IntArrayRef(dims));
}

void check_chained_view_gradients() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input =
      at::linspace(-2.0, 3.0, 120, options).reshape({2, 3, 4, 5});
  const std::vector<int64_t> dims{2, 0, 3, 1};
  constexpr int64_t dim0 = 1;
  constexpr int64_t dim1 = -1;
  const auto weights =
      at::linspace(0.25, 1.25, 120, options).reshape({4, 3, 5, 2});

  auto gradient = clad::gradient(chained_view_loss, "input");
  check_view_gradient(
      gradient,
      [&](const at::Tensor& native_input) {
        return native_input.permute(dims)
            .transpose(dim0, dim1)
            .mul(weights)
            .relu()
            .sum();
      },
      input, "chained permute/transpose gradient is incorrect", weights,
      at::IntArrayRef(dims), dim0, dim1);

  const auto empty = at::empty({2, 0, 3}, options);
  const std::vector<int64_t> empty_dims{2, 0, 1};
  const auto empty_weights = at::empty({3, 2, 0}, options);
  auto permute_gradient = clad::gradient(permute_method_loss, "input");
  check_view_gradient(
      permute_gradient,
      [&](const at::Tensor& native_input) {
        return native_input.permute(empty_dims).mul(empty_weights).sum();
      },
      empty, "empty permute gradient is incorrect", empty_weights,
      at::IntArrayRef(empty_dims));

  const auto scalar = at::scalar_tensor(2.0, options);
  const std::vector<int64_t> scalar_dims;
  check_view_gradient(
      permute_gradient,
      [&](const at::Tensor& native_input) {
        return native_input.permute(scalar_dims).mul(3.0).sum();
      },
      scalar, "scalar permute gradient is incorrect",
      at::scalar_tensor(3.0, options), at::IntArrayRef(scalar_dims));
}

void check_view_contract() {
  const auto options = at::TensorOptions().dtype(at::kFloat);
  const auto input = at::ones({2, 3, 4}, options);
  const auto weights = at::ones({2, 3, 4}, options);
  auto transpose_gradient = clad::gradient(transpose_method_loss, "input");

  require_c10_error(
      [&] {
        at::Tensor actual;
        transpose_gradient.execute(input, weights, /*dim0=*/0, /*dim1=*/3,
                                   &actual);
      },
      "out-of-range transpose dimension was not rejected");

  auto permute_gradient = clad::gradient(permute_method_loss, "input");
  require_c10_error(
      [&] {
        const std::vector<int64_t> dims{0, 1};
        at::Tensor actual;
        permute_gradient.execute(input, weights, dims, &actual);
      },
      "short permutation was not rejected");

  require_c10_error(
      [&] {
        const std::vector<int64_t> dims{0, 1, 1};
        at::Tensor actual;
        permute_gradient.execute(input, weights, dims, &actual);
      },
      "duplicate permutation dimensions were not rejected");
}

void check_input_contract() {
  require_c10_error(
      [] {
        const auto input = at::ones({4}, at::kDouble);
        auto gradient = at::zeros_like(input);
        auto clad_gradient = clad::gradient(at_loss, "input");
        clad_gradient.execute(input, &gradient);
      },
      "float64 input was not rejected");

  require_c10_error(
      [] {
        const auto input = torch::ones({4}, torch::kFloat);
        const auto bias = torch::ones({2, 3}, torch::kFloat);
        auto input_gradient = torch::zeros_like(input);
        auto bias_gradient = torch::zeros_like(bias);
        auto clad_gradient = clad::gradient(torch_loss, "input, bias");
        clad_gradient.execute(input, bias, &input_gradient, &bias_gradient);
      },
      "incompatible shapes were not rejected");

  require_c10_error(
      [] {
        const auto input = at::ones({2, 3}, at::kFloat).to_sparse();
        clad::torch::detail::validate_tensor(input);
      },
      "non-strided Tensor layout was not rejected");
}

} // namespace

int main() {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  check_torch_alias_composition();
  check_at_namespace_composition();
  check_direct_return_composition();
  check_tensor_adjoint_lifecycle();
  check_lazy_accumulation();
  check_undefined_output_adjoint();
  check_shared_pullback_destination();
  check_lazy_generated_gradient();
  check_native_relu_backward();
  check_partial_pullback_inputs();
  check_namespace_alias_composition();
  check_operator_composition();
  check_method_composition();
  check_partial_gradients_do_not_alias_inputs();
  check_sum_shape_support();
  check_sum_dim_method();
  check_sum_dims();
  check_sum_dim_contract();
  check_default_method_parameters();
  check_default_function_parameters();
  check_explicit_sum_options();
  check_broadcasting();
  check_partial_broadcast_gradients();
  check_strided_inputs();
  check_view_pullbacks();
  check_transpose_gradients();
  check_permute_gradients();
  check_chained_view_gradients();
  check_view_contract();
  check_input_contract();
  std::cout << "Clad Torch operator checks passed\n";
  return 0;
}
