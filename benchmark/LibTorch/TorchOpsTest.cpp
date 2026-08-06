#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/TorchBuiltins.h" // IWYU pragma: keep

#include <torch/torch.h> // IWYU pragma: keep

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

namespace torch_api = torch;

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
        const auto bias = torch::ones({1}, torch::kFloat);
        auto input_gradient = torch::zeros_like(input);
        auto bias_gradient = torch::zeros_like(bias);
        auto clad_gradient = clad::gradient(torch_loss, "input, bias");
        clad_gradient.execute(input, bias, &input_gradient, &bias_gradient);
      },
      "broadcasting input was not rejected");

  require_c10_error(
      [] {
        const auto input =
            torch::ones({4, 2}, torch::kFloat).select(/*dim=*/1, /*index=*/0);
        auto gradient = torch::zeros_like(input);
        auto clad_gradient = clad::gradient(at_loss, "input");
        clad_gradient.execute(input, &gradient);
      },
      "noncontiguous input was not rejected");
}

} // namespace

int main() {
  static_assert(std::is_same_v<torch::Tensor, at::Tensor>);
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  check_torch_alias_composition();
  check_at_namespace_composition();
  check_direct_return_composition();
  check_namespace_alias_composition();
  check_operator_composition();
  check_method_composition();
  check_input_contract();
  std::cout << "Clad Torch operator checks passed\n";
  return 0;
}
