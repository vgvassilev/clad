#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/TorchBuiltins.h" // IWYU pragma: keep

#include "benchmark/benchmark.h"

#include <torch/torch.h> // IWYU pragma: keep

#include <cstdint>

namespace {

float squared_error(const torch::Tensor& prediction,
                    const torch::Tensor& target) {
  auto residual = torch::sub(prediction, target);
  auto loss = torch::dot(residual, residual);
  return loss.item<float>();
}

struct Inputs {
  torch::Tensor prediction{};
  torch::Tensor target{};
};

Inputs make_inputs(std::int64_t size, bool requires_grad) {
  const auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .requires_grad(requires_grad);
  return {torch::linspace(-1.0, 1.0, size, options),
          torch::linspace(0.75, -0.25, size, options)};
}

void set_items_processed(benchmark::State& state, std::int64_t size) {
  state.SetItemsProcessed(state.iterations() * size);
}

void BM_CladGradient(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_inputs(size, /*requires_grad=*/false);
  auto prediction_gradient = torch::zeros_like(inputs.prediction);
  auto target_gradient = torch::zeros_like(inputs.target);
  auto gradient = clad::gradient(squared_error, "prediction, target");

  for ([[maybe_unused]] auto _ : state) {
    prediction_gradient.zero_();
    target_gradient.zero_();
    gradient.execute(inputs.prediction, inputs.target, &prediction_gradient,
                     &target_gradient);
    benchmark::DoNotOptimize(prediction_gradient.data_ptr<float>());
    benchmark::DoNotOptimize(target_gradient.data_ptr<float>());
  }
  set_items_processed(state, size);
}

void BM_LibTorchAutograd(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_inputs(size, /*requires_grad=*/true);

  for ([[maybe_unused]] auto _ : state) {
    auto residual = torch::sub(inputs.prediction, inputs.target);
    auto loss = torch::dot(residual, residual);
    auto gradients =
        torch::autograd::grad({loss}, {inputs.prediction, inputs.target});
    benchmark::DoNotOptimize(gradients[0].data_ptr<float>());
    benchmark::DoNotOptimize(gradients[1].data_ptr<float>());
  }
  set_items_processed(state, size);
}

void configure(benchmark::internal::Benchmark* benchmark) {
  benchmark->RangeMultiplier(4)
      ->Range(256, 1 << 20)
      ->Unit(benchmark::kMicrosecond);
}

BENCHMARK(BM_CladGradient)->Apply(configure);
BENCHMARK(BM_LibTorchAutograd)->Apply(configure);

} // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
