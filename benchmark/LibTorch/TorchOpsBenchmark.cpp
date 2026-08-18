#include "benchmark/benchmark.h"

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/TorchBuiltins.h" // IWYU pragma: keep

#include <c10/core/CPUAllocator.h>
#include <torch/torch.h> // IWYU pragma: keep

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

// Track CPU Tensor storage during Google Benchmark's separate memory run while
// leaving the timed run on LibTorch's direct allocator path.
class TensorMemoryManager final : public benchmark::MemoryManager {
  static constexpr std::uint8_t AllocatorPriority =
      std::numeric_limits<std::uint8_t>::max();

  struct AllocationContext {
    c10::DataPtr Allocation;
    TensorMemoryManager* Manager;
    std::size_t Bytes;
    std::uint64_t Epoch;

    AllocationContext(c10::DataPtr allocation, TensorMemoryManager* manager,
                      std::size_t bytes, std::uint64_t epoch)
        : Allocation(std::move(allocation)), Manager(manager), Bytes(bytes),
          Epoch(epoch) {}
  };

  class TrackingAllocator final : public c10::Allocator {
    c10::Allocator* m_Underlying;
    TensorMemoryManager& m_Manager;

    static void release(void* opaque) {
      std::unique_ptr<AllocationContext> context(
          static_cast<AllocationContext*>(opaque));
      context->Manager->recordDeallocation(context->Bytes, context->Epoch);
    }

  public:
    TrackingAllocator(c10::Allocator* underlying, TensorMemoryManager& manager)
        : m_Underlying(underlying), m_Manager(manager) {}

    c10::DataPtr allocate(std::size_t bytes) override {
      if (!m_Manager.isRecording())
        return m_Underlying->allocate(bytes);

      auto allocation = m_Underlying->allocate(bytes);
      void* data = allocation.get();
      const auto device = allocation.device();
      const auto epoch = m_Manager.currentEpoch();
      auto* context = new AllocationContext(std::move(allocation), &m_Manager,
                                            bytes, epoch);
      m_Manager.recordAllocation(bytes, epoch);
      return {data, context, &release, device};
    }

    bool is_simple_data_ptr(const c10::DataPtr& data) const override {
      return m_Underlying->is_simple_data_ptr(data);
    }

    void copy_data(void* destination, const void* source,
                   std::size_t count) const override {
      m_Underlying->copy_data(destination, source, count);
    }
  };

  std::atomic<bool> m_Recording{false};
  std::atomic<std::uint64_t> m_Epoch{0};
  std::atomic<std::uint64_t> m_Allocations{0};
  std::atomic<std::uint64_t> m_TotalAllocatedBytes{0};
  std::atomic<std::uint64_t> m_LiveBytes{0};
  std::atomic<std::uint64_t> m_PeakBytes{0};
  c10::Allocator* m_Underlying;
  TrackingAllocator m_Allocator;

  bool isRecording() const {
    return m_Recording.load(std::memory_order_acquire);
  }

  std::uint64_t currentEpoch() const {
    return m_Epoch.load(std::memory_order_relaxed);
  }

  void recordAllocation(std::size_t bytes, std::uint64_t epoch) {
    if (!isRecording() || epoch != currentEpoch())
      return;

    m_Allocations.fetch_add(1, std::memory_order_relaxed);
    m_TotalAllocatedBytes.fetch_add(bytes, std::memory_order_relaxed);
    const auto live =
        m_LiveBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak = m_PeakBytes.load(std::memory_order_relaxed);
    while (live > peak && !m_PeakBytes.compare_exchange_weak(
                              peak, live, std::memory_order_relaxed,
                              std::memory_order_relaxed)) {
    }
  }

  void recordDeallocation(std::size_t bytes, std::uint64_t epoch) {
    if (!isRecording() || epoch != currentEpoch())
      return;
    m_LiveBytes.fetch_sub(bytes, std::memory_order_relaxed);
  }

public:
  TensorMemoryManager()
      : m_Underlying(c10::GetCPUAllocator()), m_Allocator(m_Underlying, *this) {
  }

  void install() { c10::SetCPUAllocator(&m_Allocator, AllocatorPriority); }

  void uninstall() { c10::SetCPUAllocator(m_Underlying, AllocatorPriority); }

  void Start() override {
    m_Recording.store(false, std::memory_order_release);
    install();
    m_Epoch.fetch_add(1, std::memory_order_relaxed);
    m_Allocations.store(0, std::memory_order_relaxed);
    m_TotalAllocatedBytes.store(0, std::memory_order_relaxed);
    m_LiveBytes.store(0, std::memory_order_relaxed);
    m_PeakBytes.store(0, std::memory_order_relaxed);
    m_Recording.store(true, std::memory_order_release);
  }

  void Stop(Result& result) override {
    m_Recording.store(false, std::memory_order_release);
    result.num_allocs = m_Allocations.load(std::memory_order_relaxed);
    result.max_bytes_used = m_PeakBytes.load(std::memory_order_relaxed);
    result.total_allocated_bytes =
        m_TotalAllocatedBytes.load(std::memory_order_relaxed);
    result.net_heap_growth = m_LiveBytes.load(std::memory_order_relaxed);
    uninstall();
  }
};

struct BenchmarkRegistration {
  TensorMemoryManager MemoryManager;

  BenchmarkRegistration() {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    benchmark::RegisterMemoryManager(&MemoryManager);
  }

  ~BenchmarkRegistration() { benchmark::RegisterMemoryManager(nullptr); }
};

static BenchmarkRegistration benchmark_registration;

struct BinaryInputs {
  torch::Tensor lhs{};
  torch::Tensor rhs{};
};

struct ReductionInputs {
  torch::Tensor input{};
  torch::Tensor weights{};
};

torch::TensorOptions options(bool requires_grad) {
  return torch::TensorOptions()
      .dtype(torch::kFloat32)
      .requires_grad(requires_grad);
}

BinaryInputs make_vector_inputs(std::int64_t size, bool requires_grad) {
  return {torch::linspace(-1.0, 1.0, size, options(requires_grad)),
          torch::linspace(0.75, -0.25, size, options(requires_grad))};
}

BinaryInputs make_broadcast_inputs(std::int64_t batch, std::int64_t channels,
                                   std::int64_t width, bool requires_grad) {
  auto lhs = torch::linspace(-1.0, 1.0, batch * width, options(requires_grad))
                 .reshape({batch, 1, width});
  auto rhs = torch::linspace(-0.75, 1.25, channels, options(requires_grad))
                 .reshape({1, channels, 1});
  return {::std::move(lhs), ::std::move(rhs)};
}

ReductionInputs make_reduction_inputs(std::int64_t batch, std::int64_t channels,
                                      std::int64_t width, bool requires_grad) {
  auto input = torch::linspace(-1.0, 1.0, batch * channels * width,
                               options(requires_grad))
                   .reshape({batch, channels, width});
  auto weights = torch::linspace(0.5, 1.5, channels, options(requires_grad));
  return {::std::move(input), ::std::move(weights)};
}

float squared_error(const torch::Tensor& prediction,
                    const torch::Tensor& target) {
  auto residual = prediction - target;
  auto loss = residual.dot(residual);
  return loss.item<float>();
}

float broadcast_relu_loss(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  auto product = lhs * rhs;
  auto activated = product.relu();
  auto loss = activated.sum();
  return loss.item<float>();
}

float sum_dims_loss(const torch::Tensor& input, const torch::Tensor& weights,
                    at::OptionalIntArrayRef dims) {
  auto reduced = input.sum(dims, /*keepdim=*/false);
  auto weighted = reduced * weights;
  auto loss = weighted.sum();
  return loss.item<float>();
}

void report_work(benchmark::State& state, std::int64_t elements,
                 std::int64_t problem_bytes) {
  state.SetItemsProcessed(state.iterations() * elements);
  state.SetBytesProcessed(state.iterations() * problem_bytes);
}

std::int64_t volume(const benchmark::State& state) {
  return state.range(0) * state.range(1) * state.range(2);
}

float materialized_gradient_checksum(const torch::Tensor& first,
                                     const torch::Tensor& second) {
  torch::NoGradGuard guard;
  auto first_dense = first.contiguous();
  auto second_dense = second.contiguous();
  return first_dense.sum().item<float>() + second_dense.sum().item<float>();
}

float materialized_gradient_checksum(const torch::Tensor& gradient) {
  torch::NoGradGuard guard;
  return gradient.contiguous().sum().item<float>();
}

void composite_relu_vjp(const torch::Tensor& input,
                        const torch::Tensor& d_output, torch::Tensor* d_input) {
  clad::torch::detail::validate_tensor(input);
  clad::torch::detail::validate_tensor(d_output);
  torch::NoGradGuard guard;
  auto mask = torch::gt(input, 0).to(input.scalar_type());
  clad::torch::detail::accumulate(d_input, torch::mul(d_output, mask));
}

void BM_ReluVJPComposite(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_vector_inputs(size, /*requires_grad=*/false);
  auto input_gradient = torch::zeros_like(inputs.lhs);

  for (auto _ : state) {
    input_gradient.zero_();
    composite_relu_vjp(inputs.lhs, inputs.rhs, &input_gradient);
    auto checksum = materialized_gradient_checksum(input_gradient);
    benchmark::DoNotOptimize(checksum);
  }
  report_work(state, size, 3 * size * sizeof(float));
}

void BM_ReluVJPNative(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_vector_inputs(size, /*requires_grad=*/false);
  auto input_gradient = torch::zeros_like(inputs.lhs);

  for (auto _ : state) {
    input_gradient.zero_();
    clad::custom_derivatives::at::relu_pullback(inputs.lhs, inputs.rhs,
                                                &input_gradient);
    auto checksum = materialized_gradient_checksum(input_gradient);
    benchmark::DoNotOptimize(checksum);
  }
  report_work(state, size, 3 * size * sizeof(float));
}

void BM_SquaredErrorForward(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_vector_inputs(size, /*requires_grad=*/false);
  torch::NoGradGuard guard;

  for (auto _ : state) {
    auto loss = squared_error(inputs.lhs, inputs.rhs);
    benchmark::DoNotOptimize(loss);
  }
  report_work(state, size, 2 * size * sizeof(float));
}

void BM_SquaredErrorClad(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_vector_inputs(size, /*requires_grad=*/false);
  auto lhs_gradient = torch::zeros_like(inputs.lhs);
  auto rhs_gradient = torch::zeros_like(inputs.rhs);
  auto gradient = clad::gradient(squared_error, "prediction, target");

  for (auto _ : state) {
    lhs_gradient.zero_();
    rhs_gradient.zero_();
    gradient.execute(inputs.lhs, inputs.rhs, &lhs_gradient, &rhs_gradient);
    auto checksum = materialized_gradient_checksum(lhs_gradient, rhs_gradient);
    benchmark::DoNotOptimize(checksum);
  }
  report_work(state, size, 2 * size * sizeof(float));
}

void BM_SquaredErrorAutograd(benchmark::State& state) {
  const auto size = state.range(0);
  const auto inputs = make_vector_inputs(size, /*requires_grad=*/true);

  for (auto _ : state) {
    auto residual = inputs.lhs - inputs.rhs;
    auto loss = residual.dot(residual);
    auto scalar_loss = loss.item<float>();
    auto gradients = torch::autograd::grad({loss}, {inputs.lhs, inputs.rhs});
    auto checksum = materialized_gradient_checksum(gradients[0], gradients[1]);
    benchmark::DoNotOptimize(scalar_loss);
    benchmark::DoNotOptimize(checksum);
  }
  report_work(state, size, 2 * size * sizeof(float));
}

void BM_BroadcastReluForward(benchmark::State& state) {
  const auto inputs =
      make_broadcast_inputs(state.range(0), state.range(1), state.range(2),
                            /*requires_grad=*/false);
  torch::NoGradGuard guard;

  for (auto _ : state) {
    auto loss = broadcast_relu_loss(inputs.lhs, inputs.rhs);
    benchmark::DoNotOptimize(loss);
  }
  const auto elements = volume(state);
  const auto input_elements = state.range(0) * state.range(2) + state.range(1);
  report_work(state, elements, input_elements * sizeof(float));
}

void BM_BroadcastReluClad(benchmark::State& state) {
  const auto inputs =
      make_broadcast_inputs(state.range(0), state.range(1), state.range(2),
                            /*requires_grad=*/false);
  auto lhs_gradient = torch::zeros_like(inputs.lhs);
  auto rhs_gradient = torch::zeros_like(inputs.rhs);
  auto gradient = clad::gradient(broadcast_relu_loss, "lhs, rhs");

  for (auto _ : state) {
    lhs_gradient.zero_();
    rhs_gradient.zero_();
    gradient.execute(inputs.lhs, inputs.rhs, &lhs_gradient, &rhs_gradient);
    auto checksum = materialized_gradient_checksum(lhs_gradient, rhs_gradient);
    benchmark::DoNotOptimize(checksum);
  }
  const auto elements = volume(state);
  const auto input_elements = state.range(0) * state.range(2) + state.range(1);
  report_work(state, elements, input_elements * sizeof(float));
}

void BM_BroadcastReluAutograd(benchmark::State& state) {
  const auto inputs =
      make_broadcast_inputs(state.range(0), state.range(1), state.range(2),
                            /*requires_grad=*/true);

  for (auto _ : state) {
    auto product = inputs.lhs * inputs.rhs;
    auto activated = product.relu();
    auto loss = activated.sum();
    auto scalar_loss = loss.item<float>();
    auto gradients = torch::autograd::grad({loss}, {inputs.lhs, inputs.rhs});
    auto checksum = materialized_gradient_checksum(gradients[0], gradients[1]);
    benchmark::DoNotOptimize(scalar_loss);
    benchmark::DoNotOptimize(checksum);
  }
  const auto elements = volume(state);
  const auto input_elements = state.range(0) * state.range(2) + state.range(1);
  report_work(state, elements, input_elements * sizeof(float));
}

void BM_SumDimsForward(benchmark::State& state) {
  const auto inputs =
      make_reduction_inputs(state.range(0), state.range(1), state.range(2),
                            /*requires_grad=*/false);
  const std::vector<std::int64_t> dims{0, -1};
  torch::NoGradGuard guard;

  for (auto _ : state) {
    auto loss = sum_dims_loss(inputs.input, inputs.weights, dims);
    benchmark::DoNotOptimize(loss);
  }
  const auto elements = volume(state);
  report_work(state, elements, (elements + state.range(1)) * sizeof(float));
}

void BM_SumDimsClad(benchmark::State& state) {
  const auto inputs =
      make_reduction_inputs(state.range(0), state.range(1), state.range(2),
                            /*requires_grad=*/false);
  const std::vector<std::int64_t> dims{0, -1};
  auto input_gradient = torch::zeros_like(inputs.input);
  auto weights_gradient = torch::zeros_like(inputs.weights);
  auto gradient = clad::gradient(sum_dims_loss, "input, weights");

  for (auto _ : state) {
    input_gradient.zero_();
    weights_gradient.zero_();
    gradient.execute(inputs.input, inputs.weights, dims, &input_gradient,
                     &weights_gradient);
    auto checksum =
        materialized_gradient_checksum(input_gradient, weights_gradient);
    benchmark::DoNotOptimize(checksum);
  }
  const auto elements = volume(state);
  report_work(state, elements, (elements + state.range(1)) * sizeof(float));
}

void BM_SumDimsAutograd(benchmark::State& state) {
  const auto inputs =
      make_reduction_inputs(state.range(0), state.range(1), state.range(2),
                            /*requires_grad=*/true);
  const std::vector<std::int64_t> dims{0, -1};

  for (auto _ : state) {
    auto reduced = inputs.input.sum(dims, /*keepdim=*/false);
    auto weighted = reduced * inputs.weights;
    auto loss = weighted.sum();
    auto scalar_loss = loss.item<float>();
    auto gradients =
        torch::autograd::grad({loss}, {inputs.input, inputs.weights});
    auto checksum = materialized_gradient_checksum(gradients[0], gradients[1]);
    benchmark::DoNotOptimize(scalar_loss);
    benchmark::DoNotOptimize(checksum);
  }
  const auto elements = volume(state);
  report_work(state, elements, (elements + state.range(1)) * sizeof(float));
}

void configure_vector(benchmark::internal::Benchmark* benchmark) {
  benchmark->RangeMultiplier(4)
      ->Range(256, 1 << 20)
      ->ArgName("elements")
      ->Unit(benchmark::kMicrosecond);
}

void configure_3d(benchmark::internal::Benchmark* benchmark) {
  benchmark->Args({1, 8, 32})
      ->Args({4, 16, 64})
      ->Args({8, 32, 128})
      ->Args({16, 64, 256})
      ->Args({16, 128, 512})
      ->ArgNames({"batch", "channels", "width"})
      ->Unit(benchmark::kMicrosecond);
}

BENCHMARK(BM_SquaredErrorForward)
    ->Name("BM_SquaredError/Forward")
    ->Apply(configure_vector);
BENCHMARK(BM_SquaredErrorClad)
    ->Name("BM_SquaredError/Clad")
    ->Apply(configure_vector);
BENCHMARK(BM_SquaredErrorAutograd)
    ->Name("BM_SquaredError/Autograd")
    ->Apply(configure_vector);

BENCHMARK(BM_ReluVJPComposite)
    ->Name("BM_ReluVJP/Composite")
    ->Apply(configure_vector);
BENCHMARK(BM_ReluVJPNative)->Name("BM_ReluVJP/Native")->Apply(configure_vector);

BENCHMARK(BM_BroadcastReluForward)
    ->Name("BM_BroadcastRelu/Forward")
    ->Apply(configure_3d);
BENCHMARK(BM_BroadcastReluClad)
    ->Name("BM_BroadcastRelu/Clad")
    ->Apply(configure_3d);
BENCHMARK(BM_BroadcastReluAutograd)
    ->Name("BM_BroadcastRelu/Autograd")
    ->Apply(configure_3d);

BENCHMARK(BM_SumDimsForward)->Name("BM_SumDims/Forward")->Apply(configure_3d);
BENCHMARK(BM_SumDimsClad)->Name("BM_SumDims/Clad")->Apply(configure_3d);
BENCHMARK(BM_SumDimsAutograd)->Name("BM_SumDims/Autograd")->Apply(configure_3d);

} // namespace

BENCHMARK_MAIN();
