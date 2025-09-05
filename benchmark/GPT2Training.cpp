#include "benchmark/benchmark.h"

#include <clad/Differentiator/CladtorchBuiltins.h>
#include <clad/Differentiator/Differentiator.h>
#include <clad/Differentiator/STLBuiltins.h>
#include "../demos/cladtorch/llm.hpp"
#include "../demos/cladtorch/llm_opt.hpp"

class GPT2Optimized : public benchmark::Fixture {
public:
  GPT2* model;
  GPT2* d_model;
  int* inputs;
  int* targets;

  void SetUp(const ::benchmark::State& state) override {
    GPT2Config config{};
    config.max_seq_len = 1024;
    config.vocab_size = 50257;
    config.padded_vocab_size = 50304;
    config.num_layers = 12;
    config.num_heads = 12;
    config.channels = 768;

    model = new GPT2(config);
    d_model = new GPT2(config);

    // Get batch size (B) and sequence length (T) from the benchmark state
    int B = state.range(0);
    int T = state.range(1);

    model->allocate(B, T);
    d_model->allocate(B, T);

    // Allocate and fill dummy input data
    inputs = new int[B * T];
    targets = new int[B * T];
    for (int i = 0; i < B * T; ++i) {
      inputs[i] = i % model->config.vocab_size;
      targets[i] = (i + 1) % model->config.vocab_size;
    }
  }

  void TearDown(const ::benchmark::State& state) override {
    // This runs once after each benchmark test
    delete model;
    delete d_model;
    delete[] inputs;
    delete[] targets;
  }
};

float gpt2forw_opt(GPT2* model, const int* inputs, const int* targets) {
  model->forward(inputs, targets);
  return model->mean_loss;
}

// The benchmark itself
BENCHMARK_DEFINE_F(GPT2Optimized,
                   FullTrainingIteration)(benchmark::State& state) {
  auto grad = clad::gradient(gpt2forw_opt, "0");
  int B = state.range(0);
  int T = state.range(1);

  for (auto _ : state) {
    state.PauseTiming();
    d_model->zero_all();

    state.ResumeTiming();
    // The single training iteration:
    // forward pass (calculated as part of gradient), backward pass, and update
    grad.execute(model, inputs, targets, d_model);
    model->update(d_model, 1e-3f);
  }
  state.SetLabel("B=" + std::to_string(B) + " T=" + std::to_string(T));
}

BENCHMARK_REGISTER_F(GPT2Optimized, FullTrainingIteration)
    ->Args({1, 16}) // B=1, T=16
    ->Args({1, 32}) // B=1, T=32
    ->Args({2, 16}) // B=2, T=16
    ->Args({1, 64}) // B=1, T=64
    ->Args({2, 32})
    ->Args({4, 32})
    ->Args({4, 64}) // B=4, T=64
    ->Unit(benchmark::kMillisecond);

class GPT2Cladtorch : public benchmark::Fixture {
public:
  gpt2::GPT2* model;
  gpt2::GPT2* d_model;
  int* inputs;
  int* targets;

  void SetUp(const ::benchmark::State& state) override {
    const gpt2::Config config = {
        .max_seq_len = 1024,
        .vocab_size = 50257,
        .padded_vocab_size = 50304,
        .num_layers = 12,
        .num_heads = 12,
        .channels = 768,
    };
    model = new gpt2::GPT2(config);
    d_model = new gpt2::GPT2(config);

    // Get batch size (B) and sequence length (T) from the benchmark state
    int B = state.range(0);
    int T = state.range(1);

    // Allocate and fill dummy input data
    inputs = new int[B * T];
    targets = new int[B * T];
    for (int i = 0; i < B * T; ++i) {
      inputs[i] = i % model->config.vocab_size;
      targets[i] = (i + 1) % model->config.vocab_size;
    }
  }

  void TearDown(const ::benchmark::State& state) override {
    // This runs once after each benchmark test
    delete model;
    delete d_model;
    delete[] inputs;
    delete[] targets;
  }
};

float gpt2_loss(const gpt2::GPT2& model, const gpt2::ITensor& input,
                const gpt2::ITensor& targets) {
  auto probs = model.forward(input);
  auto loss = cross_entropy_loss(probs, targets);
  return loss.scalar();
}

// The benchmark itself
BENCHMARK_DEFINE_F(GPT2Cladtorch,
                   FullTrainingIteration)(benchmark::State& state) {
  auto grad = clad::gradient(gpt2_loss, "0");
  int B = state.range(0);
  int T = state.range(1);
  const gpt2::ITensor inp({B, T}, inputs);
  const gpt2::ITensor tar({B, T}, targets);
  for (auto _ : state) {
    state.PauseTiming();
    d_model->for_each_parameter([&](gpt2::FTensor* t) { t->fill(0); });
    state.ResumeTiming();
    // The single training iteration: forward pass, backward pass, and update
    grad.execute(*model, inp, tar, d_model);
    std::vector<gpt2::FTensor*> params = model->get_parameter_tensors();
    std::vector<gpt2::FTensor*> grads = d_model->get_parameter_tensors();
    for (size_t i = 0; i < params.size(); ++i) {
      // Update parameters with a learning rate of 1e-4
      *params[i] += (*grads[i]) * -1e-3f;
    }
  }

  // You can set custom counters to report B and T
  state.SetLabel("B=" + std::to_string(B) + " T=" + std::to_string(T));
}

// Register the benchmark with different arguments
// This will run the benchmark for various combinations of batch size (B) and
// sequence length (T)
BENCHMARK_REGISTER_F(GPT2Cladtorch, FullTrainingIteration)
    ->Args({1, 16}) // B=1, T=16
    ->Args({1, 32}) // B=1, T=32
    ->Args({2, 16}) // B=2, T=16
    ->Args({1, 64}) // B=1, T=64
    ->Args({2, 32})
    ->Unit(benchmark::kMillisecond);

// Define our main.
BENCHMARK_MAIN();
