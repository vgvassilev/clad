#include "../demos/cladtorch/llm_opt.hpp"

#include "benchmark/benchmark.h"


float gpt2forw(GPT2* model, const int* inputs, const int* targets) {
  model->forward(inputs, targets);
  return model->mean_loss;
}

// Define a benchmark fixture for the GPT-2 model
class GPT2TrainingBenchmark : public benchmark::Fixture {
public:
    GPT2* model;
    GPT2* d_model;
    int* inputs;
    int* targets;

    void SetUp(const ::benchmark::State& state) override {
        // This runs once before each benchmark test
        // Use a dummy checkpoint file. Assumes it's in the same directory.
        const char* checkpoint_path = "/Users/rohan/Developer/projects/gsoc25/workspace/clad/demos/cladtorch/gpt2_124M.bin";
        
        // Initialize models
        model = new GPT2(checkpoint_path);
        d_model = new GPT2(checkpoint_path);

        // Get batch size (B) and sequence length (T) from the benchmark state
        int B = state.range(0); // This will be the batch_size
        int T = state.range(1); // This will be the sequence_length

        model->allocate(B, T);
        d_model->allocate(B, T);

        // Allocate and fill dummy input data
        // For a real benchmark, you could load real data or generate more representative data
        inputs = new int[B * T];
        targets = new int[B * T];
        for (int i = 0; i < B * T; ++i) {
            inputs[i] = i % model->config.vocab_size;
            targets[i] = (i + 1) % model->config.vocab_size;
        }

        // Initialize the gradient function
        // Note: The clad::gradient call needs to be done once, likely at compile time
        // or a global scope. For a benchmark, we assume it's pre-compiled.
        // For demonstration, we'll assume a global variable or function pointer.
    }

    void TearDown(const ::benchmark::State& state) override {
        // This runs once after each benchmark test
        delete model;
        delete d_model;
        delete[] inputs;
        delete[] targets;
    }
};

// Assuming the gradient function is accessible globally as in your code


// The benchmark itself
BENCHMARK_DEFINE_F(GPT2TrainingBenchmark, FullTrainingIteration)(benchmark::State& state) {
    auto grad = clad::gradient(gpt2forw, "0");
    int B = state.range(0);
    int T = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();
        d_model->zero_all();

        state.ResumeTiming();
        // The single training iteration: forward pass, backward pass, and update
        model->forward(inputs, targets);
        grad.execute(model, inputs, targets, d_model);
        model->update(d_model, 1e-3f);
    }

    // You can set custom counters to report B and T
    state.SetLabel("B=" + std::to_string(B) + " T=" + std::to_string(T));
}

// Register the benchmark with different arguments
// This will run the benchmark for various combinations of batch size (B) and sequence length (T)
BENCHMARK_REGISTER_F(GPT2TrainingBenchmark, FullTrainingIteration)
    ->Args({1, 64})   // B=1, T=64
    ->Args({2, 32})
    ->Args({4, 32})
    ->Args({4, 64})   // B=4, T=64
    ->Unit(benchmark::kMillisecond)
    // ->Args({8, 128})  // B=8, T=128
    // ->Args({16, 256}) // B=16, T=256
    // ->Args({32, 512}) // B=32, T=512
    // ->Args({64, 1024}); // B=64, T=1024
;

// Define our main.
BENCHMARK_MAIN();