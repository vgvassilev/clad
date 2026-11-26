// Do the same thing what's done in basic-nn.cpp, but using libtorch instead of
// Clad.

/*
  To build and run
  1. Create a build directory: mkdir build
  2. Change to the build directory: cd build
  3. Run cmake: cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
  4. Run make: make
  5. Run the executable: ./torch-basic-nn
*/

#include <chrono>
#include <iostream>
#include <random>
#include <torch/torch.h>

#define M 100       // num of data points
#define N 100       // num of features
#define N_ITER 5000 // num of iterations

// C++ neural network model defined with value semantics
struct TwoLayerNet : torch::nn::Module {
  // constructor with submodules registered in the initializer list
  TwoLayerNet(int64_t D_in, int64_t D_out, int64_t H)
      : linear1(register_module("linear1", torch::nn::Linear(D_in, H))),
        linear2(register_module("linear2", torch::nn::Linear(H, D_out))) {}

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(linear1->forward(x));
    x = linear2->forward(x);
    return x;
  }
  torch::nn::Linear linear1;
  torch::nn::Linear linear2;
};

int main() {
  // Generate random data
  // std::default_random_engine generator;
  // std::normal_distribution<double> distribution(0.0, 1.0);
  torch::Tensor x = torch::randn({M, N});
  torch::Tensor y = torch::randn({M, 1});

  // Benchmark the time for 100 matrix multiplications
  torch::Tensor z;
  auto start_bench = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100; ++i)
    z = torch::matmul(x, x);
  auto end_bench = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_bench = end_bench - start_bench;
  std::cout << "Time taken for 100 matrix multiplications: "
            << elapsed_bench.count() << " seconds" << std::endl;

  // Initialize model weights
  TwoLayerNet model(N, 1, 3);

  // Create the optimizer and loss function
  torch::optim::SGD optimizer(model.parameters(), 0.01);
  torch::nn::MSELoss loss_fn;

  // Calculate the loss before optimization
  torch::Tensor output = model.forward(x);
  std::cout << "Initial loss before optimization: "
            << loss_fn(output, y).item<double>() << std::endl;

  // Optimize
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N_ITER; ++i) {
    optimizer.zero_grad();
    torch::Tensor output = model.forward(x);
    torch::Tensor loss = loss_fn(output, y);
    loss.backward();
    optimizer.step();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // Calculate the loss after optimization
  output = model.forward(x);
  std::cout << "Final loss after optimization: "
            << loss_fn(output, y).item<double>() << std::endl;
  std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;
  return 0;
}
