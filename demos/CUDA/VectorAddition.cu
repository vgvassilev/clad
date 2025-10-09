// A demo of clad's automatic differentiation capabilities on a simple CUDA
// Thrust program. This program computes the element-wise sum of two vectors `x`
// and `y`, stores the result in `z`, and returns the sum of `z`'s elements. It
// then calculates the gradient of this sum with respect to the initial values
// of `x`.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustBuiltins.h"
#include "clad/Differentiator/ThrustDerivatives.h"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <iostream>
#include <string>
#include <vector>

double vector_addition(const thrust::device_vector<double>& x,
                       const thrust::device_vector<double>& y) {
  thrust::device_vector<double> z(x.size());
  thrust::transform(x.begin(), x.end(), y.begin(), z.begin(),
                    thrust::plus<double>());
  return thrust::reduce(z.begin(), z.end(), 0.0);
}

int main() {
  const int N = 10;

  thrust::device_vector<double> x(N, 1.0);
  thrust::device_vector<double> y(N, 2.0);

  std::cout << "Running vector addition demo." << std::endl;

  auto vector_addition_grad = clad::gradient(vector_addition);

  thrust::device_vector<double> x_grad(N);
  thrust::device_vector<double> y_grad(N);

  vector_addition_grad.execute(x, y, &x_grad, &y_grad);

  thrust::host_vector<double> host_gradients = x_grad;
  std::cout << "Gradients of sum wrt initial x: ";
  for (size_t i = 0; i < host_gradients.size(); ++i)
    std::cout << host_gradients[i] << " ";
  std::cout << std::endl;

  return 0;
}
