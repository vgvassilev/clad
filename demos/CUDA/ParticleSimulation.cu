// A demo of clad's automatic differentiation capabilities on a CUDA Thrust
// program that simulates the motion of particles. This program updates the
// positions of particles over a fixed number of time steps and returns the sum
// of their final x-positions. It then calculates the gradient of this sum with
// respect to the initial x-velocities of the particles.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustBuiltins.h"
#include "clad/Differentiator/ThrustDerivatives.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <iostream>

double run_simulation(thrust::device_vector<double>& x,
                      thrust::device_vector<double>& y,
                      const thrust::device_vector<double>& vx,
                      const thrust::device_vector<double>& vy,
                      const thrust::device_vector<double>& dts) {
  const int n_steps = 5;

  // Allocate buffers.
  thrust::device_vector<double> tmp(x.size());
  thrust::device_vector<double> x_buffer(x.size());
  thrust::device_vector<double> y_buffer(x.size());

  for (int i = 0; i < n_steps; ++i) {
    // Update x-position.
    thrust::transform(vx.begin(), vx.end(), dts.begin(), tmp.begin(),
                      thrust::multiplies<double>());
    thrust::transform(x.begin(), x.end(), tmp.begin(), x_buffer.begin(),
                      thrust::plus<double>());
    thrust::copy(x_buffer.begin(), x_buffer.end(), x.begin());

    // Update y-position.
    thrust::transform(vy.begin(), vy.end(), dts.begin(), tmp.begin(),
                      thrust::multiplies<double>());
    thrust::transform(y.begin(), y.end(), tmp.begin(), y_buffer.begin(),
                      thrust::plus<double>());
    thrust::copy(y_buffer.begin(), y_buffer.end(), y.begin());
  }

  // Return the sum of the final x-positions of all particles.
  return thrust::reduce(x.begin(), x.end(), 0.0);
}

int main() {
  const int N = 10;
  const double dt = 0.1;

  // 1. Set up all vectors in main.
  thrust::device_vector<double> x(N, 0.0);
  thrust::device_vector<double> y(N, 0.0);
  thrust::device_vector<double> vx(N, 2.0);
  thrust::device_vector<double> vy(N, 1.0);
  thrust::device_vector<double> dts(N, dt);

  // 2. Differentiate `run_simulation`.
  auto run_simulation_grad = clad::gradient(run_simulation);

  // 3. Prepare storage for gradients.
  thrust::device_vector<double> x_grad(N);
  thrust::device_vector<double> y_grad(N);
  thrust::device_vector<double> vx_grad(N);
  thrust::device_vector<double> vy_grad(N);
  thrust::device_vector<double> dts_grad(N);

  // 4. Execute the generated gradient function.
  run_simulation_grad.execute(x, y, vx, vy, dts, &x_grad, &y_grad, &vx_grad,
                              &vy_grad, &dts_grad);

  // 5. Print the results.
  std::cout << "Running particle simulation demo." << std::endl;

  thrust::host_vector<double> host_gradients = vx_grad;
  std::cout << "Gradients of final x-pos sum wrt initial vx: ";
  for (size_t i = 0; i < host_gradients.size(); ++i)
    std::cout << host_gradients[i] << " ";
  std::cout << std::endl;

  return 0;
}