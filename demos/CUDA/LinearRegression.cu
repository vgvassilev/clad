// A demo of clad's automatic differentiation capabilities on a CUDA Thrust
// program that computes the loss of a simple linear regression model. The model
// calculates a prediction `y_pred` as the inner product of features `x` and
// weights `w`. The loss is the squared error between `y_pred` and a true value
// `y_true`. The program then computes the gradient of this loss with respect to
// the weights `w`.

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustDerivatives.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

#include <iostream>
#include <numeric>
#include <vector>

double linear_regression_loss(const thrust::device_vector<double>& x,
                              const thrust::device_vector<double>& w,
                              double y_true) {
  double y_pred = thrust::inner_product(x.begin(), x.end(), w.begin(), 0.0);
  double error = y_pred - y_true;
  return error * error;
}

int main() {
  const int N = 10;

  // Initialize host vectors for features (x) and weights (w)
  thrust::host_vector<double> h_x(N);
  std::iota(h_x.begin(), h_x.end(), 1.0); // x = {1, 2, ..., 10}

  thrust::host_vector<double> h_w(N, 0.1); // w = {0.1, 0.1, ..., 0.1}

  const double y_true = 10.0;

  thrust::device_vector<double> x = h_x;
  thrust::device_vector<double> w = h_w;

  auto loss_grad = clad::gradient(linear_regression_loss);

  thrust::device_vector<double> x_grad(N);
  thrust::device_vector<double> w_grad(N);
  double y_true_grad = 0;

  loss_grad.execute(x, w, y_true, &x_grad, &w_grad, &y_true_grad);

  std::cout << "Running linear regression demo." << std::endl;

  thrust::host_vector<double> host_w_gradients = w_grad;
  std::cout << "Gradients of loss wrt weights (w): ";
  for (size_t i = 0; i < host_w_gradients.size(); ++i)
    std::cout << host_w_gradients[i] << " ";
  std::cout << std::endl;

  return 0;
}
