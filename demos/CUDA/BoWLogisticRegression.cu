/*
This demo implements logistic regression with CUDA Thrust and clad automatic
differentiation. It differentiates a single-example logistic loss, then runs SGD
on a small four-document batch by invoking an L2-regularized two-document loss
twice. The program prints the single-example loss and gradient, reports
training loss periodically, and finally outputs the batch accuracy.
*/

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/ThrustBuiltins.h"
#include "clad/Differentiator/ThrustDerivatives.h"

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cmath>
#include <iostream>

using Vec = thrust::device_vector<double>;
static inline double dot(const Vec& a, const Vec& b) {
  return thrust::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
static inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// Minimal single-document logistic loss.
// x: (V), w: (V), y in {0,1}
double logistic_loss_single(const Vec& x, const Vec& w, double b, double y) {
  double logit = dot(x, w) + b;
  double p = sigmoid(logit);
  const double eps = 1e-9;
  return -y * std::log(p + eps) - (1.0 - y) * std::log(1.0 - p + eps);
}

// L2-regularized 2-doc batch loss (0.5*(l0+l1) + 0.5*lambda*(||w||^2 + b^2))
double logistic_loss_batch2_prepared_l2(const Vec& x0, const Vec& x1,
                                        const Vec& w, double b, double y0,
                                        double y1, double lambda) {
  double l0 = logistic_loss_single(x0, w, b, y0);
  double l1 = logistic_loss_single(x1, w, b, y1);
  double w2 = dot(w, w);
  return 0.5 * (l0 + l1) + 0.5 * lambda * w2 + 0.5 * lambda * (b * b);
}

static inline void zero(Vec& v) { thrust::fill(v.begin(), v.end(), 0.0); }

static inline double predict_prob(const Vec& xi, const Vec& w, double b) {
  return sigmoid(dot(xi, w) + b);
}

static inline void sgd_step(Vec& w, const Vec& dw, double& b, double db,
                            double lr) {
  thrust::transform(
      w.begin(), w.end(), dw.begin(), w.begin(),
      [=] __device__(double wi, double gi) { return wi - lr * gi; });
  b -= lr * db;
}

int main() {
  std::cout << "Running minimal logistic regression demo." << std::endl;

  const int V = 8;
  // Dense feature vector and weights.
  double hx[] = {2, 0, 1, 0, 0, 0, 1, 0};
  Vec x(hx, hx + V);
  Vec w(V, 0.1);
  double b = 0.0;
  double y = 1.0;

  auto grad = clad::gradient(logistic_loss_single);
  Vec dx(V), dw(V);
  double db = 0.0;
  double dy = 0.0;

  grad.execute(x, w, b, y, &dx, &dw, &db, &dy);
  double loss = logistic_loss_single(x, w, b, y);

  thrust::host_vector<double> hdw = dw;
  std::cout << "Loss: " << loss << "\nGradient wrt w: ";
  for (int i = 0; i < V; ++i)
    std::cout << hdw[i] << " ";
  std::cout << std::endl;

  // 4-document batch demo.
  {
    std::cout << "\nRunning SGD on 4-doc batch..." << std::endl;
    double hX4[] = {
        2, 0, 1, 0, 0, 0, 1, 0, // doc 0 (y=1)
        0, 3, 0, 1, 0, 0, 0, 0, // doc 1 (y=0)
        0, 1, 2, 0, 0, 0, 0, 0, // doc 2 (y=1)
        0, 0, 0, 0, 2, 1, 0, 0  // doc 3 (y=0)
    };
    Vec X4(hX4, hX4 + 4 * V);
    Vec a0(X4.begin() + 0 * V, X4.begin() + 1 * V);
    Vec a1(X4.begin() + 1 * V, X4.begin() + 2 * V);
    Vec a2(X4.begin() + 2 * V, X4.begin() + 3 * V);
    Vec a3(X4.begin() + 3 * V, X4.begin() + 4 * V);
    double y_a0 = 1.0, y_a1 = 0.0, y_a2 = 1.0, y_a3 = 0.0;

    auto loss2 = clad::gradient(logistic_loss_batch2_prepared_l2);
    Vec dA0(V), dA1(V), dA2(V), dA3(V), dW4(V);
    double db4 = 0.0, dlam4 = 0.0;
    const int iters4 = 50;
    const double lr4 = 0.1;
    const double lambda4 = 1e-2;
    for (int t = 0; t < iters4; ++t) {
      zero(dW4);
      zero(dA0);
      zero(dA1);
      zero(dA2);
      zero(dA3);
      db4 = dlam4 = 0.0;

      double dy_dummy = 0.0;
      loss2.execute(a0, a1, w, b, y_a0, y_a1, lambda4, &dA0, &dA1, &dW4, &db4,
                    &dy_dummy, &dy_dummy, &dlam4);
      loss2.execute(a2, a3, w, b, y_a2, y_a3, lambda4, &dA2, &dA3, &dW4, &db4,
                    &dy_dummy, &dy_dummy, &dlam4);

      thrust::transform(dW4.begin(), dW4.end(), dW4.begin(),
                        [] __device__(double g) { return 0.5 * g; });
      db4 *= 0.5;

      sgd_step(w, dW4, b, db4, lr4);

      if ((t % 10) == 0 || t == iters4 - 1) {
        double Lpair1 =
            logistic_loss_batch2_prepared_l2(a0, a1, w, b, y_a0, y_a1, lambda4);
        double Lpair2 =
            logistic_loss_batch2_prepared_l2(a2, a3, w, b, y_a2, y_a3, lambda4);
        double L4 = 0.5 * (Lpair1 + Lpair2);
        std::cout << "iter4 " << t << ": loss4=" << L4 << std::endl;
      }
    }

    int ok = 0;
    ok += ((predict_prob(a0, w, b) >= 0.5) == (y_a0 == 1.0));
    ok += ((predict_prob(a1, w, b) >= 0.5) == (y_a1 == 1.0));
    ok += ((predict_prob(a2, w, b) >= 0.5) == (y_a2 == 1.0));
    ok += ((predict_prob(a3, w, b) >= 0.5) == (y_a3 == 1.0));
    std::cout << "Batch-4 accuracy: " << (double)ok / 4.0 << std::endl;
  }

  return 0;
}
