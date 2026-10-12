//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Trains a small neural network. The only derivative anywhere is the one
// clad writes.
//
//----------------------------------------------------------------------------//

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

// Exclusive or is true when exactly one input is true. No straight line
// separates the two answers, so a single layer cannot learn it. A hidden
// layer can. Finding the weights is what the gradient is for.
static const double kInput[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const double kWanted[4] = {-1, 1, 1, -1};

// Two inputs, two hidden units, one output. Nine weights in all.
double net(const double* w, double a, double b) {
  double h0 = std::tanh(w[0] * a + w[1] * b + w[2]);
  double h1 = std::tanh(w[3] * a + w[4] * b + w[5]);
  return std::tanh(w[6] * h0 + w[7] * h1 + w[8]);
}

// How wrong the network is over all four cases. This is the function clad
// differentiates, with respect to the nine weights.
// docs-begin-xor
double loss(const double w[9]) {
  double total = 0;
  for (int i = 0; i < 4; ++i) {
    double miss = net(w, kInput[i][0], kInput[i][1]) - kWanted[i];
    total += miss * miss;
  }
  return total;
}
// docs-end-xor

int main() {
  // One call to clad. Everything below is arithmetic on what it returns.
  // Nothing in the program knows the derivative of tanh.
  // docs-begin-xor-call
  auto grad = clad::gradient(loss);
  // docs-end-xor-call

  // Fixed starting weights, not random ones, so every run prints the same
  // thing.
  double w[9] = {.7, -.6, .1, -.8, .5, -.2, .9, .4, -.3};

  for (int step = 1; step <= 4000; ++step) {
    double dw[9] = {0.};
    grad.execute(w, dw);
    for (int i = 0; i < 9; ++i)
      w[i] -= .1 * dw[i];
    if (step % 1000 == 0)
      printf("step %4d: loss = %.6f\n", step, loss(w));
  }

  for (int i = 0; i < 4; ++i)
    printf("  %g XOR %g -> %+.3f (want %+g)\n", kInput[i][0], kInput[i][1],
           net(w, kInput[i][0], kInput[i][1]), kWanted[i]);

  return 0;
}
