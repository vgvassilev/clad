// Useful things

#ifndef KOKKOS_UNITTEST_UTILS
#define KOKKOS_UNITTEST_UTILS

template <typename F, typename T> // comparison with the finite difference
                                  // approx. has been tested in the initial PR
                                  // for Kokkos-aware Clad by Kim Liegeois
T finite_difference_tangent(F func, const T& x, const T& epsilon) {
  return (func(x + epsilon) - func(x - epsilon)) / (2 * epsilon);
}

double parallel_polynomial_true_derivative(
    double x) { // the true derivative of the polynomial tested in
                // ParallelFor.cpp and ParallelReduce.cpp
  double res = 0;
  double x_c = 1;
  for (unsigned i = 0; i < 6; ++i) {
    res += x_c;
    x_c *= x;
  }
  return res;
}

#endif