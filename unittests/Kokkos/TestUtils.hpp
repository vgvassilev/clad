// Useful things

#ifndef KOKKOS_UNITTEST_UTILS
#define KOKKOS_UNITTEST_UTILS

template <typename T> // comparison with the finite difference approx. has been tested in the initial PR for Kokkos-aware Clad by Kim Liegeois
T finite_difference_tangent(std::function<T(T)> func, const T& x, const T& epsilon) {
  return (func(x+epsilon)-func(x-epsilon)) / (2 * epsilon);
}

#endif
