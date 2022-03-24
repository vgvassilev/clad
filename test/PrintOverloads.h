#ifndef TEST_PRINT_OVERLOADS_H
#define TEST_PRINT_OVERLOADS_H

#include <cstdio>
#include <utility>
#include <complex>

namespace std {
template <typename U, typename V> void print(const pair<U, V>& p) {
  test_utils::print(p.first);
  printf(", ");
  test_utils::print(p.second);
}

template<typename T>
void print(complex<T> c) {
  test_utils::print(c.real());
  printf(", ");
  test_utils::print(c.imag());
}
} // namespace std

#endif