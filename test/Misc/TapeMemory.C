// RUN: %cladclang %s -x c++ -lstdc++ -I%S/../../include -oTapeMemory.out 2>&1
// RUN: ./TapeMemory.out
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

class A {};

// Minimum viable code to test for tape pushes/pops
template <typename T> void func(T x, int n) {
  clad::tape<T> t = {};
  for (int i = 0; i < n; i++) {
    clad::push<T>(t, x);
  }
  for (int i = 0; i < n; i++) {
    clad::pop<T>(t);
  }
}

int main() {

  int block = 32, n = 5;
  int dummy = 0;
  for (int i = 0; i < n; i++, block *= 2) {
    // Scalar types
    func<bool>(0, block);
    func<char>(0, block);
    func<int>(0, block);
    func<float>(0, block);
    func<double>(0, block);
    func<long>(0, block);
    func<long double>(0, block);
    // Const/Volatile types
    func<const int>(static_cast<const int>(dummy), block);
    func<volatile float>(static_cast<volatile float>(dummy), block);
    func<const volatile double>(static_cast<const volatile double>(dummy),
                                block);
    // custom type
    func<A>(A(), block);
  }
}
