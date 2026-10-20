// A tape in C++23. Its disk offload hands work to std::async, which keeps what
// it runs in a std::tuple; libstdc++ 14's C++23 tuple does not compile with a
// lambda in it under clang 17. Pushing and popping instantiate that work; it
// only runs once a tape outgrows the 1 GiB it keeps in RAM, too much for a test.
// RUN: %cladclang -std=c++2b %s -I%S/../../include -oTapeCxx23.out
// RUN: ./TapeCxx23.out | %filecheck_exec %s
// UNSUPPORTED: clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

int main() {
  clad::tape_impl<double, 64, 1024, /*is_multithread=*/false,
                  /*DiskOffload=*/true>
      t;
  for (int i = 0; i < 10000; ++i)
    t.emplace_back(i);
  double sum = 0;
  while (t.size()) {
    sum += t.back();
    t.pop_back();
  }
  printf("%.0f\n", sum); // CHECK-EXEC: 49995000
}
