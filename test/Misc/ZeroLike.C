// RUN: %cladclang %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

#include <array>
#include <cstdio>
#include <deque>
#include <forward_list>
#include <list>
#include <string>
#include <valarray>
#include <vector>

template <class Range> bool is_zero(const Range& range) {
  for (const auto& value : range)
    if (value != 0)
      return false;
  return true;
}

int main() {
  std::array<double, 2> array{{1., 2.}};
  std::deque<double> deque{1., 2.};
  std::forward_list<double> forwardList{1., 2.};
  std::list<double> list{1., 2.};
  std::vector<double> vector{1., 2.};
  std::valarray<double> valarray{1., 2.};
  std::string string{'a', 'b'};

  std::printf("%d %d %d %d %d %d %d\n",
              is_zero(clad::zero_like(array)),
              is_zero(clad::zero_like(deque)),
              is_zero(clad::zero_like(forwardList)),
              is_zero(clad::zero_like(list)),
              is_zero(clad::zero_like(vector)),
              is_zero(clad::zero_like(valarray)),
              is_zero(clad::zero_like(string)));
}

// CHECK-EXEC: 1 1 1 1 1 1 1
