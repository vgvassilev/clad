// RUN: %cladclang -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

// Forward an lvalue.
template<typename _Tp>
constexpr _Tp&&
forward(typename std::remove_reference<_Tp>::type& __t) noexcept {
  return static_cast<_Tp&&>(__t);
}

//  Forward an rvalue.
template<typename _Tp>
constexpr _Tp&&
forward(typename std::remove_reference<_Tp>::type&& __t) noexcept {
  return static_cast<_Tp&&>(__t);
}

double fn(double u, double v) {
  return forward<double&>(u) + forward<double&&>(v);
}

int main() {
  double u = 3, v = 5;
  auto fn_grad = clad::gradient(fn);
  double du = 0, dv = 0;
  fn_grad.execute(u, v, &du, &dv);
}
