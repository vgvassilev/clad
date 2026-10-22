// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A name brought in by a block-scope using-declaration is found in the
// enclosing function, which cannot be mirrored under clad::custom_derivatives.
// Its custom derivative has to be looked up where the named function lives.
// Reduced from ImpactX.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

namespace impactx {
double sq(double x) { return x * x; }
double cube(double x) { return x * x * x; }
double g(double x) {
  using impactx::sq;
  return sq(x) + cube(x);
}
double direct(double x) { return sq(x); }
double quad(double x) { return x * x; }
} // namespace impactx

// A using-declaration at namespace scope keeps naming its own namespace.
namespace app {
using impactx::quad;
double h(double x) { return app::quad(x); }
} // namespace app

// The derivatives below are deliberately not those of the bodies, to tell
// which one clad used.
namespace clad {
namespace custom_derivatives {
namespace impactx {
clad::ValueAndPushforward<double, double> cube_pushforward(double x,
                                                           double d_x) {
  return {x * x * x, 1000 * d_x};
}
clad::ValueAndPushforward<double, double> sq_pushforward(double x, double d_x) {
  return {x * x, 7 * d_x};
}
} // namespace impactx
namespace app {
clad::ValueAndPushforward<double, double> quad_pushforward(double x,
                                                           double d_x) {
  return {x * x, 5 * d_x};
}
} // namespace app
} // namespace custom_derivatives
} // namespace clad

// CHECK: warning: declaration kind 'Using' is not supported
// CHECK: void g_grad(double x, double *_d_x) {
// CHECK: clad::custom_derivatives::impactx::sq_pushforward(x, 1.).pushforward
// CHECK: clad::custom_derivatives::impactx::cube_pushforward(x, 1.).pushforward

int main() {
  double dg = 0, dd = 0, dh = 0;
  auto g = clad::gradient(impactx::g);
  auto d = clad::gradient(impactx::direct);
  auto h = clad::gradient(app::h);
  g.execute(2, &dg);
  d.execute(2, &dd);
  h.execute(2, &dh);
  printf("%.2f %.2f %.2f\n", dg, dd, dh); // CHECK-EXEC: 1007.00 7.00 5.00
}
