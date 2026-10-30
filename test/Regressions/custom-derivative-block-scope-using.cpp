// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A name brought in by a block-scope using-declaration is found in the
// enclosing function, which cannot be mirrored under clad::custom_derivatives.
// Its custom derivative has to be looked up where the named function lives.
// The derivative re-declares the using-declaration, so that read back as
// source it names sq the way the primal did. Reduced from ImpactX.

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

// Reverse mode moves declarations to function scope; a using-directive in a
// loop moves there with them.
double loop(double x) {
  double s = 0;
  for (int i = 0; i < 2; ++i) {
    using namespace impactx;
    s += cube(x);
  }
  return s;
}

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

// CHECK: void g_grad(double x, double *_d_x) {
// CHECK-NEXT: using impactx::sq;
// CHECK: clad::custom_derivatives::impactx::sq_pushforward(x, 1.).pushforward
// CHECK: clad::custom_derivatives::impactx::cube_pushforward(x, 1.).pushforward

// CHECK: void loop_grad(double x, double *_d_x) {
// CHECK-NEXT: int _d_i = 0;
// CHECK-NEXT: int i = 0;
// CHECK-NEXT: using namespace impactx;

// CHECK: double g_darg0(double x) {
// CHECK-NEXT: double _d_x = 1;
// CHECK-NEXT: using impactx::sq;

int main() {
  double dg = 0, dd = 0, dh = 0;
  auto g = clad::gradient(impactx::g);
  auto d = clad::gradient(impactx::direct);
  auto h = clad::gradient(app::h);
  g.execute(2, &dg);
  d.execute(2, &dd);
  h.execute(2, &dh);
  printf("%.2f %.2f %.2f\n", dg, dd, dh); // CHECK-EXEC: 1007.00 7.00 5.00
  double dl = 0;
  auto l = clad::gradient(loop);
  l.execute(2, &dl);
  auto fg = clad::differentiate(impactx::g, "x");
  printf("%.2f %.2f\n", dl, fg.execute(2)); // CHECK-EXEC: 2000.00 1007.00
}
