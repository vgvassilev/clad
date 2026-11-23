// RUN: %cladclang %s -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -o%t
// RUN: %t

#include "clad/Differentiator/Differentiator.h"
#include <cassert>
#include <type_traits>

double square(double x) noexcept { return x * x; }
double square(double x, double y) { return x * y; }
using Unary = double (*)(double);
template <Unary F> auto make_pullback() { return clad::pullback(F); }
double composed(double x, double y) { return square(x + y); }
void store(double x, double* out) { *out = square(x); }
const double& borrow(const double& x) { return x; }
double* alias(double* x) { return x; }

double custom(double x) { return x; }
namespace clad {
namespace custom_derivatives {
void custom_pullback(double, double seed, double* dx) { *dx += 7 * seed; }
} // namespace custom_derivatives
} // namespace clad

// CHECK: void composed_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y)
// CHECK: void store_pullback(double x, double *out, double *_d_x, double *_d_out)
int main() {
  auto pb = clad::pullback(composed);
  static_assert(std::is_same<decltype(pb.getFunctionPtr()),
                             void (*)(double, double, double, double*, double*)>::value,
                "the output seed precedes the input adjoints");
  auto same = clad::pullback(composed);
  assert(same.getFunctionPtr() == pb.getFunctionPtr() && pb.getFunctionPtr());
  double dx = 1, dy = 2;
  pb.execute(2., 3., -2., &dx, &dy);
  pb.execute(2., 3., 0., &dx, &dy);
  assert(dx == -19 && dy == -18);

  double out = 0, d_out = 3;
  dx = 0;
  clad::pullback(store).execute(4., &out, &dx, &d_out);
  assert(dx == 24 && d_out == 0);
  clad::pullback(alias).execute(&out, &d_out);
  assert(d_out == 0);
  clad::pullback(borrow).execute(out, 2., &dx);
  assert(dx == 26);
  clad::pullback(custom).execute(0., 3., &dx);
  assert(dx == 47);
  constexpr Unary unary = static_cast<Unary>(&square);
  clad::pullback(unary).execute(2., 1., &dx);
  assert(dx == 51);
  auto from_template = make_pullback<square>();
  assert(from_template.getFunctionPtr());
  from_template.execute(3., 2., &dx);
  assert(dx == 63);
}
