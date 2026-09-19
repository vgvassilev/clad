// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s
//
// The regions between the docs- markers are included verbatim by
// docs/userDocs/source/user/CustomDerivatives.rst. Keep them readable: they
// are documentation that happens to be executed, not a test that happens to
// be quoted. Everything outside the markers is the harness that keeps it
// honest -- in particular the dumped derivatives, which are what show clad
// called the custom derivatives instead of differentiating the constructor.

#include "clad/Differentiator/Differentiator.h"
#include <iostream>

// docs-begin-constructor-pushforward
class Coordinates {
  public:
  Coordinates(double px, double py, double pz) :
    x(px), y(py), z(pz) {}

  public:
  double x, y, z;
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
// custom constructor pushforward function
clad::ValueAndPushforward<::Coordinates, ::Coordinates>
constructor_pushforward(clad::ConstructorPushforwardTag<::Coordinates>, double x, double y,
                        double z, double d_x, double d_y, double d_z) {
  return {::Coordinates(x, y, z), ::Coordinates(d_x, d_y, d_z) };
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad
// docs-end-constructor-pushforward

// docs-begin-constructor-pullback
namespace clad {
namespace custom_derivatives {
namespace class_functions {
void constructor_pullback(double x, double y, double z, ::Coordinates *d_coordinates,
    double *d_x, double *d_y, double *d_z) {
  *d_x += d_coordinates->x;
  d_coordinates->x = 0;
  *d_y += d_coordinates->y;
  d_coordinates->y = 0;
  *d_z += d_coordinates->z;
  d_coordinates->z = 0;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad
// docs-end-constructor-pullback

// A function whose only use of Coordinates is to construct one, so its
// derivative has to come through the constructor's custom derivatives.
double len2(double a, double b, double c) {
  Coordinates p(a, 2 * b, 3 * c);
  return p.x * p.x + p.y * p.y + p.z * p.z;
}

int main() {
  // len2(a, b, c) is a * a + 4 * b * b + 9 * c * c.
  auto len2_da = clad::differentiate(len2, "a");
  std::cout << len2_da.execute(1, 2, 3) << "\n"; // prints: 2
  len2_da.dump();
  // prints: {{.*}}constructor_pushforward(clad::Tag<Coordinates>(), a, 2 * b, 3 * c, _d_a, 2 * _d_b, 3 * _d_c);

  auto len2_grad = clad::gradient(len2);
  double da = 0, db = 0, dc = 0;
  len2_grad.execute(1, 2, 3, &da, &db, &dc);
  std::cout << da << " " << db << " " << dc << "\n"; // prints: 2 16 54
  len2_grad.dump();
  // prints: clad::custom_derivatives::class_functions::constructor_pullback(a, 2 * b, 3 * c, &_d_p, &_r0, &_r1, &_r2);
}
