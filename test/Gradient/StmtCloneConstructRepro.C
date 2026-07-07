// RUN: %cladclang %s -I%S/../../include -oStmtCloneConstructRepro.out 2>&1

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

struct Vec {
  double x, y, z;
  Vec(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
  Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
inline void operator_star_pullback(const Vec* a, double b, Vec d_out,
                                   Vec* d_a, double* d_b) {
  d_a->x += d_out.x * b;
  d_a->y += d_out.y * b;
  d_a->z += d_out.z * b;
  *d_b += a->x * d_out.x + a->y * d_out.y + a->z * d_out.z;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

static inline Vec normalize_vec(const Vec& v) {
  double inv = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return v * inv;
}

// CXXTemporaryObjectExpr with expression arguments; crashes without StmtClone deep clone.
double inline_vec_normal(double cx, double cy, double cz) {
  Vec hit(cx, cy, cz);
  Vec origin(0, 0, 0);
  Vec n = normalize_vec(
      Vec(hit.x - origin.x, hit.y - origin.y, hit.z - origin.z));
  return n.x + n.y + n.z;
}

int main() {
  auto grad = clad::gradient(inline_vec_normal, "cx");
  double gcx = 0.;
  grad.execute(1.0, 2.0, 3.0, &gcx);
  return 0;
}
