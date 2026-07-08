// RUN: %cladclang %s -I%S/../../include -oReverseModeLoopPullbackRepro.out 2>&1
// RUN: ./ReverseModeLoopPullbackRepro.out
// XFAIL: valgrind

#include "clad/Differentiator/Differentiator.h"

#include <cmath>

struct Vec {
  double x, y, z;
  Vec(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
  Vec operator+(const Vec& b) const {
    return Vec(x + b.x, y + b.y, z + b.z);
  }
  Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
  double operator*(const Vec& b) const {
    return x * b.x + y * b.y + z * b.z;
  }
  Vec mult(const Vec& b) const {
    return Vec(x * b.x, y * b.y, z * b.z);
  }
};

namespace clad {
namespace custom_derivatives {
namespace class_functions {
inline void operator_plus_pullback(const Vec* a, const Vec& b, Vec d_out,
                                    Vec* d_a, Vec* d_b) {
  d_a->x += d_out.x;
  d_a->y += d_out.y;
  d_a->z += d_out.z;
  d_b->x += d_out.x;
  d_b->y += d_out.y;
  d_b->z += d_out.z;
}
inline void operator_star_pullback(const Vec* a, double b, Vec d_out,
                                   Vec* d_a, double* d_b) {
  d_a->x += d_out.x * b;
  d_a->y += d_out.y * b;
  d_a->z += d_out.z * b;
  *d_b += a->x * d_out.x + a->y * d_out.y + a->z * d_out.z;
}
inline void operator_star_pullback(const Vec* a, const Vec& b, double d_out,
                                   Vec* d_a, Vec* d_b) {
  d_a->x += d_out * b.x;
  d_a->y += d_out * b.y;
  d_a->z += d_out * b.z;
  d_b->x += d_out * a->x;
  d_b->y += d_out * a->y;
  d_b->z += d_out * a->z;
}
inline void mult_pullback(const Vec* a, const Vec& b, Vec d_out, Vec* d_a,
                          Vec* d_b) {
  d_a->x += d_out.x * b.x;
  d_a->y += d_out.y * b.y;
  d_a->z += d_out.z * b.z;
  d_b->x += d_out.x * a->x;
  d_b->y += d_out.y * a->y;
  d_b->z += d_out.z * a->z;
}
} // namespace class_functions
} // namespace custom_derivatives
} // namespace clad

static inline Vec normalize_vec(const Vec& v) {
  double inv = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return v * inv;
}

double loop_vec_radiance(double cx, double cy, double cz) {
  Vec cl(0, 0, 0);
  Vec cf(1, 1, 1);
  Vec origin(cx, cy, cz);
  Vec dir(0, -1, 0);
  int depth = 0;
  for (;;) {
    if (depth > 2)
      return cl.x + cl.y + cl.z;
    Vec hit = origin + dir * 1.0;
    Vec n = normalize_vec(hit);
    Vec nl = n;
    if (n * dir >= 0)
      nl = n * -1.0;
    Vec f(0.5, 0.5, 0.5);
    cl = cl + cf.mult(Vec(0.1, 0.1, 0.1));
    depth = depth + 1;
    cf = cf.mult(f);
    Vec basis(0, 1, 0);
    if (fabs(nl.x) > .1)
      basis = Vec(1, 0, 0);
    Vec u = normalize_vec(Vec(basis.y * nl.z - basis.z * nl.y,
                              basis.z * nl.x - basis.x * nl.z,
                              basis.x * nl.y - basis.y * nl.x));
    origin = hit;
    dir = normalize_vec(u + nl);
  }
}

int main() {
  auto grad = clad::gradient(loop_vec_radiance, "cx");
  double gcx = 0.;
  grad.execute(1.0, 2.0, 3.0, &gcx);
  return 0;
}
