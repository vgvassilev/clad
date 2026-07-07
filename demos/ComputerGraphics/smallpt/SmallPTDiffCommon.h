#ifndef SMALLPT_DIFF_COMMON_H
#define SMALLPT_DIFF_COMMON_H

#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdlib>

#define non_differentiable __attribute__((annotate("non_differentiable")))

struct Vec {
  double x, y, z;
  Vec(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
  Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
  Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
  Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
  double operator*(const Vec& b) const { return x * b.x + y * b.y + z * b.z; }
  Vec operator%(const Vec& b) const {
    return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
  }
  Vec mult(const Vec& b) const { return Vec(x * b.x, y * b.y, z * b.z); }
  Vec norm() const;
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
inline void operator_minus_pullback(const Vec* a, const Vec& b, Vec d_out,
                                    Vec* d_a, Vec* d_b) {
  d_a->x += d_out.x;
  d_a->y += d_out.y;
  d_a->z += d_out.z;
  d_b->x -= d_out.x;
  d_b->y -= d_out.y;
  d_b->z -= d_out.z;
}
inline void operator_star_pullback(const Vec* a, double b, Vec d_out, Vec* d_a,
                                   double* d_b) {
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
inline void operator_percent_pullback(const Vec* a, const Vec& b, Vec d_out,
                                      Vec* d_a, Vec* d_b) {
  d_a->x += d_out.y * b.z - d_out.z * b.y;
  d_a->y += d_out.x * b.z - d_out.z * b.x;
  d_a->z += d_out.y * b.x - d_out.x * b.y;
  d_b->x += d_out.y * a->z - d_out.z * a->y;
  d_b->y += d_out.z * a->x - d_out.x * a->z;
  d_b->z += d_out.x * a->y - d_out.y * a->x;
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

non_differentiable inline double sample_erand48(unsigned short* Xi) {
  return erand48(Xi);
}

namespace clad {
namespace custom_derivatives {
inline void sample_erand48_pullback(unsigned short* /*Xi*/, double /*d_out*/,
                                    unsigned short* /*d_Xi*/) {}
} // namespace custom_derivatives
} // namespace clad

struct Ray {
  Vec o, d;
  Ray() : o(), d() {}
  Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };

static constexpr double kInf = 1e6;
static constexpr double kEps = 1e-6;
static constexpr double kPi = 3.14159265358979323846;

static inline Vec normalize_vec(const Vec& v) {
  double inv = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  return v * inv;
}

inline Vec Vec::norm() const { return normalize_vec(*this); }

static inline double vec_avg(const Vec& v) { return (v.x + v.y + v.z) / 3.0; }

struct DiffSphere {
  double rad;
  Vec pos, emission, color;
  Refl_t refl;
};

static constexpr int kNumSpheres = 9;
static constexpr int kLightSphereId = 8;
static constexpr int kMirrorSphereId = 6;
static constexpr double kDefaultLightY = 681.6 - .27;
static constexpr double kDefaultMirrorX = 27.0;
// Floor patch where mirror-reflected path-traced radiance is non-zero at
// defaults.
static constexpr int kMirrorRadiancePatchX0 = 200;
static constexpr int kMirrorRadiancePatchY0 = 320;
static constexpr int kMirrorRadiancePatchW = 4;
static constexpr int kMirrorRadiancePatchH = 4;
static constexpr double kDefaultCamDirX = 0.0;
static constexpr double kDefaultCamDirY = -0.042612;
static constexpr double kDefaultCamDirZ = -1.0;
static const DiffSphere kDiffScene[kNumSpheres] = {
    {1e5, Vec(1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF},
    {1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF},
    {1e5, Vec(50, 40.8, 1e5), Vec(), Vec(.75, .75, .75), DIFF},
    {1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(), DIFF},
    {1e5, Vec(50, 1e5, 81.6), Vec(), Vec(.75, .75, .75), DIFF},
    {1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF},
    {16.5, Vec(27, 16.5, 47), Vec(), Vec(.999, .999, .999), SPEC},
    {16.5, Vec(73, 16.5, 78), Vec(), Vec(.999, .999, .999), REFR},
    {600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec(), DIFF}};

static inline double diff_sphere_intersect(double srad, double spx, double spy,
                                           double spz, double rox, double roy,
                                           double roz, double rdx, double rdy,
                                           double rdz) {
  double opx = spx - rox, opy = spy - roy, opz = spz - roz;
  double b = opx * rdx + opy * rdy + opz * rdz;
  double det = b * b - (opx * opx + opy * opy + opz * opz) + srad * srad;
  if (det < 0)
    return 0;
  det = sqrt(det);
  double t1 = b - det, t2 = b + det;
  if (t1 > kEps)
    return t1;
  if (t2 > kEps)
    return t2;
  return 0;
}

static inline bool diff_intersect(double rox, double roy, double roz,
                                  double rdx, double rdy, double rdz, double& t,
                                  int& id, double light_y = kDefaultLightY,
                                  double mirror_x = kDefaultMirrorX) {
  t = kInf;
  id = 0;
  for (int i = kNumSpheres; i--;) {
    double spx = kDiffScene[i].pos.x;
    if (i == kMirrorSphereId)
      spx = mirror_x;
    double spy = kDiffScene[i].pos.y;
    if (i == kLightSphereId)
      spy = light_y;
    double d =
        diff_sphere_intersect(kDiffScene[i].rad, spx, spy, kDiffScene[i].pos.z,
                              rox, roy, roz, rdx, rdy, rdz);
    if (d > 0 && d < t) {
      t = d;
      id = i;
    }
  }
  return t < kInf;
}

#endif // SMALLPT_DIFF_COMMON_H
