#ifndef SMALLPT_DIFF_RADIANCE_H
#define SMALLPT_DIFF_RADIANCE_H

#include "SmallPTDiffCommon.h"

// smallpt-style path tracer. light_y and mirror_x move the emissive and mirror
// spheres; omit them (defaults) to use the fixed scene in kDiffScene.
inline Vec diff_radiance(Ray r, int depth, unsigned short Xi[3],
                         double light_y = kDefaultLightY,
                         double mirror_x = kDefaultMirrorX) {
  Vec cl(0, 0, 0); // accumulated radiance
  Vec cf(1, 1, 1); // path throughput

  for (;;) {
    double t = 0;
    int id = 0;
    if (!diff_intersect(r.o.x, r.o.y, r.o.z, r.d.x, r.d.y, r.d.z, t, id,
                        light_y, mirror_x))
      return cl; // miss

    const DiffSphere& obj = kDiffScene[id];
    Vec x = r.o + r.d * t;

    // Outward-facing normal (use parametric center for light / mirror spheres).
    double center_x = obj.pos.x;
    if (id == kMirrorSphereId)
      center_x = mirror_x;
    double center_y = obj.pos.y;
    if (id == kLightSphereId)
      center_y = light_y;
    double norm_dx = x.x - center_x;
    double norm_dy = x.y - center_y;
    double norm_dz = x.z - obj.pos.z;
    Vec hit_dx(norm_dx, norm_dy, norm_dz);
    Vec n = normalize_vec(hit_dx);
    Vec nl = n;
    if (n * r.d >= 0) {
      Vec neg_n = n * -1.0;
      nl = neg_n;
    }

    Vec f = obj.color;
    double p = f.x; // max reflectance, used for Russian roulette
    if (f.y > p)
      p = f.y;
    if (f.z > p)
      p = f.z;

    Vec emission_term = cf.mult(obj.emission);
    cl = cl + emission_term;

    depth = depth + 1;
    if (depth > 5 || !p) {
      // Russian roulette: survive with prob p, else terminate path.
      if (sample_erand48(Xi) < p) {
        Vec scaled_f = f * (1.0 / p);
        f = scaled_f;
      } else {
        return cl;
      }
    }

    cf = cf.mult(f);

    if (obj.refl == DIFF) {
      // Cosine-weighted hemisphere sample.
      double r1 = 2 * kPi * sample_erand48(Xi);
      double r2 = sample_erand48(Xi);
      double r2s = sqrt(r2);
      Vec w = nl;
      Vec basis(0, 1, 0);
      if (fabs(w.x) > .1)
        basis = Vec(1, 0, 0);
      Vec u = normalize_vec(basis % w);
      Vec v = w % u;
      Vec u_cos = u * cos(r1);
      Vec u_term = u_cos * r2s;
      Vec v_sin = v * sin(r1);
      Vec v_term = v_sin * r2s;
      Vec uv_sum = u_term + v_term;
      Vec w_sqrt = w * sqrt(1 - r2);
      Vec bounce_dir = normalize_vec(uv_sum + w_sqrt);
      r = Ray(x, bounce_dir);
      continue;
    }

    if (obj.refl == SPEC) {
      r = Ray(x, r.d - n * (2.0 * (n * r.d)));
      continue;
    }

    // REFR: reflect or refract via Fresnel (Schlick).
    double ndotr = n * r.d;
    Vec reflect_term = n * (2.0 * ndotr);
    Vec spec_dir = r.d - reflect_term;
    Ray reflRay(x, spec_dir);

    bool into = n * nl > 0;
    double nc = 1.0, nt = 1.5;
    double nnt = into ? nc / nt : nt / nc;
    double ddn = r.d * nl;
    double cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

    if (cos2t < 0) {
      r = reflRay; // total internal reflection
      continue;
    }

    double scale = (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t));
    Vec nnt_vec = r.d * nnt;
    Vec ddn_vec = n * scale;
    Vec tdir_unnorm = nnt_vec - ddn_vec;
    Vec tdir = normalize_vec(tdir_unnorm);

    double a = nt - nc, b = nt + nc;
    double R0 = a * a / (b * b);
    double c = 1 - (into ? -ddn : tdir * n);
    double Re = R0 + (1 - R0) * c * c * c * c * c;
    double Tr = 1 - Re;
    double P = .25 + .5 * Re;
    double RP = Re / P;
    double TP = Tr / (1 - P);

    if (sample_erand48(Xi) < P) {
      cf = cf * RP;
      r = reflRay;
    } else {
      cf = cf * TP;
      r = Ray(x, tdir);
    }
  }
}

#endif // SMALLPT_DIFF_RADIANCE_H
