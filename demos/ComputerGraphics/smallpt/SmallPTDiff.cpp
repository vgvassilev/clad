// Differentiable SmallPT path tracer (full radiance kernel via clad).

#include "clad/Differentiator/Differentiator.h"

#include "SmallPTDiffRadiance.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Refl_t = ::Refl_t;

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

inline Vec radiance(const Ray& r, int depth, unsigned short* Xi) {
  return diff_radiance(r, depth, Xi);
}

int main(int argc, char* argv[]) {
  int w = 512, h = 384, samps = argc >= 2 ? atoi(argv[1]) / 4 : 1;

  int fr = 0;
  Ray cam(Vec(50, 52, 295.6 - fr * 10), Vec(0, -0.042612, -1).norm());
  Vec cx = Vec(w * .5135 / h);
  Vec cy = (cx % cam.d).norm() * .5135;
  Vec r;
  Vec* frame = new Vec[w * h];

#pragma omp parallel for schedule(dynamic, 1) private(r)
  for (unsigned short y = 0; y < h; y++) {
    for (unsigned short x = 0, Xi[3] = {0, 0, (unsigned short)(y * y * y)};
         x < w; x++) {
      for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++) {
        for (int sx = 0; sx < 2; sx++, r = Vec()) {
          for (int s = 0; s < samps; s++) {
            double r1 = 2 * erand48(Xi);
            double dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
            double r2 = 2 * erand48(Xi);
            double dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
            Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
                    cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
            Vec d_norm = d.norm();
            Vec origin = cam.o + d_norm * 140;
            r = r + radiance(Ray(origin, d_norm), 0, Xi) * (1. / samps);
          }
          Vec clamped = Vec(clamp(r.x), clamp(r.y), clamp(r.z));
          Vec scaled = clamped * .25;
          frame[i] = frame[i] + scaled;
        }
      }
    }
  }

  char filename[100];
  snprintf(filename, 100, "image-%d.ppm", fr);
  fprintf(stderr, "\rSave %s\n", filename);
  FILE* f = fopen(filename, "wb");
  fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
  for (int i = 0; i < w * h; i++)
    fprintf(f, "%c%c%c", toInt(frame[i].x), toInt(frame[i].y),
            toInt(frame[i].z));
  fclose(f);
  delete[] frame;
  return 0;
}
