#ifndef SMALLPT_DIFF_OBJECTIVES_H
#define SMALLPT_DIFF_OBJECTIVES_H

#include "SmallPTDiffRadiance.h"

inline double diff_subpixel_jitter(double r) {
  if (r < 1)
    return sqrt(r) - 1;
  return 1 - sqrt(2 - r);
}

inline double diff_objective_pixel(int pixel_x, int pixel_y, int samps,
                                   double cam_x, double cam_y, double cam_z,
                                   double light_y, double mirror_x) {
  const int w = 512;
  const int h = 384;

  unsigned short Xi[3] = {0, 0, (unsigned short)(pixel_y * pixel_y * pixel_y)};

  Vec cam_dir_raw(kDefaultCamDirX, kDefaultCamDirY, kDefaultCamDirZ);
  Vec cam_dir = normalize_vec(cam_dir_raw);
  Vec cxv(w * .5135 / h);
  Vec cx_cross = cxv % cam_dir;
  Vec cyv = normalize_vec(cx_cross);
  Vec cy_scaled = cyv * .5135;
  Vec cam_pos(cam_x, cam_y, cam_z);

  double inv_samps = 1.0 / samps;
  double sum = 0;
  for (int sy = 0; sy < 2; ++sy) {
    for (int sx = 0; sx < 2; ++sx) {
      double subpix_sum = 0;
      for (int s = 0; s < samps; ++s) {
        double r1 = 2 * sample_erand48(Xi);
        double jitter_x = diff_subpixel_jitter(r1);
        double r2 = 2 * sample_erand48(Xi);
        double jitter_y = diff_subpixel_jitter(r2);

        double u = ((sx + .5 + jitter_x) / 2 + pixel_x) / (double)w - .5;
        double v = ((sy + .5 + jitter_y) / 2 + pixel_y) / (double)h - .5;
        Vec cx_term = cxv * u;
        Vec cy_term = cy_scaled * v;
        Vec sum_xy = cx_term + cy_term;
        Vec d_unnorm = sum_xy + cam_dir;
        Vec d = normalize_vec(d_unnorm);
        Vec offset = d * 140;
        Vec origin = cam_pos + offset;
        Ray ray(origin, d);
        Vec L = diff_radiance(ray, 0, Xi, light_y, mirror_x);
        double sample_avg = vec_avg(L);
        subpix_sum = subpix_sum + sample_avg * inv_samps;
      }
      sum = sum + subpix_sum;
    }
  }
  return sum * 0.25;
}

inline double diff_objective_demo_aa(double cam_x, double cam_y, double cam_z) {
  return diff_objective_pixel(256, 192, 1, cam_x, cam_y, cam_z, kDefaultLightY,
                              kDefaultMirrorX);
}

inline double diff_objective_pixel_aa_samps(int pixel_x, int pixel_y,
                                            double cam_x, double cam_y,
                                            double cam_z, int samps) {
  return diff_objective_pixel(pixel_x, pixel_y, samps, cam_x, cam_y, cam_z,
                              kDefaultLightY, kDefaultMirrorX);
}

inline double diff_objective_patch_mean(double cam_x, double cam_y,
                                        double cam_z, int x0, int y0,
                                        int patch_w, int patch_h, int samps) {
  double sum = 0;
  int count = patch_w * patch_h;
  for (int py = y0; py < y0 + patch_h; ++py) {
    for (int px = x0; px < x0 + patch_w; ++px) {
      double pix =
          diff_objective_pixel_aa_samps(px, py, cam_x, cam_y, cam_z, samps);
      sum = sum + pix;
    }
  }
  return sum / count;
}

inline double diff_objective_light_y(double light_y) {
  return diff_objective_pixel(256, 192, 1, 50.0, 52.0, 295.6, light_y,
                              kDefaultMirrorX);
}

inline double diff_objective_pixel_aa_samps_mirror_x(int pixel_x, int pixel_y,
                                                     double mirror_x,
                                                     int samps) {
  return diff_objective_pixel(pixel_x, pixel_y, samps, 50.0, 52.0, 295.6,
                              kDefaultLightY, mirror_x);
}

inline double diff_objective_patch_mirror_x(double mirror_x, int x0, int y0,
                                            int patch_w, int patch_h,
                                            int samps) {
  double sum = 0;
  int count = patch_w * patch_h;
  for (int py = y0; py < y0 + patch_h; ++py) {
    for (int px = x0; px < x0 + patch_w; ++px) {
      double pix =
          diff_objective_pixel_aa_samps_mirror_x(px, py, mirror_x, samps);
      sum = sum + pix;
    }
  }
  return sum / count;
}

#endif // SMALLPT_DIFF_OBJECTIVES_H
