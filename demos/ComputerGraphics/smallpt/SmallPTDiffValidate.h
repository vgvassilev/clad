#ifndef SMALLPT_DIFF_VALIDATE_H
#define SMALLPT_DIFF_VALIDATE_H

#include "SmallPTDiffObjectives.h"

#include <cmath>
#include <cstdio>

static constexpr int kSmallPTWidth = 512;
static constexpr int kSmallPTHeight = 384;

static inline bool smallpt_essentially_equal(double a, double b,
                                             double rel = 5e-2) {
  if (std::fabs(a) < 1e-12 && std::fabs(b) < 1e-12)
    return true;
  return std::fabs(a - b) <= rel * std::fmax(std::fabs(a), std::fabs(b));
}

static inline double smallpt_central_fd3(double (*f)(double, double, double),
                                         double x, double y, double z, int dim,
                                         double h = 1e-4) {
  double args[3] = {x, y, z};
  args[dim] += h;
  double fp = f(args[0], args[1], args[2]);
  args[dim] -= 2 * h;
  double fm = f(args[0], args[1], args[2]);
  return (fp - fm) / (2 * h);
}

using smallpt_patch_fd_fn = double (*)(double, double, double, int, int, int,
                                       int, int);

static inline double smallpt_central_fd_patch(smallpt_patch_fd_fn f, double x,
                                              double y, double z, int x0,
                                              int y0, int pw, int ph, int samps,
                                              int dim, double h = 1e-4) {
  double args[3] = {x, y, z};
  args[dim] += h;
  double fp = f(args[0], args[1], args[2], x0, y0, pw, ph, samps);
  args[dim] -= 2 * h;
  double fm = f(args[0], args[1], args[2], x0, y0, pw, ph, samps);
  return (fp - fm) / (2 * h);
}

static inline int smallpt_run_camera_grad_check() {
  double cam_x = 50.0, cam_y = 52.0, cam_z = 295.6;
  double g_x = 0, g_y = 0, g_z = 0;

  auto grad = clad::gradient(diff_objective_demo_aa, "cam_x,cam_y,cam_z");
  grad.execute(cam_x, cam_y, cam_z, &g_x, &g_y, &g_z);

  bool pass = smallpt_essentially_equal(
      g_x, smallpt_central_fd3(diff_objective_demo_aa, cam_x, cam_y, cam_z, 0));
  pass &= smallpt_essentially_equal(
      g_y, smallpt_central_fd3(diff_objective_demo_aa, cam_x, cam_y, cam_z, 1));
  pass &= smallpt_essentially_equal(
      g_z, smallpt_central_fd3(diff_objective_demo_aa, cam_x, cam_y, cam_z, 2));

  printf("SMALLPT_GRAD_FD_PASS=%d\n", pass ? 1 : 0);
  return pass ? 0 : 1;
}

static inline int smallpt_run_patch_grad_check(int patch_w, int patch_h,
                                               int samps) {
  double cam_x = 50.0, cam_y = 52.0, cam_z = 295.6;
  int x0 = kSmallPTWidth / 2 - patch_w / 2;
  int y0 = kSmallPTHeight / 2 - patch_h / 2;
  double g_x = 0, g_y = 0, g_z = 0;

  auto grad = clad::gradient(diff_objective_patch_mean, "cam_x,cam_y,cam_z");
  grad.execute(cam_x, cam_y, cam_z, x0, y0, patch_w, patch_h, samps, &g_x, &g_y,
               &g_z);

  bool pass = smallpt_essentially_equal(
      g_x, smallpt_central_fd_patch(diff_objective_patch_mean, cam_x, cam_y,
                                    cam_z, x0, y0, patch_w, patch_h, samps, 0));
  pass &= smallpt_essentially_equal(
      g_y, smallpt_central_fd_patch(diff_objective_patch_mean, cam_x, cam_y,
                                    cam_z, x0, y0, patch_w, patch_h, samps, 1));
  pass &= smallpt_essentially_equal(
      g_z, smallpt_central_fd_patch(diff_objective_patch_mean, cam_x, cam_y,
                                    cam_z, x0, y0, patch_w, patch_h, samps, 2));

  printf("SMALLPT_PATCH_GRAD_FD_PASS=%d\n", pass ? 1 : 0);
  return pass ? 0 : 1;
}

static inline double smallpt_central_fd1(double (*f)(double), double x,
                                         double h = 1e-4) {
  double fp = f(x + h);
  double fm = f(x - h);
  return (fp - fm) / (2 * h);
}

static inline int smallpt_run_light_y_grad_check() {
  double light_y = kDefaultLightY;
  double g_y = 0;

  auto grad = clad::gradient(diff_objective_light_y, "light_y");
  grad.execute(light_y, &g_y);

  bool pass = smallpt_essentially_equal(
      g_y, smallpt_central_fd1(diff_objective_light_y, light_y));

  printf("SMALLPT_LIGHT_Y_GRAD_FD_PASS=%d\n", pass ? 1 : 0);
  return pass ? 0 : 1;
}

using smallpt_mirror_patch_fd_fn = double (*)(double, int, int, int, int, int);

static inline double
smallpt_central_fd_mirror_patch(smallpt_mirror_patch_fd_fn f, double mirror_x,
                                int x0, int y0, int patch_w, int patch_h,
                                int samps, double h = 1e-4) {
  double fp = f(mirror_x + h, x0, y0, patch_w, patch_h, samps);
  double fm = f(mirror_x - h, x0, y0, patch_w, patch_h, samps);
  return (fp - fm) / (2 * h);
}

static inline int smallpt_run_mirror_patch_grad_check() {
  double mirror_x = kDefaultMirrorX;
  int x0 = kMirrorRadiancePatchX0;
  int y0 = kMirrorRadiancePatchY0;
  int patch_w = kMirrorRadiancePatchW;
  int patch_h = kMirrorRadiancePatchH;
  int samps = 1;

  double radiance =
      diff_objective_patch_mirror_x(mirror_x, x0, y0, patch_w, patch_h, samps);
  if (radiance < 0.01) {
    printf("PATCH_MIRROR_RADIANCE_FAIL=1\n");
    return 1;
  }
  printf("PATCH_MIRROR_RADIANCE=%g\n", radiance);

  double g_x = 0;
  auto grad = clad::gradient(diff_objective_patch_mirror_x, "mirror_x");
  grad.execute(mirror_x, x0, y0, patch_w, patch_h, samps, &g_x);

  bool pass = smallpt_essentially_equal(
      g_x,
      smallpt_central_fd_mirror_patch(diff_objective_patch_mirror_x, mirror_x,
                                      x0, y0, patch_w, patch_h, samps));

  printf("SMALLPT_MIRROR_PATCH_GRAD_FD_PASS=%d\n", pass ? 1 : 0);
  return pass ? 0 : 1;
}

static inline int smallpt_run_light_e_grad_check() {
  double light_e = kDefaultLightE;
  double g_e = 0;

  auto grad = clad::gradient(diff_objective_light_e, "light_e");
  grad.execute(light_e, &g_e);

  bool pass = smallpt_essentially_equal(
      g_e, smallpt_central_fd1(diff_objective_light_e, light_e));
  pass &= std::fabs(g_e) > 1e-6;

  printf("SMALLPT_LIGHT_E_GRAD_FD_PASS=%d\n", pass ? 1 : 0);
  return pass ? 0 : 1;
}

static inline int smallpt_run_light_e_gd_smoke() {
  constexpr int patch_w = kLightEmitPatchW;
  constexpr int patch_h = kLightEmitPatchH;
  constexpr int samps = 1;
  constexpr int steps = 10;
  const int x0 = kLightEmitPatchX0;
  const int y0 = kLightEmitPatchY0;
  const double gt = kDefaultLightE;
  double light_e = 0.4;

  double targets[patch_w * patch_h];
  for (int py = 0; py < patch_h; ++py) {
    for (int px = 0; px < patch_w; ++px) {
      targets[py * patch_w + px] =
          diff_objective_pixel_aa_samps_light_e(x0 + px, y0 + py, gt, samps);
    }
  }

  auto patch_loss = [&](double le) {
    double loss = 0;
    for (int py = 0; py < patch_h; ++py) {
      for (int px = 0; px < patch_w; ++px) {
        loss += diff_pixel_loss_light_e(le, x0 + px, y0 + py, samps,
                                        targets[py * patch_w + px]);
      }
    }
    return loss / (patch_w * patch_h);
  };

  const double loss0 = patch_loss(light_e);
  const double err0 = std::fabs(light_e - gt);

  auto grad = clad::gradient(diff_pixel_loss_light_e, "light_e");
  double lr = 0.05;
  double m = 0.0;
  double v = 0.0;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;

  for (int step = 1; step <= steps; ++step) {
    double g = 0.0;
    for (int py = 0; py < patch_h; ++py) {
      for (int px = 0; px < patch_w; ++px) {
        double dg = 0.0;
        grad.execute(light_e, x0 + px, y0 + py, samps,
                     targets[py * patch_w + px], &dg);
        g += dg;
      }
    }
    g /= (patch_w * patch_h);
    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g * g;
    double m_hat = m / (1 - std::pow(beta1, step));
    double v_hat = v / (1 - std::pow(beta2, step));
    light_e -= lr * m_hat / (std::sqrt(v_hat) + eps);
    if (light_e < 0.0)
      light_e = 0.0;
    lr *= 0.995;
  }

  const double loss = patch_loss(light_e);
  const double err = std::fabs(light_e - gt);
  bool pass = loss < loss0 && err < err0;
  printf("SMALLPT_LIGHT_E_GD_SMOKE_PASS=%d\n", pass ? 1 : 0);
  return pass ? 0 : 1;
}

#endif // SMALLPT_DIFF_VALIDATE_H
