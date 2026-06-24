// A demo of clad's automatic differentiation capabilities on a minimal SmallPT
// renderer. This program simulates a 3-sphere scene using soft rasterization
// and recovers the position of a sphere from a target image using gradient
// descent. Uses clad to calculate the gradient of the pixel loss with respect
// to the sphere's center coordinates.
//
// To compile the demo please type:
// path/to/clang++ -O2 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang
// path/to/libclad.so -I../../include/ -std=c++17 SmallPT_DiffSphere.cpp
// -o SmallPT_DiffSphere
//
// To run the demo please type:
// ./SmallPT_DiffSphere

#include "clad/Differentiator/Differentiator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

template <size_t N>
void save_ppm(const std::string& filename, int w, int h,
              const std::array<double, N>& img) {
  std::ofstream ofs(filename, std::ios::binary);
  if (!ofs)
    return;
  ofs << "P6\n" << w << " " << h << "\n255\n";
  for (double val : img) {
    int c = std::min(255, std::max(0, (int)(val * 255.0)));
    auto cv = (unsigned char)c;
    ofs << cv << cv << cv;
  }
}

// Image Resolution
const int W = 16;
const int H = 12;

// Scene: 3 spheres.

// Sphere 0: left, red
const double S0_CX = -4.5, S0_CY = 0.0, S0_CZ = 9.0, S0_R = 1.0;
const double S0_CR = 0.9, S0_CG = 0.2, S0_CB = 0.1;

// Sphere 1: center, green (Target for optimization)
const double S1_R = 1.5;
const double S1_CR = 0.2, S1_CG = 0.8, S1_CB = 0.1;

// Sphere 2: right, blue
const double S2_CX = 4.5, S2_CY = -0.2, S2_CZ = 9.0, S2_R = 1.0;
const double S2_CR = 0.1, S2_CG = 0.2, S2_CB = 0.9;

// Compute the squared distance from a ray to a sphere center, minus radius^2.
// This is fully differentiable w.r.t. the sphere center (cx, cy, cz).
double ray_sphere_proximity(double ox, double oy, double oz, double dx,
                            double dy, double dz, double cx, double cy,
                            double cz, double r) {
  // Vector from ray origin to sphere center
  double vx = cx - ox;
  double vy = cy - oy;
  double vz = cz - oz;

  // Project onto ray direction (d is normalized)
  double t_closest = vx * dx + vy * dy + vz * dz;

  // If closest point is behind camera, sphere is behind us
  // Use a soft version: t_eff = max(0, t_closest)
  // Approximated as: t_eff = (t_closest + sqrt(t_closest^2 + epsilon)) / 2
  double t_eff = (t_closest + sqrt(t_closest * t_closest + 0.01)) * 0.5;

  // Closest point on ray to sphere center
  double px = ox + dx * t_eff - cx;
  double py = oy + dy * t_eff - cy;
  double pz = oz + dz * t_eff - cz;

  // Distance from closest point to sphere center
  double dist_sq = px * px + py * py + pz * pz;
  double dist = sqrt(dist_sq + 1e-8);

  // Signed distance to sphere surface
  double sd = dist - r;

  return sd;
}

// Soft weight: gaussian falloff from sphere surface.
double soft_weight(double sd, double sharpness) {
  return exp(-sharpness * sd * sd);
}

// Render one pixel: soft-blend of all sphere colors.
// Returns a scalar (weighted average of sphere intensities).
double render_pixel(double s1_cx, double s1_cy, double s1_cz, double px,
                    double py, double img_w, double img_h) {
  // Hardcoded Camera Position
  double cam_x = 0.0;
  double cam_y = 1.0;
  double cam_z = 0.0;

  double aspect = img_w / img_h;
  double u = ((px + 0.5) / img_w - 0.5) * aspect;
  double v = (py + 0.5) / img_h - 0.5;

  double dx = u;
  double dy = v;
  double dz = 1.0;
  double dl = sqrt(dx * dx + dy * dy + dz * dz);
  dx /= dl;
  dy /= dl;
  dz /= dl;

  double sharpness = 8.0; // Controls silhouette sharpness

  // Compute soft weights for each sphere
  double sd0 = ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz, S0_CX,
                                    S0_CY, S0_CZ, S0_R);
  double w0 = soft_weight(sd0, sharpness);

  double sd1 = ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz, s1_cx,
                                    s1_cy, s1_cz, S1_R);
  double w1 = soft_weight(sd1, sharpness);

  double sd2 = ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz, S2_CX,
                                    S2_CY, S2_CZ, S2_R);
  double w2 = soft_weight(sd2, sharpness);

  // Weighted color (using grayscale luminance of each sphere)
  double lum0 = 0.299 * S0_CR + 0.587 * S0_CG + 0.114 * S0_CB;
  double lum1 = 0.299 * S1_CR + 0.587 * S1_CG + 0.114 * S1_CB;
  double lum2 = 0.299 * S2_CR + 0.587 * S2_CG + 0.114 * S2_CB;

  double total_w = w0 + w1 + w2 + 1e-8; // avoid div by zero
  double intensity = (w0 * lum0 + w1 * lum1 + w2 * lum2) / total_w;

  return intensity;
}

// Per-pixel loss (MSE).
double pixel_loss(double s1_cx, double s1_cy, double s1_cz, double px,
                  double py, double img_w, double img_h, double target) {
  double rendered = render_pixel(s1_cx, s1_cy, s1_cz, px, py, img_w, img_h);
  double diff = rendered - target;
  return diff * diff;
}

// Main: gradient descent to recover sphere 1 position.
int main() {
  // 1. Ground truth S1 position.
  double gt_cx = 0.5;
  double gt_cy = 0.3;
  double gt_cz = 10.0;

  std::filesystem::create_directories(
      "demos/ComputerGraphics/smallpt/visualizations/object");

  // 2. Render target image.
  std::array<double, W * H> tgt{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      tgt.at(y * W + x) = render_pixel(gt_cx, gt_cy, gt_cz, (double)x,
                                       (double)y, (double)W, (double)H);

  save_ppm("demos/ComputerGraphics/smallpt/visualizations/object/target.ppm", W,
           H, tgt);

  std::cout << "Target rendered with S1 at =(" << std::fixed
            << std::setprecision(2) << gt_cx << ", " << gt_cy << ", " << gt_cz
            << ")" << '\n';

  // 3. Start from perturbed sphere position.
  double cx = 2.5;
  double cy = -2.5;
  double cz = 6.0;
  std::cout << "Start: S1=(" << std::fixed << std::setprecision(2) << cx << ", "
            << cy << ", " << cz << ")" << '\n';

  std::ofstream csv("demos/ComputerGraphics/smallpt/visualizations/object/"
                    "sphere_trajectory.csv");
  csv << "step,cx,cy,cz,loss\n";

  std::array<double, W * H> start_frame{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      start_frame.at(y * W + x) =
          render_pixel(cx, cy, cz, (double)x, (double)y, (double)W, (double)H);
  save_ppm("demos/ComputerGraphics/smallpt/visualizations/object/start.ppm", W,
           H, start_frame);

  // 4. Differentiate `pixel_loss` w.r.t S1 position.
  auto grad = clad::gradient(pixel_loss, "s1_cx, s1_cy, s1_cz");

  double lr = 15.0;
  int max_steps = 750;

  // 5. Execute gradient descent.
  for (int step = 0; step < max_steps; step++) {
    double gx = 0;
    double gy = 0;
    double gz = 0;
    double loss = 0;

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        double tv = tgt.at(y * W + x);
        double dx = 0;
        double dy = 0;
        double dz = 0;
        grad.execute(cx, cy, cz, (double)x, (double)y, (double)W, (double)H, tv,
                     &dx, &dy, &dz);
        gx += dx;
        gy += dy;
        gz += dz;
        loss += pixel_loss(cx, cy, cz, (double)x, (double)y, (double)W,
                           (double)H, tv);
      }
    }

    double n = W * H;
    gx /= n;
    gy /= n;
    gz /= n;
    loss /= n;

    // Gradient clipping.
    double gn = sqrt(gx * gx + gy * gy + gz * gz);
    if (gn > 2.0) {
      gx = gx * 2.0 / gn;
      gy = gy * 2.0 / gn;
      gz = gz * 2.0 / gn;
    }

    cx -= lr * gx;
    cy -= lr * gy;
    cz -= lr * gz;

    double ex = cx - gt_cx;
    double ey = cy - gt_cy;
    double ez = cz - gt_cz;
    double err = sqrt(ex * ex + ey * ey + ez * ez);

    csv << step << "," << cx << "," << cy << "," << cz << "," << loss << "\n";

    if (step % 25 == 0 || err < 0.5) {
      std::array<double, W * H> frame{};
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
          frame.at(y * W + x) = render_pixel(cx, cy, cz, (double)x, (double)y,
                                             (double)W, (double)H);
      save_ppm("demos/ComputerGraphics/smallpt/visualizations/object/frame_" +
                   std::to_string(step) + ".ppm",
               W, H, frame);

      std::cout << "Step " << std::setw(3) << step << ": loss=" << std::fixed
                << std::setprecision(6) << loss << " S1=("
                << std::setprecision(4) << cx << "," << cy << "," << cz
                << ") err=" << err << " grad=(" << std::scientific
                << std::setprecision(4) << gx << "," << gy << "," << gz << ")"
                << '\n';
      std::cout << std::fixed; // Reset to fixed for other outputs
    }

    if (err < 0.1) {
      std::array<double, W * H> frame{};
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
          frame.at(y * W + x) = render_pixel(cx, cy, cz, (double)x, (double)y,
                                             (double)W, (double)H);
      save_ppm("demos/ComputerGraphics/smallpt/visualizations/object/final.ppm",
               W, H, frame);

      std::cout << "Converged at step " << step << ". S1=("
                << std::setprecision(4) << cx << "," << cy << "," << cz
                << ") err=" << std::setprecision(6) << err << '\n';
      return 0;
    }
  }

  double ex = cx - gt_cx;
  double ey = cy - gt_cy;
  double ez = cz - gt_cz;
  double err = sqrt(ex * ex + ey * ey + ez * ez);
  if (err < 1.0) {
    std::cout << "Converged (within tolerance). S1=(" << std::fixed
              << std::setprecision(4) << cx << "," << cy << "," << cz
              << ") err=" << err << '\n';
  } else {
    std::cout << "Did not converge after " << max_steps << " steps. S1=("
              << std::fixed << std::setprecision(2) << cx << "," << cy << ","
              << cz << ") err=" << std::setprecision(4) << err << '\n';
  }

  std::array<double, W * H> final_frame{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      final_frame.at(y * W + x) =
          render_pixel(cx, cy, cz, (double)x, (double)y, (double)W, (double)H);
  save_ppm("demos/ComputerGraphics/smallpt/visualizations/object/final.ppm", W,
           H, final_frame);

  return 0;
}
