// A demo of clad's automatic differentiation capabilities on a minimal SmallPT
// renderer. This program simulates a 3-sphere scene using soft rasterization
// and recovers the camera position from a target image using gradient descent.
// Uses clad to calculate the gradient of the pixel loss with respect to the
// camera position.
//
// To compile the demo please type:
// path/to/clang++ -O2 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang
// path/to/libclad.so -I../../include/ -std=c++17 SmallPT_DiffGeom.cpp
// -o SmallPT_DiffGeom
//
// To run the demo please type:
// ./SmallPT_DiffGeom

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

// Scene: 3 spheres.

// Sphere 0: left, red
const double S0_CX = -2.0, S0_CY = 0.0, S0_CZ = 8.0, S0_R = 1.0;
const double S0_CR = 0.9, S0_CG = 0.2, S0_CB = 0.1;

// Sphere 1: center, green
const double S1_CX = 0.5, S1_CY = 0.3, S1_CZ = 10.0, S1_R = 1.5;
const double S1_CR = 0.2, S1_CG = 0.8, S1_CB = 0.1;

// Sphere 2: right, blue
const double S2_CX = 2.5, S2_CY = -0.2, S2_CZ = 7.0, S2_R = 1.0;
const double S2_CR = 0.1, S2_CG = 0.2, S2_CB = 0.9;

// Soft rendering: for each sphere, compute a smooth visibility weight
// based on the minimum distance from the ray to the sphere surface.
//
// For a ray R(t) = O + t*D and sphere center C with radius r:
//   closest_approach = ||(C-O) - ((C-O)·D)*D||
//   signed_distance = closest_approach - r
//
// Weight = exp(-k * max(0, signed_distance)^2) for sharp = k
// Higher k = sharper silhouettes.

// Compute the squared distance from a ray to a sphere center, minus radius^2.
// This is fully differentiable w.r.t. cam position (ox, oy, oz).
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
  // When sd <= 0 (inside sphere), weight ≈ 1
  // When sd > 0 (outside), weight falls off as exp(-k * sd^2)
  // Smooth version: use sd^2 regardless of sign for gradient flow
  return exp(-sharpness * sd * sd);
}

// Render one pixel: soft-blend of all sphere colors.
// Returns a scalar (weighted average of sphere intensities).

double render_pixel(double cam_x, double cam_y, double cam_z, double px,
                    double py, double img_w, double img_h) {
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

  double sd1 = ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz, S1_CX,
                                    S1_CY, S1_CZ, S1_R);
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

double pixel_loss(double cam_x, double cam_y, double cam_z, double px,
                  double py, double img_w, double img_h, double target) {
  double rendered = render_pixel(cam_x, cam_y, cam_z, px, py, img_w, img_h);
  double diff = rendered - target;
  return diff * diff;
}

// Main: gradient descent to recover camera position.

int main() {
  const int W = 16;
  const int H = 12;

  // 1. Ground truth camera position.
  double gt_x = 0.0;
  double gt_y = 1.0;
  double gt_z = 0.0;

  std::filesystem::create_directories(
      "demos/ComputerGraphics/smallpt/visualizations/geom");

  // 2. Render target image.
  std::array<double, W * H> tgt{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      tgt.at(y * W + x) = render_pixel(gt_x, gt_y, gt_z, (double)x, (double)y,
                                       (double)W, (double)H);

  save_ppm("demos/ComputerGraphics/smallpt/visualizations/geom/target.ppm", W,
           H, tgt);

  std::cout << "Target rendered at cam=(" << std::fixed << std::setprecision(2)
            << gt_x << ", " << gt_y << ", " << gt_z << ")" << '\n';

  // 3. Start from perturbed position.
  double cx = 1.5;
  double cy = 1.8;
  double cz = -1.0;
  std::cout << "Start: cam=(" << std::fixed << std::setprecision(2) << cx
            << ", " << cy << ", " << cz << ")" << '\n';

  std::ofstream csv(
      "demos/ComputerGraphics/smallpt/visualizations/geom/geom_trajectory.csv");
  csv << "step,cx,cy,cz,loss\n";

  std::array<double, W * H> start_frame{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      start_frame.at(y * W + x) =
          render_pixel(cx, cy, cz, (double)x, (double)y, (double)W, (double)H);
  save_ppm("demos/ComputerGraphics/smallpt/visualizations/geom/start.ppm", W, H,
           start_frame);

  // 4. Differentiate `pixel_loss` w.r.t camera position.
  auto grad = clad::gradient(pixel_loss, "cam_x, cam_y, cam_z");

  double lr = 2.0;
  int max_steps = 500;

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

    double ex = cx - gt_x;
    double ey = cy - gt_y;
    double ez = cz - gt_z;
    double err = sqrt(ex * ex + ey * ey + ez * ez);

    csv << step << "," << cx << "," << cy << "," << cz << "," << loss << "\n";

    if (step % 25 == 0 || err < 0.5) {
      std::array<double, W * H> frame{};
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
          frame.at(y * W + x) = render_pixel(cx, cy, cz, (double)x, (double)y,
                                             (double)W, (double)H);
      save_ppm("demos/ComputerGraphics/smallpt/visualizations/geom/frame_" +
                   std::to_string(step) + ".ppm",
               W, H, frame);

      std::cout << "Step " << std::setw(3) << step << ": loss=" << std::fixed
                << std::setprecision(6) << loss << " cam=("
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
      save_ppm("demos/ComputerGraphics/smallpt/visualizations/geom/final.ppm",
               W, H, frame);

      std::cout << "Converged at step " << step << ". cam=("
                << std::setprecision(4) << cx << "," << cy << "," << cz
                << ") err=" << std::setprecision(6) << err << '\n';
      return 0;
    }
  }

  double ex = cx - gt_x;
  double ey = cy - gt_y;
  double ez = cz - gt_z;
  double err = sqrt(ex * ex + ey * ey + ez * ez);
  if (err < 1.0) {
    std::cout << "Converged (within tolerance). cam=(" << std::fixed
              << std::setprecision(4) << cx << "," << cy << "," << cz
              << ") err=" << err << '\n';
  } else {
    std::cout << "Did not converge after " << max_steps << " steps. cam=("
              << std::fixed << std::setprecision(2) << cx << "," << cy << ","
              << cz << ") err=" << std::setprecision(4) << err << '\n';
  }

  std::array<double, W * H> final_frame{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      final_frame.at(y * W + x) =
          render_pixel(cx, cy, cz, (double)x, (double)y, (double)W, (double)H);
  save_ppm("demos/ComputerGraphics/smallpt/visualizations/geom/final.ppm", W, H,
           final_frame);

  return 0;
}
