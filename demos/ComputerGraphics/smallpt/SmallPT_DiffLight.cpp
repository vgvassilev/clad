// A demo of clad's automatic differentiation capabilities on a minimal SmallPT
// renderer. This program simulates a static 3-sphere scene and recovers the
// position of a light source from a target image using gradient descent.
// Uses clad to calculate the gradient of the pixel loss with respect to the
// light source's center coordinates.
//
// To compile the demo please type:
// path/to/clang++ -O2 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang
// path/to/libclad.so -I../../include/ -std=c++17 SmallPT_DiffLight.cpp
// -o SmallPT_DiffLight
//
// To run the demo please type:
// ./SmallPT_DiffLight

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

// Static Scene: 3 spheres.
const double S0_CX = -3.0, S0_CY = 0.0, S0_CZ = 9.0, S0_R = 1.0;
const double S1_CX = 0.0, S1_CY = 0.3, S1_CZ = 10.0, S1_R = 1.5;
const double S2_CX = 3.0, S2_CY = -0.2, S2_CZ = 9.0, S2_R = 1.0;

// Base colors (white)
const double S_R = 0.8, S_G = 0.8, S_B = 0.8;

// Proximity calculation for soft silhouettes.
double ray_sphere_proximity(double ox, double oy, double oz, double dx,
                            double dy, double dz, double cx, double cy,
                            double cz, double r) {
  double vx = cx - ox;
  double vy = cy - oy;
  double vz = cz - oz;
  double t_closest = vx * dx + vy * dy + vz * dz;
  double t_eff = (t_closest + sqrt(t_closest * t_closest + 0.01)) * 0.5;
  double px = ox + dx * t_eff - cx;
  double py = oy + dy * t_eff - cy;
  double pz = oz + dz * t_eff - cz;
  double dist = sqrt(px * px + py * py + pz * pz + 1e-8);
  return dist - r;
}

double soft_weight(double sd, double sharpness) {
  return exp(-sharpness * sd * sd);
}

// Render a single pixel with a point light source.
double render_pixel(double lx, double ly, double lz, double px, double py,
                    double img_w, double img_h) {
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

  double sharpness = 8.0;

  // 1. Find the weight contribution for each sphere
  double w0 = soft_weight(ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz,
                                               S0_CX, S0_CY, S0_CZ, S0_R),
                          sharpness);
  double w1 = soft_weight(ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz,
                                               S1_CX, S1_CY, S1_CZ, S1_R),
                          sharpness);
  double w2 = soft_weight(ray_sphere_proximity(cam_x, cam_y, cam_z, dx, dy, dz,
                                               S2_CX, S2_CY, S2_CZ, S2_R),
                          sharpness);

  double total_w = w0 + w1 + w2 + 1e-8;

  // 2. Simple Lighting Model: Shading intensity based on light distance/angle.
  // Approximation: Intensity at hit point is proportional to (1 / distance)
  // Inverse linear rather than inverse-square to improve gradient conditioning
  double dist0 =
      sqrt((lx - S0_CX) * (lx - S0_CX) + (ly - S0_CY) * (ly - S0_CY) +
           (lz - S0_CZ) * (lz - S0_CZ) + 1e-8);
  double lum0 = 5.0 / dist0;

  double dist1 =
      sqrt((lx - S1_CX) * (lx - S1_CX) + (ly - S1_CY) * (ly - S1_CY) +
           (lz - S1_CZ) * (lz - S1_CZ) + 1e-8);
  double lum1 = 5.0 / dist1;

  double dist2 =
      sqrt((lx - S2_CX) * (lx - S2_CX) + (ly - S2_CY) * (ly - S2_CY) +
           (lz - S2_CZ) * (lz - S2_CZ) + 1e-8);
  double lum2 = 5.0 / dist2;

  return (w0 * lum0 + w1 * lum1 + w2 * lum2) / total_w;
}

double pixel_loss(double lx, double ly, double lz, double px, double py,
                  double img_w, double img_h, double target) {
  double rendered = render_pixel(lx, ly, lz, px, py, img_w, img_h);
  double diff = rendered - target;
  return diff * diff;
}

int main() {
  // 1. Ground truth Light position (at the top-left).
  double gt_lx = -3.0;
  double gt_ly = 3.0;
  double gt_lz = 5.0;

  std::filesystem::create_directories(
      "demos/ComputerGraphics/smallpt/visualizations/light");

  // 2. Render target image with ground truth light.
  std::array<double, W * H> tgt{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      tgt.at(y * W + x) = render_pixel(gt_lx, gt_ly, gt_lz, (double)x,
                                       (double)y, (double)W, (double)H);

  save_ppm("demos/ComputerGraphics/smallpt/visualizations/light/target.ppm", W,
           H, tgt);

  std::cout << "Target light at =(" << std::fixed << std::setprecision(2)
            << gt_lx << ", " << gt_ly << ", " << gt_lz << ")" << '\n';

  // 3. Start from perturbed light position.
  double lx = 2.0;
  double ly = -1.0;
  double lz = 2.0;
  std::cout << "Start light: L=(" << std::fixed << std::setprecision(2) << lx
            << ", " << ly << ", " << lz << ")" << '\n';

  std::ofstream csv("demos/ComputerGraphics/smallpt/visualizations/light/"
                    "light_trajectory.csv");
  csv << "step,lx,ly,lz,loss\n";

  std::array<double, W * H> start_frame{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      start_frame.at(y * W + x) =
          render_pixel(lx, ly, lz, (double)x, (double)y, (double)W, (double)H);
  save_ppm("demos/ComputerGraphics/smallpt/visualizations/light/start.ppm", W,
           H, start_frame);

  // 4. Differentiate `pixel_loss` w.r.t light position.
  auto grad = clad::gradient(pixel_loss, "lx, ly, lz");

  // Learning rate corresponds to the magnitude of the update step.
  double lr = 0.3;
  int max_steps = 60;

  // 5. Execute gradient descent.
  double m_x = 0;
  double m_y = 0;
  double m_z = 0;
  double v_x = 0;
  double v_y = 0;
  double v_z = 0;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double eps = 1e-8;
  for (int step = 1; step <= max_steps; step++) {
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
        grad.execute(lx, ly, lz, (double)x, (double)y, (double)W, (double)H, tv,
                     &dx, &dy, &dz);
        gx += dx;
        gy += dy;
        gz += dz;
        loss += pixel_loss(lx, ly, lz, (double)x, (double)y, (double)W,
                           (double)H, tv);
      }
    }

    gx /= (W * H);
    gy /= (W * H);
    gz /= (W * H);
    loss /= (W * H);

    // Adam Optimizer
    m_x = beta1 * m_x + (1 - beta1) * gx;
    m_y = beta1 * m_y + (1 - beta1) * gy;
    m_z = beta1 * m_z + (1 - beta1) * gz;

    v_x = beta2 * v_x + (1 - beta2) * gx * gx;
    v_y = beta2 * v_y + (1 - beta2) * gy * gy;
    v_z = beta2 * v_z + (1 - beta2) * gz * gz;

    double m_x_hat = m_x / (1 - pow(beta1, step));
    double m_y_hat = m_y / (1 - pow(beta1, step));
    double m_z_hat = m_z / (1 - pow(beta1, step));

    double v_x_hat = v_x / (1 - pow(beta2, step));
    double v_y_hat = v_y / (1 - pow(beta2, step));
    double v_z_hat = v_z / (1 - pow(beta2, step));

    lx -= lr * m_x_hat / (sqrt(v_x_hat) + eps);
    ly -= lr * m_y_hat / (sqrt(v_y_hat) + eps);
    lz -= lr * m_z_hat / (sqrt(v_z_hat) + eps);

    // Apply geometric learning rate decay.
    lr *= 0.995;

    double err =
        sqrt((lx - gt_lx) * (lx - gt_lx) + (ly - gt_ly) * (ly - gt_ly) +
             (lz - gt_lz) * (lz - gt_lz));

    csv << step << "," << lx << "," << ly << "," << lz << "," << loss << "\n";

    if (step % 25 == 0 || err < 0.5) {
      std::array<double, W * H> frame{};
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
          frame.at(y * W + x) = render_pixel(lx, ly, lz, (double)x, (double)y,
                                             (double)W, (double)H);
      save_ppm("demos/ComputerGraphics/smallpt/visualizations/light/frame_" +
                   std::to_string(step) + ".ppm",
               W, H, frame);

      std::cout << "Step " << std::setw(3) << step << ": loss=" << std::fixed
                << std::setprecision(6) << loss << " Light=("
                << std::setprecision(4) << lx << "," << ly << "," << lz
                << ") err=" << err << '\n';
    }

    if (err < 0.1) {
      std::array<double, W * H> frame{};
      for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
          frame.at(y * W + x) = render_pixel(lx, ly, lz, (double)x, (double)y,
                                             (double)W, (double)H);
      save_ppm("demos/ComputerGraphics/smallpt/visualizations/light/final.ppm",
               W, H, frame);

      std::cout << "Converged at step " << step << ". Light=("
                << std::setprecision(4) << lx << "," << ly << "," << lz
                << ") err=" << err << '\n';
      return 0;
    }
  }

  std::array<double, W * H> final_frame{};
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      final_frame.at(y * W + x) =
          render_pixel(lx, ly, lz, (double)x, (double)y, (double)W, (double)H);
  save_ppm("demos/ComputerGraphics/smallpt/visualizations/light/final.ppm", W,
           H, final_frame);

  return 0;
}
