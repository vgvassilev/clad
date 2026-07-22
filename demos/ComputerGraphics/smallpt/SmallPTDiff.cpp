// Inverse-rendering demo: recover light emission scale via clad + Adam.
//
// To compile the demo please type:
// path/to/clang++ -O0 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang
// path/to/clad.so -I../../include -I. -std=c++17 SmallPTDiff.cpp -o SmallPTDiff
//
// To run the demo please type:
// ./SmallPTDiff

#include "clad/Differentiator/Differentiator.h"

#include "SmallPTDiffObjectives.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kImgW = 512;
constexpr int kImgH = 384;
constexpr int kPatchW = 16;
constexpr int kPatchH = 16;
constexpr int kSamps = 1;
constexpr int kMaxSteps = 40;
constexpr int kVizW = 64;
constexpr int kVizH = 48;

inline double clamp01(double x) {
  if (x < 0)
    return 0;
  if (x > 1)
    return 1;
  return x;
}

inline int toInt(double x) {
  return int(std::pow(clamp01(x), 1.0 / 2.2) * 255 + 0.5);
}

void save_gray_ppm(const std::string& path, int w, int h,
                   const std::vector<double>& img) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs)
    return;
  ofs << "P6\n" << w << " " << h << "\n255\n";
  for (double val : img) {
    unsigned char c = static_cast<unsigned char>(toInt(val));
    ofs.put(static_cast<char>(c));
    ofs.put(static_cast<char>(c));
    ofs.put(static_cast<char>(c));
  }
}

void render_patch(double light_e, int x0, int y0, int pw, int ph, int samps,
                  std::vector<double>& out) {
  out.resize(static_cast<size_t>(pw * ph));
  for (int py = 0; py < ph; ++py) {
    for (int px = 0; px < pw; ++px) {
      out[static_cast<size_t>(py * pw + px)] =
          diff_objective_pixel_aa_samps_light_e(x0 + px, y0 + py, light_e,
                                                samps);
    }
  }
}

void render_downsampled(double light_e, int viz_w, int viz_h, int samps,
                        std::vector<double>& out) {
  out.resize(static_cast<size_t>(viz_w * viz_h));
  for (int vy = 0; vy < viz_h; ++vy) {
    int py = vy * kImgH / viz_h;
    for (int vx = 0; vx < viz_w; ++vx) {
      int px = vx * kImgW / viz_w;
      out[static_cast<size_t>(vy * viz_w + vx)] =
          diff_objective_pixel_aa_samps_light_e(px, py, light_e, samps);
    }
  }
}

double patch_mse(double light_e, int x0, int y0, int pw, int ph, int samps,
                 const std::vector<double>& target) {
  double loss = 0;
  for (int py = 0; py < ph; ++py) {
    for (int px = 0; px < pw; ++px) {
      double tv = target[static_cast<size_t>(py * pw + px)];
      loss += diff_pixel_loss_light_e(light_e, x0 + px, y0 + py, samps, tv);
    }
  }
  return loss / (pw * ph);
}

} // namespace

int main() {
  const int x0 = 248;
  const int y0 = 316;
  const double gt_light_e = kDefaultLightE;
  double light_e = 0.35;
  const std::string outdir = "visualizations/light_e";

  std::filesystem::create_directories(outdir);

  std::vector<double> target;
  render_patch(gt_light_e, x0, y0, kPatchW, kPatchH, kSamps, target);
  save_gray_ppm(outdir + "/target_patch.ppm", kPatchW, kPatchH, target);

  std::vector<double> start_patch;
  render_patch(light_e, x0, y0, kPatchW, kPatchH, kSamps, start_patch);
  save_gray_ppm(outdir + "/start_patch.ppm", kPatchW, kPatchH, start_patch);

  std::vector<double> viz;
  render_downsampled(gt_light_e, kVizW, kVizH, kSamps, viz);
  save_gray_ppm(outdir + "/target.ppm", kVizW, kVizH, viz);
  render_downsampled(light_e, kVizW, kVizH, kSamps, viz);
  save_gray_ppm(outdir + "/start.ppm", kVizW, kVizH, viz);

  std::cout << "Target light_e=" << std::fixed << std::setprecision(4)
            << gt_light_e << '\n';
  std::cout << "Start  light_e=" << light_e << '\n';

  std::ofstream csv(outdir + "/trajectory.csv");
  csv << "step,light_e,loss,err\n";

  auto grad = clad::gradient(diff_pixel_loss_light_e, "light_e");

  double lr = 0.08;
  double m = 0.0;
  double v = 0.0;
  const double beta1 = 0.9;
  const double beta2 = 0.999;
  const double eps = 1e-8;

  const double loss0 =
      patch_mse(light_e, x0, y0, kPatchW, kPatchH, kSamps, target);
  double loss = loss0;

  for (int step = 1; step <= kMaxSteps; ++step) {
    double g = 0.0;
    loss = 0.0;
    for (int py = 0; py < kPatchH; ++py) {
      for (int px = 0; px < kPatchW; ++px) {
        double tv = target[static_cast<size_t>(py * kPatchW + px)];
        double dg = 0.0;
        grad.execute(light_e, x0 + px, y0 + py, kSamps, tv, &dg);
        g += dg;
        loss +=
            diff_pixel_loss_light_e(light_e, x0 + px, y0 + py, kSamps, tv);
      }
    }
    g /= (kPatchW * kPatchH);
    loss /= (kPatchW * kPatchH);

    m = beta1 * m + (1 - beta1) * g;
    v = beta2 * v + (1 - beta2) * g * g;
    double m_hat = m / (1 - std::pow(beta1, step));
    double v_hat = v / (1 - std::pow(beta2, step));
    light_e -= lr * m_hat / (std::sqrt(v_hat) + eps);
    if (light_e < 0.0)
      light_e = 0.0;
    lr *= 0.995;

    double err = std::fabs(light_e - gt_light_e);
    csv << step << "," << light_e << "," << loss << "," << err << "\n";

    if (step % 5 == 0 || err < 0.05) {
      std::cout << "Step " << std::setw(3) << step << ": loss=" << std::fixed
                << std::setprecision(6) << loss << " light_e="
                << std::setprecision(4) << light_e << " err=" << err << '\n';
    }

    if (err < 0.05)
      break;
  }

  render_patch(light_e, x0, y0, kPatchW, kPatchH, kSamps, start_patch);
  save_gray_ppm(outdir + "/final_patch.ppm", kPatchW, kPatchH, start_patch);
  render_downsampled(light_e, kVizW, kVizH, kSamps, viz);
  save_gray_ppm(outdir + "/final.ppm", kVizW, kVizH, viz);

  const double err = std::fabs(light_e - gt_light_e);
  std::cout << "Final light_e=" << std::setprecision(4) << light_e
            << " err=" << err << " loss0=" << std::setprecision(6) << loss0
            << " loss=" << loss << '\n';

  const bool ok = err < 0.15 && loss < loss0;
  std::cout << "SMALLPT_DIFF_INVERSE_PASS=" << (ok ? 1 : 0) << '\n';
  return ok ? 0 : 1;
}
