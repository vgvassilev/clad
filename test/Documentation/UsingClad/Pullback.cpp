// RUN: %cladclang %s -I%S/../../../include -o%t 2>&1 | %filecheck_nodiag %s
// RUN: %t | %filecheck_prints %s

// docs-begin-pullback
#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct Calibration {
  explicit Calibration(double gain) : gain(gain) {}
  Calibration(const Calibration&) = delete;
  Calibration& operator=(const Calibration&) = delete;
  double gain;
};

double scale(const Calibration& cal, double x) { return cal.gain * x; }

int main() {
  auto pb = clad::pullback(scale);
  Calibration cal(5), d_cal(0);
  double dx = 1;
  pb.execute(cal, 3., 2., &d_cal, &dx);
  std::printf("d_gain = %g, dx = %g\n", d_cal.gain, dx);
  // prints: d_gain = 6, dx = 11
}
// docs-end-pullback
