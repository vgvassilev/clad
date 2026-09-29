//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Fits a helix to measured points with Levenberg-Marquardt. Clad supplies the
// derivatives of the residuals, which is the whole of what the method needs
// from the model.
//
// Reduced from the helix fitter contributed in
// https://github.com/vgvassilev/clad/pull/1202 by PPaye.
//
//----------------------------------------------------------------------------//

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <cmath>
#include <cstdio>

const int kHits = 60;
const int kParams = 5;

// The helix being fitted, written the way it would be anywhere else in a
// physics program. Clad differentiates with respect to the whole thing and
// hands back another Helix, so a derivative is named rather than indexed:
// the change in a residual per unit change in the radius is g.r.
struct Helix {
  double cx, cy; // where the circle sits, seen down the field
  double r;      // its radius, which is what the momentum is read from
  double z0;     // height at angle zero
  double lam;    // how fast height rises with angle
};

// A charged particle in a magnetic field moves along a helix, and a detector
// reports points it passed through. Recovering the helix from those points is
// how the particle's momentum is measured.
//
// Seen down the field direction the helix is a circle of radius r about
// (cx, cy); along the field the height rises steadily with the angle. So each
// point gives two numbers that should be zero when the helix is right: how far
// it sits from the circle, and how far its height is from the expected one.
// Those are the residuals, and fitting means making them small.
//
// Taking the field along z is what makes the residuals this short. A helix at
// any orientation needs two more parameters and a distance found by solving
// for the nearest point, which is a bigger program than this one.
// docs-begin-helix
double radial_residual(Helix h, double x, double y, double z) {
  double dx = x - h.cx, dy = y - h.cy;
  return std::sqrt(dx * dx + dy * dy) - h.r;
}

double z_residual(Helix h, double x, double y, double z) {
  return z - (h.z0 + h.lam * std::atan2(y - h.cy, x - h.cx));
}
// docs-end-helix

// Solve A s = b. After damping, J'J is symmetric and positive definite, so
// it factors as L L' with no pivoting to choose and no rows to swap. A
// negative pivot means it was not positive definite after all, which tells
// the caller to damp harder.
static bool solve(double A[kParams][kParams], double b[kParams],
                  double s[kParams]) {
  for (int i = 0; i < kParams; ++i)
    for (int j = 0; j <= i; ++j) {
      double acc = A[i][j];
      for (int k = 0; k < j; ++k)
        acc -= A[i][k] * A[j][k];
      if (i != j)
        A[i][j] = acc / A[j][j];
      else if (acc <= 1e-14)
        return false;
      else
        A[i][i] = std::sqrt(acc);
    }

  // Forward through L, then back through its transpose.
  for (int i = 0; i < kParams; ++i) {
    double acc = b[i];
    for (int k = 0; k < i; ++k)
      acc -= A[i][k] * s[k];
    s[i] = acc / A[i][i];
  }
  for (int i = kParams - 1; i >= 0; --i) {
    double acc = s[i];
    for (int k = i + 1; k < kParams; ++k)
      acc -= A[k][i] * s[k];
    s[i] = acc / A[i][i];
  }
  return true;
}

int main() {
  const Helix truth = {0.4, -0.3, 2.5, 0.1, 0.8};

  // Points along the true helix, nudged off it so the fit has something to do.
  // The generator is written out rather than drawn from <random> so the demo
  // prints the same numbers wherever it runs.
  double hx[kHits], hy[kHits], hz[kHits];
  unsigned seed = 12345;
  for (int i = 0; i < kHits; ++i) {
    double phi = -1.2 + 2.4 * i / (kHits - 1);
    seed = seed * 1103515245u + 12345u;
    double jitter = ((int)((seed >> 16) % 2001) - 1000) / 1000.0 * 0.02;
    hx[i] = truth.cx + truth.r * std::cos(phi) + jitter;
    hy[i] = truth.cy + truth.r * std::sin(phi) - jitter;
    hz[i] = truth.z0 + truth.lam * phi + jitter;
  }

  // Everything the fit knows about the model comes from these two lines. Each
  // gives the derivatives of one residual with respect to the five helix
  // parameters, which is one row of the Jacobian per measured point.
  // docs-begin-helix-call
  auto d_radial = clad::gradient(radial_residual, "h");
  auto d_z = clad::gradient(z_residual, "h");
  // docs-end-helix-call

  Helix p = {0., 0., 2., 0., 1.};
  double lambda = 1e-3;
  double cost = 0.;

  for (int step = 1; step <= 30; ++step) {
    // Levenberg-Marquardt solves (J'J + lambda diag) s = -J'r for the step s.
    // Build J'J and J'r a point at a time; neither matrix is ever stored whole.
    // rhs accumulates -J'r as it goes, so it is the right-hand side already.
    double JtJ[kParams][kParams] = {{0.}}, rhs[kParams] = {0.};
    cost = 0.;
    for (int i = 0; i < kHits; ++i) {
      // One Helix of derivatives per residual. Laying them out as rows is
      // the only place the parameters stop being named and become a vector,
      // which is what the linear algebra below needs them to be.
      Helix g0 = {}, g1 = {};
      d_radial.execute(p, hx[i], hy[i], hz[i], &g0);
      d_z.execute(p, hx[i], hy[i], hz[i], &g1);
      double row[2][kParams] = {{g0.cx, g0.cy, g0.r, g0.z0, g0.lam},
                                {g1.cx, g1.cy, g1.r, g1.z0, g1.lam}};
      double res[2] = {radial_residual(p, hx[i], hy[i], hz[i]),
                       z_residual(p, hx[i], hy[i], hz[i])};
      for (int k = 0; k < 2; ++k) {
        cost += res[k] * res[k];
        for (int a = 0; a < kParams; ++a) {
          rhs[a] -= row[k][a] * res[k];
          for (int b = 0; b < kParams; ++b)
            JtJ[a][b] += row[k][a] * row[k][b];
        }
      }
    }

    // lambda decides how much of a Newton step this is and how much of a
    // short step downhill. Large lambda is cautious and always improves a
    // little; small lambda converges fast once the model is nearly right.
    for (int a = 0; a < kParams; ++a)
      JtJ[a][a] *= (1. + lambda);

    double s[kParams] = {0.};
    if (!solve(JtJ, rhs, s)) {
      lambda *= 10;
      continue;
    }

    Helix trial = {p.cx + s[0], p.cy + s[1], p.r + s[2], p.z0 + s[3],
                   p.lam + s[4]};
    double newcost = 0.;
    for (int i = 0; i < kHits; ++i) {
      double rr = radial_residual(trial, hx[i], hy[i], hz[i]);
      double rz = z_residual(trial, hx[i], hy[i], hz[i]);
      newcost += rr * rr + rz * rz;
    }

    // Take the step only if it helped, and let that decide which way lambda
    // moves. This is the whole of Levenberg-Marquardt.
    if (newcost < cost) {
      p = trial;
      lambda /= 10;
      if (cost - newcost < 1e-12) {
        cost = newcost;
        printf("converged after %d steps, cost %.6f\n", step, cost);
        break;
      }
      cost = newcost;
    } else {
      lambda *= 10;
    }
  }

  printf("  cx  = %+.4f   true %+.4f\n", p.cx, truth.cx);
  printf("  cy  = %+.4f   true %+.4f\n", p.cy, truth.cy);
  printf("  r   = %+.4f   true %+.4f\n", p.r, truth.r);
  printf("  z0  = %+.4f   true %+.4f\n", p.z0, truth.z0);
  printf("  lam = %+.4f   true %+.4f\n", p.lam, truth.lam);

  return 0;
}
