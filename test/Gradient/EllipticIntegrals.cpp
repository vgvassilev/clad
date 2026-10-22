// REQUIRES: std-special-math
// RUN: %cladclang %s -I%S/../../include -oEllipticIntegrals.out 2>&1 | %filecheck %s
// RUN: ./EllipticIntegrals.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oEllipticIntegrals.out
// RUN: ./EllipticIntegrals.out | %filecheck_exec %s

// %cladclang compiles with -DCLAD_NO_NUM_DIFF, so a numerical-differentiation
// fallback would emit a warning and fail %filecheck: every derivative below is
// forced through the builtin pushforwards and pullbacks.
//
// The expected values are mpmath finite differences of the integrals
// themselves, independent of the closed forms in BuiltinDerivatives.h.
// mpmath takes the parameter m = k^2 where std:: takes the modulus k:
//   from mpmath import mp, mpf, diff, ellipk, ellipe, ellipf, ellippi
//   mp.dps = 40
//   K   = lambda k: ellipk(k**2)                    # std::comp_ellint_1(k)
//   E   = lambda k: ellipe(k**2)                    # std::comp_ellint_2(k)
//   PiC = lambda k, nu: ellippi(nu, k**2)           # std::comp_ellint_3(k, nu)
//   Fi  = lambda k, phi: ellipf(phi, k**2)          # std::ellint_1(k, phi)
//   Ei  = lambda k, phi: ellipe(phi, k**2)          # std::ellint_2(k, phi)
//   Pi  = lambda k, nu, phi: ellippi(nu, phi, k**2) # std::ellint_3(k, nu, phi)
//   print(diff(lambda nu: PiC(mpf('0.7'), nu), mpf('0.25')))
//   print(diff(lambda nu: Pi(mpf('0.5'), nu, mpf('1.0')), 1))

#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cstdio>

// Complete elliptic integrals are single-argument real functions: reverse
// mode differentiates them through their pushforwards.
double fKE(double k) { return std::comp_ellint_1(k) * std::comp_ellint_2(k); }

// CHECK: void fKE_grad(double k, double *_d_k) {
// CHECK: _r0 += 1 * _t0 * clad::custom_derivatives::std::comp_ellint_1_pushforward(k, 1.).pushforward;
// CHECK: _r1 += _t1 * 1 * clad::custom_derivatives::std::comp_ellint_2_pushforward(k, 1.).pushforward;
// CHECK: }

// Complete elliptic integral of the third kind Pi(nu, k).
double fPiC(double k, double nu) { return std::comp_ellint_3(k, nu); }
float fPiC_f(float k, float nu) { return std::comp_ellint_3(k, nu); }
long double fPiC_l(long double k, long double nu) {
  return std::comp_ellint_3(k, nu);
}

// Incomplete elliptic integral of the first kind F(phi, k); std:: takes the
// modulus k first.
double fF(double k, double phi) { return std::ellint_1(k, phi); }

// CHECK: void fF_grad(double k, double phi, double *_d_k, double *_d_phi) {
// CHECK:     clad::custom_derivatives::std::ellint_1_pullback(k, phi, 1, &_r0, &_r1);

// Incomplete elliptic integral of the second kind E(phi, k).
double fE(double k, double phi) { return std::ellint_2(k, phi); }

// CHECK: void fE_grad(double k, double phi, double *_d_k, double *_d_phi) {
// CHECK:     clad::custom_derivatives::std::ellint_2_pullback(k, phi, 1, &_r0, &_r1);

// Incomplete elliptic integral of the third kind Pi(nu; phi, k).
double fPi(double k, double nu, double phi) { return std::ellint_3(k, nu, phi); }

// CHECK: void fPi_grad(double k, double nu, double phi, double *_d_k, double *_d_nu, double *_d_phi) {
// CHECK:     clad::custom_derivatives::std::ellint_3_pullback(k, nu, phi, 1, &_r0, &_r1, &_r2);

// The pullbacks must compute in the precision of the primal, like the
// pushforwards do, so forward and reverse mode agree bitwise in every width.
float fF_f(float k, float phi) { return std::ellint_1(k, phi); }
float fE_f(float k, float phi) { return std::ellint_2(k, phi); }
float fPi_f(float k, float nu, float phi) { return std::ellint_3(k, nu, phi); }
long double fF_l(long double k, long double phi) { return std::ellint_1(k, phi); }
long double fE_l(long double k, long double phi) { return std::ellint_2(k, phi); }
long double fPi_l(long double k, long double nu, long double phi) {
  return std::ellint_3(k, nu, phi);
}

// The hessian differentiates ellint_1_pushforward itself in reverse mode,
// which exercises the pullbacks from inside the pushforward's body.
// CHECK: void fF_hessian(double k, double phi, double *hessianMatrix) {
// CHECK: inline void ellint_1_pushforward_pullback(double k, double phi, double d_k, double d_phi, ValueAndPushforward<double, double> _d_y, double *_d_k, double *_d_phi, double *_d_d_k, double *_d_d_phi)

// Prints the forward-mode partials of a two-argument function and whether
// the gradient reproduces them bitwise.
template <typename T, typename D1, typename D2, typename G>
void check2(const char* name, int prec, D1& d1, D2& d2, G& g, T a, T b) {
  T fwd[2] = {d1.execute(a, b), d2.execute(a, b)};
  T rev[2] = {0, 0};
  g.execute(a, b, &rev[0], &rev[1]);
  printf("%s(%g, %g): {%.*Lf, %.*Lf} rev==fwd: %d\n", name, (double)a,
         (double)b, prec, (long double)fwd[0], prec, (long double)fwd[1],
         fwd[0] == rev[0] && fwd[1] == rev[1]);
}

// Same for the three-argument ellint_3.
template <typename T, typename D1, typename D2, typename D3, typename G>
void check3(const char* name, int prec, D1& d1, D2& d2, D3& d3, G& g, T a,
            T b, T c) {
  T fwd[3] = {d1.execute(a, b, c), d2.execute(a, b, c), d3.execute(a, b, c)};
  T rev[3] = {0, 0, 0};
  g.execute(a, b, c, &rev[0], &rev[1], &rev[2]);
  printf("%s(%g, %g, %g): {%.*Lf, %.*Lf, %.*Lf} rev==fwd: %d\n", name,
         (double)a, (double)b, (double)c, prec, (long double)fwd[0], prec,
         (long double)fwd[1], prec, (long double)fwd[2],
         fwd[0] == rev[0] && fwd[1] == rev[1] && fwd[2] == rev[2]);
}

int main() {
  auto dKE = clad::gradient(fKE);
  double dk = 0;
  dKE.execute(0.6, &dk);
  printf("fKE: {%.8f}\n", dk);
  // CHECK-EXEC: fKE: {0.12831154}

  auto dPiC = clad::gradient(fPiC);
  auto dPiC_k = clad::differentiate(fPiC, "k");
  auto dPiC_nu = clad::differentiate(fPiC, "nu");
  check2("fPiC", 8, dPiC_k, dPiC_nu, dPiC, 0.7, 0.25);
  // CHECK-EXEC: fPiC(0.7, 0.25): {1.46152812, 1.55847670} rev==fwd: 1

  // nu = 0 is a removable singularity of the closed form of dPi/dnu, whose
  // limit is (K - E) / k^2, or pi/4 at k = 0.
  check2("fPiC", 8, dPiC_k, dPiC_nu, dPiC, 0.7, 0.0);
  // CHECK-EXEC: fPiC(0.7, 0): {1.16066442, 1.00006707} rev==fwd: 1
  check2("fPiC", 8, dPiC_k, dPiC_nu, dPiC, 0.0, 0.0);
  // CHECK-EXEC: fPiC(0, 0): {0.00000000, 0.78539816} rev==fwd: 1

  // The pullback must compute in the precision of the primal, like the
  // pushforward does, so forward and reverse mode agree bitwise.
  auto dPiC_f = clad::gradient(fPiC_f);
  auto dPiC_f_k = clad::differentiate(fPiC_f, "k");
  auto dPiC_f_nu = clad::differentiate(fPiC_f, "nu");
  check2("fPiC_f", 5, dPiC_f_k, dPiC_f_nu, dPiC_f, 0.6f, 0.5f);
  // CHECK-EXEC: fPiC_f(0.6, 0.5): {1.32062, 2.66153} rev==fwd: 1
  auto dPiC_l = clad::gradient(fPiC_l);
  auto dPiC_l_k = clad::differentiate(fPiC_l, "k");
  auto dPiC_l_nu = clad::differentiate(fPiC_l, "nu");
  check2("fPiC_l", 10, dPiC_l_k, dPiC_l_nu, dPiC_l, 0.6L, 0.5L);
  // CHECK-EXEC: fPiC_l(0.6, 0.5): {1.3206231627, 2.6615298127} rev==fwd: 1

  auto dF = clad::gradient(fF);
  auto dF_k = clad::differentiate(fF, "k");
  auto dF_phi = clad::differentiate(fF, "phi");
  check2("fF", 8, dF_k, dF_phi, dF, 0.7, 1.0);
  // CHECK-EXEC: fF(0.7, 1): {0.28518953, 1.23745316} rev==fwd: 1
  // The k-derivatives have the limit 0 at k = 0.
  check2("fF", 8, dF_k, dF_phi, dF, 0.0, 1.0);
  // CHECK-EXEC: fF(0, 1): {0.00000000, 1.00000000} rev==fwd: 1

  auto dE = clad::gradient(fE);
  auto dE_k = clad::differentiate(fE, "k");
  auto dE_phi = clad::differentiate(fE, "phi");
  check2("fE", 8, dE_k, dE_phi, dE, 0.9, 1.3);
  // CHECK-EXEC: fE(0.9, 1.3): {-0.71068153, 0.49795588} rev==fwd: 1

  auto dPi = clad::gradient(fPi);
  auto dPi_k = clad::differentiate(fPi, "k");
  auto dPi_nu = clad::differentiate(fPi, "nu");
  auto dPi_phi = clad::differentiate(fPi, "phi");
  check3("fPi", 8, dPi_k, dPi_nu, dPi_phi, dPi, 0.7, 0.25, 1.1);
  // CHECK-EXEC: fPi(0.7, 0.25, 1.1): {0.45467245, 0.54454903, 1.59652152} rev==fwd: 1
  check3("fPi", 8, dPi_k, dPi_nu, dPi_phi, dPi, -0.6, -0.4, 2.0);
  // CHECK-EXEC: fPi(-0.6, -0.4, 2): {-0.92013894, 0.81438477, 0.89667587} rev==fwd: 1

  // Pi is smooth at nu = 0, nu = 1 (unless phi reaches pi/2) and nu = k^2,
  // where the closed forms divide by zero and the limits take over.
  check3("fPi", 8, dPi_k, dPi_nu, dPi_phi, dPi, 0.7, 0.0, 1.1);
  // CHECK-EXEC: fPi(0.7, 0, 1.1): {0.39025448, 0.40545495, 1.27951200} rev==fwd: 1
  check3("fPi", 8, dPi_k, dPi_nu, dPi_phi, dPi, 0.0, 0.0, 1.1);
  // CHECK-EXEC: fPi(0, 0, 1.1): {0.00000000, 0.34787590, 1.00000000} rev==fwd: 1
  check3("fPi", 8, dPi_k, dPi_nu, dPi_phi, dPi, 0.5, 1.0, 1.0);
  // CHECK-EXEC: fPi(0.5, 1, 1): {0.34347542, 1.35812145, 3.77599389} rev==fwd: 1
  check3("fPi", 8, dPi_k, dPi_nu, dPi_phi, dPi, 0.5, 0.25, 1.1);
  // CHECK-EXEC: fPi(0.5, 0.25, 1.1): {0.24975178, 0.49950356, 1.39378448} rev==fwd: 1

  auto dF_f = clad::gradient(fF_f);
  auto dF_f_k = clad::differentiate(fF_f, "k");
  auto dF_f_phi = clad::differentiate(fF_f, "phi");
  check2("fF_f", 6, dF_f_k, dF_f_phi, dF_f, 0.3f, 0.8f);
  // CHECK-EXEC: fF_f(0.3, 0.8): {0.047074, 1.023994} rev==fwd: 1

  auto dE_f = clad::gradient(fE_f);
  auto dE_f_k = clad::differentiate(fE_f, "k");
  auto dE_f_phi = clad::differentiate(fE_f, "phi");
  check2("fE_f", 6, dE_f_k, dE_f_phi, dE_f, 0.3f, 0.8f);
  // CHECK-EXEC: fE_f(0.3, 0.8): {-0.045701, 0.976568} rev==fwd: 1

  auto dPi_f = clad::gradient(fPi_f);
  auto dPi_f_k = clad::differentiate(fPi_f, "k");
  auto dPi_f_nu = clad::differentiate(fPi_f, "nu");
  auto dPi_f_phi = clad::differentiate(fPi_f, "phi");
  check3("fPi_f", 6, dPi_f_k, dPi_f_nu, dPi_f_phi, dPi_f, 0.3f, 0.5f, 0.8f);
  // CHECK-EXEC: fPi_f(0.3, 0.5, 0.8): {0.056518, 0.220480, 1.378745} rev==fwd: 1
  check3("fPi_f", 6, dPi_f_k, dPi_f_nu, dPi_f_phi, dPi_f, 0.5f, 0.25f, 1.1f);
  // CHECK-EXEC: fPi_f(0.5, 0.25, 1.1): {0.249752, 0.499504, 1.393784} rev==fwd: 1

  auto dF_l = clad::gradient(fF_l);
  auto dF_l_k = clad::differentiate(fF_l, "k");
  auto dF_l_phi = clad::differentiate(fF_l, "phi");
  check2("fF_l", 10, dF_l_k, dF_l_phi, dF_l, 0.3L, 0.8L);
  // CHECK-EXEC: fF_l(0.3, 0.8): {0.0470739318, 1.0239937159} rev==fwd: 1

  auto dE_l = clad::gradient(fE_l);
  auto dE_l_k = clad::differentiate(fE_l, "k");
  auto dE_l_phi = clad::differentiate(fE_l, "phi");
  check2("fE_l", 10, dE_l_k, dE_l_phi, dE_l, 0.3L, 0.8L);
  // CHECK-EXEC: fE_l(0.3, 0.8): {-0.0457008788, 0.9765684930} rev==fwd: 1

  auto dPi_l = clad::gradient(fPi_l);
  auto dPi_l_k = clad::differentiate(fPi_l, "k");
  auto dPi_l_nu = clad::differentiate(fPi_l, "nu");
  auto dPi_l_phi = clad::differentiate(fPi_l, "phi");
  check3("fPi_l", 10, dPi_l_k, dPi_l_nu, dPi_l_phi, dPi_l, 0.3L, 0.5L, 0.8L);
  // CHECK-EXEC: fPi_l(0.3, 0.5, 0.8): {0.0565183237, 0.2204801724, 1.3787445150} rev==fwd: 1
  check3("fPi_l", 10, dPi_l_k, dPi_l_nu, dPi_l_phi, dPi_l, 0.5L, 1.0L, 1.0L);
  // CHECK-EXEC: fPi_l(0.5, 1, 1): {0.3434754195, 1.3581214469, 3.7759938934} rev==fwd: 1

  // Second order: hessian of F(phi, k) in (k, phi) at (0.7, 1.0).
  auto hF = clad::hessian(fF);
  double hess[4] = {0, 0, 0, 0};
  hF.execute(0.7, 1.0, hess);
  printf("fF hessian: {%.6f, %.6f, %.6f, %.6f}\n", hess[0], hess[1], hess[2],
         hess[3]);
  // CHECK-EXEC: fF hessian: {0.806157, 0.939210, 0.939210, 0.422142}

  return 0;
}
