// The RooFit math functions and clad custom derivatives used by the
// HistFactory-style likelihood benchmark that histfactory.py generates.
//
// Extracted from a standalone version of a real ATLAS RooFit likelihood:
// the subset of RooFit::Detail::MathFuncs the generated model code calls,
// with TMath and ROOT::Math calls replaced by their std:: equivalents; the
// incomplete gamma function (needed by poissonIntegral) is the Cephes
// implementation that ROOT's ROOT::Math::inc_gamma wraps, and its custom
// clad derivatives are copied verbatim from ROOT's Math/CladDerivator.h.
//
// `constraintSum` is NOT here but in the generated file: like in ROOT it
// enables clad's `#pragma clad checkpoint loop` on its loop (its pullback
// needs nothing from previous iterations, so checkpointing replaces the
// tape at zero cost), and as of 2026-08 clad mis-attributes a checkpoint
// pragma that lives in an included header -- the planner selects pragmas by
// raw SourceLocation order, which does not match translation-unit order
// across files -- so the pragma'd function must sit in the main file.
// `flexibleInterp` does NOT take the pragma: its loop carries `total` into
// the next iteration's call, so taping it beats recomputing.

// The llvm-header-guard check derives the guard name from the checkout path,
// which no fixed name can satisfy.
#ifndef CLAD_BENCHMARK_HISTFACTORY_MATHFUNCS_H // NOLINT(llvm-header-guard)
#define CLAD_BENCHMARK_HISTFACTORY_MATHFUNCS_H

// This file keeps the formatting and the code style of the ROOT sources it
// was extracted from, so it stays diffable against RooFit/Detail/MathFuncs.h,
// Math/CladDerivator.h and the Cephes code in mathcore. The markers below
// shield the copied code from clang-format and from the clang-tidy checks it
// predates; the dead stores are the hand-derived pullbacks' unused restores.
// clang-format off

#include "clad/Differentiator/BuiltinDerivatives.h"
#include "clad/Differentiator/Differentiator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

// The copied code below stays verbatim, so it neither follows this project's
// clang-tidy configuration nor is warning-clean: the ROOT pullbacks snapshot
// not-yet-initialized values on purpose (their restores are dead), which
// trips the uninitialized-use compiler diagnostics that clang-tidy surfaces.
// NOLINTBEGIN

namespace ROOT {
namespace Math {

namespace Cephes {

// Incomplete gamma function, from ROOT's Cephes copy
// (math/mathcore/src/SpecFuncCephes.cxx), with lgam() replaced by
// std::lgamma().

constexpr double kMACHEP = 1.11022302462515654042363166809e-16;
constexpr double kMAXLOG = 709.782712893383973096206318587;
constexpr double kBig = 4.503599627370496e15;
constexpr double kBiginv = 2.22044604925031308085e-16;

inline double igamc(double a, double x);

inline double igam(double a, double x)
{
   double ans, ax, c, r;

   // LM: for negative values returns 1.0 instead of zero
   // This is correct if a is a negative integer since Gamma(-n) = +/- inf
   if (a <= 0)
      return 1.0;

   if (x <= 0)
      return 0.0;

   if ((x > 1.0) && (x > a))
      return 1.0 - igamc(a, x);

   /* Compute  x**a * exp(-x) / gamma(a)  */
   ax = a * std::log(x) - x - std::lgamma(a);
   if (ax < -kMAXLOG)
      return 0.0;

   ax = std::exp(ax);

   /* power series */
   r = a;
   c = 1.0;
   ans = 1.0;

   do {
      r += 1.0;
      c *= x / r;
      ans += c;
   } while (c / ans > kMACHEP);

   return ans * ax / a;
}

inline double igamc(double a, double x)
{
   double ans, ax, c, yc, r, t, y, z;
   double pk, pkm1, pkm2, qk, qkm1, qkm2;

   // LM: for negative values returns 0.0
   // This is correct if a is a negative integer since Gamma(-n) = +/- inf
   if (a <= 0)
      return 0.0;

   if (x <= 0)
      return 1.0;

   if ((x < 1.0) || (x < a))
      return 1.0 - igam(a, x);

   ax = a * std::log(x) - x - std::lgamma(a);
   if (ax < -kMAXLOG)
      return 0.0;

   ax = std::exp(ax);

   /* continued fraction */
   y = 1.0 - a;
   z = x + y + 1.0;
   c = 0.0;
   pkm2 = 1.0;
   qkm2 = x;
   pkm1 = x + 1.0;
   qkm1 = z * x;
   ans = pkm1 / qkm1;

   do {
      c += 1.0;
      y += 1.0;
      z += 2.0;
      yc = y * c;
      pk = pkm1 * z - pkm2 * yc;
      qk = qkm1 * z - qkm2 * yc;
      if (qk) {
         r = pk / qk;
         t = std::abs((ans - r) / r);
         ans = r;
      } else
         t = 1.0;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if (std::abs(pk) > kBig) {
         pkm2 *= kBiginv;
         pkm1 *= kBiginv;
         qkm2 *= kBiginv;
         qkm1 *= kBiginv;
      }
   } while (t > kMACHEP);

   return ans * ax;
}

} // namespace Cephes

inline double inc_gamma(double a, double x)
{
   return Cephes::igam(a, x);
}

inline double inc_gamma_c(double a, double x)
{
   return Cephes::igamc(a, x);
}

} // namespace Math
} // namespace ROOT

// The subset of RooFit::Detail::MathFuncs (from
// roofit/roofitcore/inc/RooFit/Detail/MathFuncs.h) used by the generated
// model code below. TMath calls are replaced by std:: equivalents:
// TMath::LnGamma -> std::lgamma, TMath::QuietNaN ->
// std::numeric_limits::quiet_NaN, TMath::Limits<double>::Min() ->
// std::numeric_limits::min, TMath::TwoPi / TMath::Sqrt2 -> literals.

namespace RooFit {
namespace Detail {
namespace MathFuncs {

/// @brief Function to evaluate an un-normalized RooGaussian.
inline double gaussian(double x, double mean, double sigma)
{
   const double arg = x - mean;
   const double sig = sigma;
   return std::exp(-0.5 * arg * arg / (sig * sig));
}

template <typename DoubleArray>
double product(DoubleArray factors, std::size_t nFactors)
{
   double out = 1.0;
   for (std::size_t i = 0; i < nFactors; ++i) {
      out *= factors[i];
   }
   return out;
}

// constraintSum lives in the generated main file; see the header comment.

inline unsigned int uniformBinNumber(double low, double high, double val, unsigned int numBins, double coef)
{
   double binWidth = (high - low) / numBins;
   return coef * (val >= high ? numBins - 1 : std::abs((val - low) / binWidth));
}

inline double poisson(double x, double par)
{
   if (par < 0)
      return std::numeric_limits<double>::quiet_NaN();

   if (x < 0) {
      return 0;
   } else if (x == 0.0) {
      return std::exp(-par);
   } else {
      double out = x * std::log(par) - std::lgamma(x + 1.) - par;
      return std::exp(out);
   }
}

inline double flexibleInterpSingle(unsigned int code, double low, double high, double boundary, double nominal,
                                   double paramVal, double res)
{
   if (code == 0) {
      // piece-wise linear
      if (paramVal > 0) {
         return paramVal * (high - nominal);
      } else {
         return paramVal * (nominal - low);
      }
   } else if (code == 1) {
      // piece-wise log
      if (paramVal >= 0) {
         return res * (std::pow(high / nominal, +paramVal) - 1);
      } else {
         return res * (std::pow(low / nominal, -paramVal) - 1);
      }
   } else if (code == 2) {
      // parabolic with linear
      double a = 0.5 * (high + low) - nominal;
      double b = 0.5 * (high - low);
      double c = 0;
      if (paramVal > 1) {
         return (2 * a + b) * (paramVal - 1) + high - nominal;
      } else if (paramVal < -1) {
         return -1 * (2 * a - b) * (paramVal + 1) + low - nominal;
      } else {
         return a * paramVal * paramVal + b * paramVal + c;
      }
   } else if (code == 4 || code == 6) {
      double x = paramVal;
      double mod = 1.0;
      if (code == 6) {
         high /= nominal;
         low /= nominal;
         nominal = 1;
      }
      if (x >= boundary) {
         mod = x * (high - nominal);
      } else if (x <= -boundary) {
         mod = x * (nominal - low);
      } else {
         // interpolate 6th degree
         double t = x / boundary;
         double eps_plus = high - nominal;
         double eps_minus = nominal - low;
         double S = 0.5 * (eps_plus + eps_minus);
         double A = 0.0625 * (eps_plus - eps_minus);

         mod = x * (S + t * A * (15 + t * t * (-10 + t * t * 3)));
      }

      // code 6 is multiplicative version of code 4
      if (code == 6) {
         mod *= res;
      }
      return mod;

   } else if (code == 5) {
      double x = paramVal;
      double mod = 1.0;
      if (x >= boundary) {
         mod = std::pow(high / nominal, +paramVal);
      } else if (x <= -boundary) {
         mod = std::pow(low / nominal, -paramVal);
      } else {
         // interpolate 6th degree exp
         double x0 = boundary;

         high /= nominal;
         low /= nominal;

         // GHL: Swagato's suggestions
         double logHi = std::log(high);
         double logLo = std::log(low);
         double powUp = std::exp(x0 * logHi);
         double powDown = std::exp(x0 * logLo);
         double powUpLog = high <= 0.0 ? 0.0 : powUp * logHi;
         double powDownLog = low <= 0.0 ? 0.0 : -powDown * logLo;
         double powUpLog2 = high <= 0.0 ? 0.0 : powUpLog * logHi;
         double powDownLog2 = low <= 0.0 ? 0.0 : -powDownLog * logLo;

         double S0 = 0.5 * (powUp + powDown);
         double A0 = 0.5 * (powUp - powDown);
         double S1 = 0.5 * (powUpLog + powDownLog);
         double A1 = 0.5 * (powUpLog - powDownLog);
         double S2 = 0.5 * (powUpLog2 + powDownLog2);
         double A2 = 0.5 * (powUpLog2 - powDownLog2);

         // fcns+der+2nd_der are eq at bd

         double x0Sq = x0 * x0;

         double a = 1. / (8 * x0) * (15 * A0 - 7 * x0 * S1 + x0 * x0 * A2);
         double b = 1. / (8 * x0Sq) * (-24 + 24 * S0 - 9 * x0 * A1 + x0 * x0 * S2);
         double c = 1. / (4 * x0Sq * x0) * (-5 * A0 + 5 * x0 * S1 - x0 * x0 * A2);
         double d = 1. / (4 * x0Sq * x0Sq) * (12 - 12 * S0 + 7 * x0 * A1 - x0 * x0 * S2);
         double e = 1. / (8 * x0Sq * x0Sq * x0) * (+3 * A0 - 3 * x0 * S1 + x0 * x0 * A2);
         double f = 1. / (8 * x0Sq * x0Sq * x0Sq) * (-8 + 8 * S0 - 5 * x0 * A1 + x0 * x0 * S2);

         // evaluate the 6-th degree polynomial using Horner's method
         double value = 1. + x * (a + x * (b + x * (c + x * (d + x * (e + x * f)))));
         mod = value;
      }
      return res * (mod - 1.0);
   }

   return 0.0;
}

template <typename ParamsArray, typename DoubleArray>
double flexibleInterp(unsigned int code, ParamsArray params, unsigned int n, DoubleArray low, DoubleArray high,
                      double boundary, double nominal, int doCutoff)
{
   double total = nominal;
   // No checkpoint pragma here (see the header comment): the loop carries
   // `total` into the next iteration's call, so taping it beats recomputing.
   for (std::size_t i = 0; i < n; ++i) {
      total += flexibleInterpSingle(code, low[i], high[i], boundary, nominal, params[i], total);
   }

   return doCutoff && total <= 0 ? std::numeric_limits<double>::min() : total;
}

inline double nll(double pdf, double weight, int binnedL, int doBinOffset)
{
   if (binnedL) {
      // Special handling of this case since std::log(Poisson(0,0)=0 but can't be
      // calculated with usual log-formula since std::log(mu)=0. No update of result
      // is required since term=0.
      if (std::abs(pdf) < 1e-10 && std::abs(weight) < 1e-10) {
         return 0.0;
      }
      if (doBinOffset) {
         return pdf - weight - weight * (std::log(pdf) - std::log(weight));
      }
      return pdf - weight * std::log(pdf) + std::lgamma(weight + 1);
   } else {
      return -weight * std::log(pdf);
   }
}

/// @brief Function to calculate the integral of an un-normalized RooGaussian over x. To calculate the integral over
/// mean, just interchange the respective values of x and mean.
inline double gaussianIntegral(double xMin, double xMax, double mean, double sigma)
{
   // The normalisation constant 1./sqrt(2*pi*sigma^2) is left out in evaluate().
   // Therefore, the integral is scaled up by that amount to make RooFit normalise
   // correctly.
   const double sqrtTwoPi = 2.50662827463100050241576528481; // sqrt(2*pi)
   const double sqrtTwo = 1.41421356237309514547462185874;   // sqrt(2)
   double resultScale = 0.5 * sqrtTwoPi * sigma;

   // Here everything is scaled and shifted into a standard normal distribution:
   double xscale = sqrtTwo * sigma;
   double scaledMin = 0.;
   double scaledMax = 0.;
   scaledMin = (xMin - mean) / xscale;
   scaledMax = (xMax - mean) / xscale;

   // Here we go for maximum precision: We compute all integrals in the UPPER
   // tail of the Gaussian, because erfc has the highest precision there.
   // Therefore, the different cases for range limits in the negative hemisphere are mapped onto
   // the equivalent points in the upper hemisphere using erfc(-x) = 2. - erfc(x)
   double ecmin = std::erfc(std::abs(scaledMin));
   double ecmax = std::erfc(std::abs(scaledMax));

   double cond = 0.0;
   // Don't put this "prd" inside the "if" because clad will not be able to differentiate the code correctly (as of
   // v1.1)!
   double prd = scaledMin * scaledMax;
   if (prd < 0.0) {
      cond = 2.0 - (ecmin + ecmax);
   } else if (scaledMax <= 0.0) {
      cond = ecmax - ecmin;
   } else {
      cond = ecmin - ecmax;
   }
   return resultScale * cond;
}

// The last param should be of type bool but it is not as that causes some issues with Cling for some reason...
inline double
poissonIntegral(int code, double mu, double x, double integrandMin, double integrandMax, unsigned int protectNegative)
{
   if (protectNegative && mu < 0.0) {
      return std::exp(-2.0 * mu); // make it fall quickly
   }

   if (code == 1) {
      // Implement integral over x as summation. Add special handling in case
      // range boundaries are not on integer values of x
      integrandMin = std::max(0., integrandMin);

      if (integrandMax < 0. || integrandMax < integrandMin) {
         return 0;
      }
      const double delta = 100.0 * std::sqrt(mu);
      // If the limits are more than many standard deviations away from the mean,
      // we might as well return the integral of the full Poisson distribution to
      // save computing time.
      if (integrandMin < std::max(mu - delta, 0.0) && integrandMax > mu + delta) {
         return 1.;
      }

      // The range as integers. ixMin is included, ixMax outside.
      const unsigned int ixMin = integrandMin;
      const unsigned int ixMax = std::min(integrandMax + 1, (double)std::numeric_limits<unsigned int>::max());

      // Sum from 0 to just before the bin outside of the range.
      if (ixMin == 0) {
         return ROOT::Math::inc_gamma_c(ixMax, mu);
      } else {
         // If necessary, subtract from 0 to the beginning of the range
         if (ixMin <= mu) {
            return ROOT::Math::inc_gamma_c(ixMax, mu) - ROOT::Math::inc_gamma_c(ixMin, mu);
         } else {
            // Avoid catastrophic cancellation in the high tails:
            return ROOT::Math::inc_gamma(ixMin, mu) - ROOT::Math::inc_gamma(ixMax, mu);
         }
      }
   }

   // the integral with respect to the mean is the integral of a gamma distribution
   // negative ix does not need protection (gamma returns 0.0)
   const double ix = 1 + x;

   return ROOT::Math::inc_gamma(ix, integrandMax) - ROOT::Math::inc_gamma(ix, integrandMin);
}

} // namespace MathFuncs
} // namespace Detail
} // namespace RooFit

// Custom clad derivatives for the incomplete gamma function, copied verbatim
// from ROOT's math/mathcore/inc/Math/CladDerivator.h. The pushforwards make
// forward mode work; the pullbacks additionally enable second derivatives
// (clad::hessian differentiates the pushforward in reverse mode).

namespace clad {
namespace custom_derivatives {
namespace ROOT {
namespace Math {

inline void inc_gamma_c_pullback(double a, double x, double _d_y, double *_d_a, double *_d_x);

inline void inc_gamma_pullback(double a, double x, double _d_y, double *_d_a, double *_d_x)
{
   // Synced with SpecFuncCephes.h
   constexpr double kMACHEP = 1.11022302462515654042363166809e-16;
   constexpr double kMAXLOG = 709.782712893383973096206318587;

   double _d_ans = 0, _d_ax = 0, _d_c = 0, _d_r = 0;
   double _t1;
   double _t2;
   double _t3;
   double _t4;
   double _t5;
   clad::tape<double> _t7 = {};
   clad::tape<double> _t8 = {};
   clad::tape<double> _t9 = {};
   double ans, ax, c, r;
   if (a <= 0)
      return;
   if (x <= 0)
      return;
   if ((x > 1.) && (x > a)) {
      double _r0 = 0;
      double _r1 = 0;
      inc_gamma_c_pullback(a, x, -_d_y, &_r0, &_r1);
      *_d_a += _r0;
      *_d_x += _r1;
      return;
   }
   _t1 = ::std::log(x);
   ax = a * _t1 - x - ::std::lgamma(a);
   if (ax < -kMAXLOG) {
      *_d_x += (a * _d_ax / x) - _d_ax;
      *_d_a += _d_ax * (_t1 - ::clad::custom_derivatives::std::clad_digamma(
                                 a)); // numerical_diff::forward_central_difference(::std::lgamma, a, 0, 0, a);
      _d_ax = 0.;
      return;
   }
   _t2 = ax;
   ax = ::std::exp(ax);
   _t3 = r;
   r = a;
   _t4 = c;
   c = 1.;
   _t5 = ans;
   ans = 1.;
   unsigned long _t6 = 0;
   do {
      _t6++;
      clad::push(_t7, r);
      r += 1.;
      clad::push(_t8, c);
      c *= x / r;
      clad::push(_t9, ans);
      ans += c;
   } while (c / ans > kMACHEP);
   {
      _d_ans += _d_y / a * ax;
      _d_ax += ans * _d_y / a;
      double _r6 = _d_y * -(ans * ax / (a * a));
      *_d_a += _r6;
   }
   do {
      {
         {
            ans = clad::pop(_t9);
            double _r_d7 = _d_ans;
            _d_c += _r_d7;
         }
         {
            c = clad::pop(_t8);
            double _r_d6 = _d_c;
            _d_c -= _r_d6;
            _d_c += _r_d6 * x / r;
            *_d_x += c * _r_d6 / r;
            double _r5 = c * _r_d6 * -(x / (r * r));
            _d_r += _r5;
         }
         {
            r = clad::pop(_t7);
            double _r_d5 = _d_r;
         }
      }
      _t6--;
   } while (_t6);
   {
      ans = _t5;
      double _r_d4 = _d_ans;
      _d_ans -= _r_d4;
   }
   {
      c = _t4;
      double _r_d3 = _d_c;
      _d_c -= _r_d3;
   }
   {
      r = _t3;
      double _r_d2 = _d_r;
      _d_r -= _r_d2;
      *_d_a += _r_d2;
   }
   {
      ax = _t2;
      double _r_d1 = _d_ax;
      _d_ax -= _r_d1;
      double _r4 = 0;
      _r4 += _r_d1 * ::std::exp(ax);
      _d_ax += _r4;
   }
   {
      *_d_x += (a * _d_ax / x) - _d_ax;
      *_d_a += _d_ax * (_t1 - ::clad::custom_derivatives::std::clad_digamma(
                                 a)); // numerical_diff::forward_central_difference(::std::lgamma, a, 0, 0, a);
      _d_ax = 0.;
   }
}

inline void inc_gamma_c_pullback(double a, double x, double _d_y, double *_d_a, double *_d_x)
{
   // Synced with SpecFuncCephes.h
   constexpr double kMACHEP = 1.11022302462515654042363166809e-16;
   constexpr double kMAXLOG = 709.782712893383973096206318587;
   constexpr double kBig = 4.503599627370496e15;
   constexpr double kBiginv = 2.22044604925031308085e-16;

   double _d_ans = 0, _d_ax = 0, _d_c = 0, _d_yc = 0, _d_r = 0, _d_y0 = 0, _d_z = 0;
   double _d_pk = 0, _d_pkm1 = 0, _d_pkm2 = 0, _d_qk = 0, _d_qkm1 = 0, _d_qkm2 = 0;
   double _t1;
   double _t2;
   double _t3;
   double _t4;
   double _t5;
   double _t6;
   double _t7;
   double _t8;
   double _t9;
   double _t10;
   unsigned long _t11;
   clad::tape<double> _t12 = {};
   clad::tape<double> _t13 = {};
   clad::tape<double> _t14 = {};
   clad::tape<double> _t15 = {};
   clad::tape<double> _t16 = {};
   clad::tape<double> _t17 = {};
   clad::tape<double> _t19 = {};
   clad::tape<double> _t20 = {};
   clad::tape<double> _t22 = {};
   clad::tape<double> _t24 = {};
   clad::tape<double> _t25 = {};
   clad::tape<double> _t26 = {};
   clad::tape<double> _t27 = {};
   clad::tape<bool> _t29 = {};
   clad::tape<double> _t30 = {};
   clad::tape<double> _t31 = {};
   clad::tape<double> _t32 = {};
   clad::tape<double> _t33 = {};
   double ans, ax, c, yc, r, t, y, z;
   double pk, pkm1, pkm2, qk, qkm1, qkm2;
   if (a <= 0)
      return;
   if (x <= 0)
      return;
   if ((x < 1.) || (x < a)) {
      double _r0 = 0;
      double _r1 = 0;
      inc_gamma_pullback(a, x, -_d_y, &_r0, &_r1);
      *_d_a += _r0;
      *_d_x += _r1;
      return;
   }
   _t1 = ::std::log(x);
   ax = a * _t1 - x - ::std::lgamma(a);
   if (ax < -kMAXLOG) {
      *_d_x += a * _d_ax / x - _d_ax;
      *_d_a += _d_ax * (_t1 - ::clad::custom_derivatives::std::clad_digamma(
                                 a)); // numerical_diff::forward_central_difference(::std::lgamma, a, 0, 0, a);
      _d_ax = 0.;
      return;
   }
   _t2 = ax;
   ax = ::std::exp(ax);
   _t3 = y;
   y = 1. - a;
   _t4 = z;
   z = x + y + 1.;
   _t5 = c;
   c = 0.;
   _t6 = pkm2;
   pkm2 = 1.;
   _t7 = qkm2;
   qkm2 = x;
   _t8 = pkm1;
   pkm1 = x + 1.;
   _t9 = qkm1;
   qkm1 = z * x;
   _t10 = ans;
   ans = pkm1 / qkm1;
   _t11 = 0;
   do {
      _t11++;
      clad::push(_t12, c);
      c += 1.;
      clad::push(_t13, y);
      y += 1.;
      clad::push(_t14, z);
      z += 2.;
      clad::push(_t15, yc);
      yc = y * c;
      clad::push(_t16, pk);
      pk = pkm1 * z - pkm2 * yc;
      clad::push(_t17, qk);
      qk = qkm1 * z - qkm2 * yc;
      double _t18 = qk;
      {
         if (_t18) {
            clad::push(_t20, r);
            r = pk / qk;
            t = ::std::abs((ans - r) / r);
            clad::push(_t22, ans);
            ans = r;
         } else {
            t = 1.;
         }
         clad::push(_t19, _t18);
      }
      clad::push(_t24, pkm2);
      pkm2 = pkm1;
      clad::push(_t25, pkm1);
      pkm1 = pk;
      clad::push(_t26, qkm2);
      qkm2 = qkm1;
      clad::push(_t27, qkm1);
      qkm1 = qk;
      bool _t28 = ::std::abs(pk) > kBig;
      {
         if (_t28) {
            clad::push(_t30, pkm2);
            pkm2 *= kBiginv;
            clad::push(_t31, pkm1);
            pkm1 *= kBiginv;
            clad::push(_t32, qkm2);
            qkm2 *= kBiginv;
            clad::push(_t33, qkm1);
            qkm1 *= kBiginv;
         }
         clad::push(_t29, _t28);
      }
   } while (t > kMACHEP);
   {
      _d_ans += _d_y * ax;
      _d_ax += ans * _d_y;
   }
   do {
      {
         if (clad::pop(_t29)) {
            {
               qkm1 = clad::pop(_t33);
               double _r_d27 = _d_qkm1;
               _d_qkm1 -= _r_d27;
               _d_qkm1 += _r_d27 * kBiginv;
            }
            {
               qkm2 = clad::pop(_t32);
               double _r_d26 = _d_qkm2;
               _d_qkm2 -= _r_d26;
               _d_qkm2 += _r_d26 * kBiginv;
            }
            {
               pkm1 = clad::pop(_t31);
               double _r_d25 = _d_pkm1;
               _d_pkm1 -= _r_d25;
               _d_pkm1 += _r_d25 * kBiginv;
            }
            {
               pkm2 = clad::pop(_t30);
               double _r_d24 = _d_pkm2;
               _d_pkm2 -= _r_d24;
               _d_pkm2 += _r_d24 * kBiginv;
            }
         }
         {
            qkm1 = clad::pop(_t27);
            double _r_d23 = _d_qkm1;
            _d_qkm1 -= _r_d23;
            _d_qk += _r_d23;
         }
         {
            qkm2 = clad::pop(_t26);
            double _r_d22 = _d_qkm2;
            _d_qkm2 -= _r_d22;
            _d_qkm1 += _r_d22;
         }
         {
            pkm1 = clad::pop(_t25);
            double _r_d21 = _d_pkm1;
            _d_pkm1 -= _r_d21;
            _d_pk += _r_d21;
         }
         {
            pkm2 = clad::pop(_t24);
            double _r_d20 = _d_pkm2;
            _d_pkm2 -= _r_d20;
            _d_pkm1 += _r_d20;
         }
         // t only controls the loop exit, so its adjoint is identically zero
         // and it needs neither a tape nor a restore.
         if (clad::pop(_t19)) {
            {
               ans = clad::pop(_t22);
               double _r_d18 = _d_ans;
               _d_ans -= _r_d18;
               _d_r += _r_d18;
            }
            {
               r = clad::pop(_t20);
               double _r_d16 = _d_r;
               _d_r -= _r_d16;
               _d_pk += _r_d16 / qk;
               double _r6 = _r_d16 * -(pk / (qk * qk));
               _d_qk += _r6;
            }
         }
         {
            qk = clad::pop(_t17);
            double _r_d15 = _d_qk;
            _d_qk -= _r_d15;
            _d_qkm1 += _r_d15 * z;
            _d_z += qkm1 * _r_d15;
            _d_qkm2 += -_r_d15 * yc;
            _d_yc += qkm2 * -_r_d15;
         }
         {
            pk = clad::pop(_t16);
            double _r_d14 = _d_pk;
            _d_pk -= _r_d14;
            _d_pkm1 += _r_d14 * z;
            _d_z += pkm1 * _r_d14;
            _d_pkm2 += -_r_d14 * yc;
            _d_yc += pkm2 * -_r_d14;
         }
         {
            yc = clad::pop(_t15);
            double _r_d13 = _d_yc;
            _d_yc -= _r_d13;
            _d_y0 += _r_d13 * c;
            _d_c += y * _r_d13;
         }
         {
            z = clad::pop(_t14);
            double _r_d12 = _d_z;
         }
         {
            y = clad::pop(_t13);
            double _r_d11 = _d_y0;
         }
         {
            c = clad::pop(_t12);
            double _r_d10 = _d_c;
         }
      }
      _t11--;
   } while (_t11);
   {
      ans = _t10;
      double _r_d9 = _d_ans;
      _d_ans -= _r_d9;
      _d_pkm1 += _r_d9 / qkm1;
      double _r5 = _r_d9 * -(pkm1 / (qkm1 * qkm1));
      _d_qkm1 += _r5;
   }
   {
      qkm1 = _t9;
      double _r_d8 = _d_qkm1;
      _d_qkm1 -= _r_d8;
      _d_z += _r_d8 * x;
      *_d_x += z * _r_d8;
   }
   {
      pkm1 = _t8;
      double _r_d7 = _d_pkm1;
      _d_pkm1 -= _r_d7;
      *_d_x += _r_d7;
   }
   {
      qkm2 = _t7;
      double _r_d6 = _d_qkm2;
      _d_qkm2 -= _r_d6;
      *_d_x += _r_d6;
   }
   {
      pkm2 = _t6;
      double _r_d5 = _d_pkm2;
      _d_pkm2 -= _r_d5;
   }
   {
      c = _t5;
      double _r_d4 = _d_c;
      _d_c -= _r_d4;
   }
   {
      z = _t4;
      double _r_d3 = _d_z;
      _d_z -= _r_d3;
      *_d_x += _r_d3;
      _d_y0 += _r_d3;
   }
   {
      y = _t3;
      double _r_d2 = _d_y0;
      _d_y0 -= _r_d2;
      *_d_a += -_r_d2;
   }
   {
      ax = _t2;
      double _r_d1 = _d_ax;
      _d_ax -= _r_d1;
      double _r4 = _r_d1 * ::std::exp(ax);
      _d_ax += _r4;
   }
   {
      *_d_x += a * _d_ax / x - _d_ax;
      *_d_a += _d_ax * (_t1 - ::clad::custom_derivatives::std::clad_digamma(
                                 a)); // numerical_diff::forward_central_difference(::std::lgamma, a, 0, 0, a);
      _d_ax = 0.;
   }
}

/// Derivative of the normalized lower incomplete gamma function P(a, x) with
/// respect to x. This is the integrand of P(a, x), i.e. the gamma
/// distribution density: x^(a-1) * exp(-x) / Gamma(a).
inline double inc_gamma_dx(double a, double x)
{
   if (a <= 0 || x <= 0)
      return 0.;
   return ::std::exp((a - 1.) * ::std::log(x) - x - ::std::lgamma(a));
}

/// Pullback of inc_gamma_dx(), using the closed forms of the second
/// derivatives of P(a, x):
///
///    d2P/dx2  = inc_gamma_dx(a, x) * ((a - 1) / x - 1)
///    d2P/dxda = inc_gamma_dx(a, x) * (log(x) - digamma(a))
inline void inc_gamma_dx_pullback(double a, double x, double _d_y, double *_d_a, double *_d_x)
{
   if (a <= 0 || x <= 0)
      return;
   const double g = inc_gamma_dx(a, x);
   *_d_a += _d_y * g * (::std::log(x) - ::clad::custom_derivatives::std::clad_digamma(a));
   *_d_x += _d_y * g * ((a - 1.) / x - 1.);
}

/// Derivative of the normalized lower incomplete gamma function P(a, x) with
/// respect to a. It has no closed form, but inc_gamma_pullback() computes it
/// exactly by differentiating through the algorithm that evaluates P(a, x).
inline double inc_gamma_da(double a, double x)
{
   double da = 0.;
   double dx = 0.;
   inc_gamma_pullback(a, x, 1., &da, &dx);
   return da;
}

/// Pullback of inc_gamma_da(). The mixed second derivative is known in
/// closed form (it is the same as the a-derivative of inc_gamma_dx(), see
/// inc_gamma_dx_pullback()). For d2P/da2 there is no closed form, so it is
/// approximated by a central difference of the exact first derivative.
///
/// For a <= h, the lower stencil point leaves the domain (inc_gamma_da()
/// returns zero for non-positive a), so d2P/da2 is unreliable there. This is
/// acceptable because RooFit never differentiates with respect to a, which is
/// data there.
inline void inc_gamma_da_pullback(double a, double x, double _d_y, double *_d_a, double *_d_x)
{
   if (a <= 0 || x <= 0)
      return;
   *_d_x += _d_y * inc_gamma_dx(a, x) * (::std::log(x) - ::clad::custom_derivatives::std::clad_digamma(a));
   // A first-order central difference of the exact derivative is much more
   // accurate than a second-order finite difference of P(a, x) itself. The
   // step size balances truncation and roundoff error (~ cbrt of the machine
   // epsilon).
   const double h = 6e-6 * ::std::max(1., ::std::abs(a));
   *_d_a += _d_y * (inc_gamma_da(a + h, x) - inc_gamma_da(a - h, x)) / (2. * h);
}

/// Pushforward of ROOT::Math::inc_gamma. Besides forward-mode differentiation,
/// this enables second derivatives (e.g. clad::hessian): clad differentiates
/// this function in reverse mode, and all derivatives it needs for that are
/// provided by custom pullbacks.
inline clad::ValueAndPushforward<double, double> inc_gamma_pushforward(double a, double x, double d_a, double d_x)
{
   return {::ROOT::Math::inc_gamma(a, x), inc_gamma_da(a, x) * d_a + inc_gamma_dx(a, x) * d_x};
}

/// Pushforward of ROOT::Math::inc_gamma_c, which is 1 - inc_gamma. See
/// inc_gamma_pushforward().
inline clad::ValueAndPushforward<double, double> inc_gamma_c_pushforward(double a, double x, double d_a, double d_x)
{
   return {::ROOT::Math::inc_gamma_c(a, x), -inc_gamma_da(a, x) * d_a - inc_gamma_dx(a, x) * d_x};
}

} // namespace Math
} // namespace ROOT
} // namespace custom_derivatives
} // namespace clad

// NOLINTEND

// clang-format on

#endif // CLAD_BENCHMARK_HISTFACTORY_MATHFUNCS_H
