#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H
#include "clad/Differentiator/ArrayRef.h"
#include "clad/Differentiator/Differentiator.h"

#include <algorithm>
#include <cstdio>
#include <type_traits>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <limits>
#include <string>

namespace test_utils {

template <typename T> void print(T t) {
  fprintf(stderr, "Print method not defined for type: %s", typeid(t).name());
}

void print(float f) { printf("%.2f", f); }

void print(const char* s) { printf("%s", s); }

void print(double d) { printf("%.2f", d); }

void print(int i) { printf("%d", i); }

void print(long double ld) { printf("%.2Lf", ld); }

void print() {}

template <typename T> void print(clad::array_ref<T> arr) {
  for (std::size_t i = 0; i < arr.size(); ++i) {
    print(arr[i]);
    if (i != arr.size() - 1)
      printf(", ");
  }
}

template <typename Arg, typename... Args> void print(Arg* arg, Args... args);

// Prints comma separated list of all the arguments values.
template <typename Arg, typename... Args,
          class = typename std::enable_if<!std::is_pointer<Arg>::value>::type>
void print(Arg arg, Args... args) {
  print(arg);
  if (sizeof...(Args) > 0)
    printf(", ");
  print(args...);
}

template <typename Arg, typename... Args> void print(Arg* arg, Args... args) {
  print(*arg);
  if (sizeof...(Args) > 0)
    printf(", ");
  print(args...);
}

// Prints comma separated list of all the arguments values enclosed in curly
// braces and terminated by a new line.
template <typename... Args> void display(Args... args) {
  printf("{");
  print(args...);
  printf("}\n");
}

template <typename T> void displayarr(T* arr, std::size_t n) {
  printf("{");
  for (std::size_t i = 0; i < n; ++i) {
    print(arr[i]);
    if (i != n - 1) {
      printf(", ");
    }
  }
  printf("}\n");
}

template <typename T> void displayarray_ref(clad::array_ref<T> arr) {
  printf("{");
  for (std::size_t i = 0; i < arr.size(); ++i) {
    print(arr[i]);
    if (i != arr.size() - 1)
      printf(", ");
  }
  printf("}\n");
}

template <std::size_t...> struct index_pack {};

template <std::size_t l, std::size_t r, std::size_t... S>
struct GenerateRange : GenerateRange<l + 1, r, S..., l> {};

template <std::size_t l, std::size_t... S> struct GenerateRange<l, l, S...> {
  using type = index_pack<S..., l>;
};

void reset() {}

template <typename Arg,
          class = typename std::enable_if<!std::is_pointer<Arg>::value>::type>
void reset(Arg& arg) {
  arg = Arg();
}

template <typename Arg> void reset(Arg* arg) { *arg = Arg(); }

template <typename T> void reset(clad::array_ref<T> arr) {
  for (std::size_t i = 0; i < arr.size(); ++i)
    reset(arr[i]);
}

template <typename Arg, typename... Args> void reset(Arg& arg, Args&... args) {
  reset(arg);
  reset(args...);
}

template <class CF, std::size_t... S, class... Args>
void run_gradient_impl(CF cf, index_pack<S...> s, Args&&... args) {
  std::tuple<Args...> t = {args...};
  reset(std::get<S>(t)...);
  cf.execute(args...);
  display(std::get<S>(t)...);
}
template <typename PrecissionType, class CF, std::size_t... S, class... Args>
void run_error_estimation_impl(CF cf, index_pack<S...> s, Args&&... args) {
  std::tuple<Args...> t = {args...};
  reset(std::get<S>(t)...);
  double error = 0;
  cf.execute(args..., error);
  display(std::ldexp(error, std::numeric_limits<PrecissionType>::digits - 1));
}

template <class CF, class... Args>
void run_jacobian_impl(CF cf, std::size_t size, Args&&... args) {
  std::tuple<Args...> t = {args...};
  cf.execute(args...);
  typedef
      typename std::remove_reference<decltype(*std::get<sizeof...(args) - 1>(
          t))>::type arrElemType;
  auto& res = *std::get<sizeof...(args) - 1>(t);
  unsigned numOfParameters = sizeof...(args) - 2;
  unsigned numOfOutputs = res.rows();
  printf("{");
  for (unsigned i = 0; i < numOfOutputs; ++i) {
    for (unsigned j = 0; j < numOfParameters; ++j) {
      printf("%.2f", res[i][j]);
      if (i != numOfOutputs - 1 || j != numOfParameters - 1)
        printf(", ");
    }
  }
  printf("}\n");
}

template <class CF, std::size_t... S, class... Args>
void run_hessian_impl(CF cf, index_pack<S...> s, Args&&... args) {
  std::tuple<Args...> t = {args...};
  reset(std::get<S>(t)...);
  cf.execute(args...);
  typedef
      typename std::remove_reference<decltype(*std::get<sizeof...(args) - 1>(
          t))>::type arrRefElemType;
  displayarray_ref<arrRefElemType>(std::get<sizeof...(args) - 1>(t));
}

template <std::size_t NumOfDerivativeArgs, class CF, class... Args>
void run_gradient(CF cf, Args&&... args) {
  using DerivativeArgsRange =
      typename GenerateRange<sizeof...(Args) - NumOfDerivativeArgs,
                             sizeof...(Args) - 1>::type;
  run_gradient_impl(cf, DerivativeArgsRange(), std::forward<Args>(args)...);
}

template <typename PrecissionType, class CF, class... Args>
void run_error_estimation(CF cf, Args&&... args) {
  using DerivativeArgsRange =
      typename GenerateRange<sizeof...(Args) / 2,
                             sizeof...(Args) - 1>::type;
  run_error_estimation_impl<PrecissionType>(cf, DerivativeArgsRange(), std::forward<Args>(args)...);
}

template <std::size_t NumOfDerivativeArgs, std::size_t size, class CF,
          class... Args>
void run_jacobian(CF cf, Args&&... args) {
  run_jacobian_impl(cf, size, std::forward<Args>(args)...);
}

template <std::size_t NumOfDerivativeArgs, class CF, class... Args>
void run_hessian(CF cf, Args&&... args) {
  using DerivativeArgsRange =
      typename GenerateRange<sizeof...(Args) - NumOfDerivativeArgs,
                             sizeof...(Args) - 1>::type;
  run_hessian_impl(cf, DerivativeArgsRange(), std::forward<Args>(args)...);
}

template <class CF, class... Args>
void run_differentiate(CF cf, Args&&... args) {
  display(cf.execute(std::forward<Args>(args)...));
}

template <typename A, typename B>
void EssentiallyEqual(A a, B b) {
  // FIXME: We should select epsilon value in a more robust way.
  const A epsilon = 1e-12;
  bool ans = std::fabs(a - b) <=
              ((std::fabs(a > b) ? std::fabs(b) : std::fabs(a)) * epsilon);

  assert(ans && "Clad Gradient is not equal to Enzyme Gradient");
}

template <typename A, typename B>
void EssentiallyEqualArrays(A* a, B* b, unsigned size) {
  for (int i = 0; i < size; i++) {
    EssentiallyEqual(a[i], b[i]);
  }
}

// Checking a derivative clad computed against a finite-difference reference.
//
// Everything below is written so that a disagreement means the derivative is
// wrong, not the reference. The reference is the inaccurate side by
// construction -- it differences a function that is itself rounded -- so it
// is built to be as good as the function allows and the tolerance is derived
// from how good that is, rather than picked until the suite went quiet.
//
// Step. A central difference with step h has truncation error f'''h^2/6 and
// roundoff error about eps|f|/h, where eps is the epsilon of the type the
// function is evaluated in. The two balance at h ~ cbrt(eps), and the step
// has to be relative to |x|: a fixed one is too coarse near zero and below
// the spacing of the representable numbers far from it. So h = cbrt(eps)|x|,
// or cbrt(eps) at x = 0.
//
// Richardson. Differencing at h and h/2 and taking (4 D(h/2) - D(h)) / 3
// cancels the h^2 term, which leaves roundoff as the only error worth
// counting. The differences are taken in long double, so the only rounding
// is the function's own; the stencil x +- h is rounded to T before the
// function sees it and the step actually taken is measured back from it.
//
// Tolerance. Let every value of f carry an error of at most k ulps, d =
// k eps |f|. D(h) is then off by at most d/h and D(h/2) by 2d/h, so the
// combination is off by at most (4 * 2d/h + d/h) / 3 = 3d/h. With
// h = cbrt(eps)|x| that is 3k eps^(2/3) |f|/|x|. The check is relative,
//
//   |derivative - reference| <= tol_T * (1 + |reference|),
//
// so tol_T has to cover 3k eps^(2/3) times the worst ratio of |f|/|x| to
// 1 + |f'| over the sample points. Allowing k = 4 (libm's special functions
// are not correctly rounded, the float ones least of all) and a ratio of 8
// gives tol_T = 96 eps^(2/3); rounded up to 128 for headroom, that is about
// 3e-3 for float, 5e-9 for double and 3e-11 for x87 long double. A wider
// long double (IEEE quad on AArch64 Linux, 4e-21) is held to the x87
// tolerance: exp2, log2, log10, erf and erfc take ln 2, ln 10 and pi in
// double, and digamma (lgamma, beta) truncates its series near 1e-11, all
// within 3e-11 and not within 4e-21. Under Valgrind, which does long double
// arithmetic in 64 bits, the long double rows get the double tolerance,
// because the reference is no better than double there. The worst sample
// point in stl-cmath uses under a third of it (acoshf at 1.01, next to the
// singularity at 1); the worst double one, a sixteenth. A wrong sign or
// factor is off by O(1) relative and fails everywhere. A small wrong term may
// stay under the float floor; the double and long double rows still catch it
// when the error is in the template every width shares. Where f itself jumps
// (floor, fmod) the one-sided differences grow like 1/h and the kink check
// below would accept almost anything, so no such point is sampled.
//
// Kinks. At a point where the one-sided derivatives differ -- |x| at 0, fmin
// where its arguments tie -- the central difference averages them, and clad
// returns one of them (a subgradient) by design. Neither is wrong, so such a
// point is not a failure. It is recognised from the function alone: for a
// smooth function the gap between the forward and the backward difference is
// about f''h and halves with the step, at a kink it is the jump and does not.
// The derivative at a kink is still checked to lie between the two one-sided
// values, and the point is reported as `SKIP (kink)`.
//
// Every result goes to stdout, where the RUN line's FileCheck reads it.

/// The epsilon of long double arithmetic as the machine performs it, which
/// can be coarser than numeric_limits says: Valgrind keeps long double in
/// 64 bits. The volatiles keep the compiler from folding the loop.
inline long double long_double_epsilon_in_use() {
  static const long double Eps = [] {
    volatile long double e = 1;
    volatile long double sum = 2;
    while (true) {
      sum = 1 + e / 2;
      if (sum == 1)
        return (long double)e;
      e = e / 2;
    }
  }();
  return Eps;
}

template <typename T> long double fd_tolerance() {
  // 2^-63 is the epsilon of x87 long double; see above. The reference is
  // computed in long double, so it is only as good as that arithmetic.
  const long double eps =
      std::max({(long double)std::numeric_limits<T>::epsilon(), 0x1p-63L,
                long_double_epsilon_in_use()});
  return 128 * std::cbrt(eps * eps);
}

/// A reference derivative of \p f at \p x, and what the stencil says about
/// the shape of \p f around \p x.
struct FDReference {
  long double Value;  ///< The Richardson-combined central difference.
  long double Left;   ///< The backward difference at the finer step.
  long double Right;  ///< The forward difference at the finer step.
  bool Kink;          ///< The one-sided derivatives differ at x.
};

template <typename T, typename F> FDReference fd_reference(F f, T x) {
  const long double eps = std::numeric_limits<T>::epsilon();
  const long double scale = x != 0 ? std::fabs((long double)x) : 1.0L;
  const long double f0 = f(x);
  long double fmax = std::fabs(f0);
  // The differences at step h; returns the central one.
  auto at = [&](long double h, long double& left, long double& right) {
    // Rounded to T so that f is evaluated where the step says, and the step
    // measured back from the rounded points so that it is the one taken.
    volatile T xp = (T)(x + h);
    volatile T xm = (T)(x - h);
    long double hp = (long double)xp - x, hm = x - (long double)xm;
    long double fp = f((T)xp), fm = f((T)xm);
    fmax = std::fmax(fmax, std::fmax(std::fabs(fp), std::fabs(fm)));
    right = (fp - f0) / hp;
    left = (f0 - fm) / hm;
    return (fp - fm) / (hp + hm);
  };
  const long double h = std::cbrt(eps) * scale;
  long double l1, r1, l2, r2;
  long double d1 = at(h, l1, r1);
  long double d2 = at(h / 2, l2, r2);
  FDReference R{(4 * d2 - d1) / 3, l2, r2, false};
  // Roundoff in a one-sided difference at the finer step; a gap that is not
  // well above it says nothing about the shape of f.
  long double noise = 4 * eps * fmax / (h / 2);
  long double gap1 = std::fabs(r1 - l1), gap2 = std::fabs(r2 - l2);
  R.Kink = gap2 > 64 * noise && gap2 > 0.75L * gap1 &&
           gap2 > fd_tolerance<T>() * (1 + std::fabs(d2));
  return R;
}

/// Checks the forward- and reverse-mode derivatives of \p f at \p x: they
/// must agree bitwise -- they are the same derivative, computed twice -- and
/// agree with the finite-difference reference to fd_tolerance. Returns
/// whether the point passed or was skipped as a kink.
template <typename T, typename F>
bool check_derivative(const std::string& name, F f, T x, T dfwd, T drev) {
  const long double tol = fd_tolerance<T>();
  FDReference R = fd_reference(f, x);
  std::cout << std::setprecision(10);
  long double d = dfwd;
  bool ok;
  const char* verdict = "PASS";
  if (R.Kink) {
    long double lo = std::fmin(R.Left, R.Right);
    long double hi = std::fmax(R.Left, R.Right);
    long double slack = tol * (1 + std::fmax(std::fabs(lo), std::fabs(hi)));
    ok = d >= lo - slack && d <= hi + slack;
    verdict = "SKIP (kink)";
  } else {
    ok = std::fabs(d - R.Value) <= tol * (1 + std::fabs(R.Value));
  }
  ok = ok && dfwd == drev;
  if (!ok)
    verdict = "FAIL";
  std::cout << verdict << ": " << name << " at x=" << x << ": dfwd=" << dfwd
            << ", drev=" << drev << ", numerical=" << (double)R.Value;
  if (R.Kink)
    std::cout << " (one-sided " << (double)R.Left << ", " << (double)R.Right
              << ")";
  std::cout << "\n";
  return ok;
}

// 1D Test function for comparing Forward vs Reverse vs Numerical
template<typename T>
void test_func(const std::string &name,
               T (*f)(T), const clad::CladFunction<T(*)(T)> &clad_fwd,
               const decltype(clad::gradient((T (*)(T))nullptr)) &clad_rev,
               const T (&test_points)[7] = {-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0}) {
    for (T x : test_points) {
        T dfwd_val = clad_fwd.execute(x);
        T drev_val = T();
        clad_rev.execute(x, &drev_val);
        check_derivative(name, f, x, dfwd_val, drev_val);
    }
}


#define INIT_GRADIENT_ALL(fn) auto fn##_grad = clad::gradient(fn);

#define INIT_JACOBIAN_ALL(fn) auto fn##_jac = clad::jacobian(fn);

#define INIT_HESSIAN_ALL(fn) auto fn##_hessian = clad::hessian(fn);

#define INIT_DIFFERENTIATE(fn, ...)                                            \
  auto fn##_diff = clad::differentiate(fn, __VA_ARGS__);

#define INIT_DIFFERENTIATE_UA(fn, ...)                                         \
  auto fn##_diff = clad::differentiate<clad::opts::enable_ua>(fn, __VA_ARGS__);

#define INIT_GRADIENT_SPECIFIC(fn, args)                                       \
  auto fn##_grad = clad::gradient(fn, args);

#define INIT_JACOBIAN_SPECIFIC(fn, args)                                       \
  auto fn##_jac = clad::jacobian(fn, args);

#define INIT_HESSIAN_SPECIFIC(fn, args)                                        \
  auto fn##_hessian = clad::hessian(fn, args);

#define GET_MACRO(_1, _2, MACRO, ...) MACRO

#define INIT_GRADIENT(...)                                                     \
  GET_MACRO(__VA_ARGS__, INIT_GRADIENT_SPECIFIC, INIT_GRADIENT_ALL)            \
  (__VA_ARGS__)

#define INIT_JACOBIAN(...)                                                     \
  GET_MACRO(__VA_ARGS__, INIT_JACOBIAN_SPECIFIC, INIT_JACOBIAN_ALL)            \
  (__VA_ARGS__)

#define INIT_HESSIAN(...)                                                      \
  GET_MACRO(__VA_ARGS__, INIT_HESSIAN_SPECIFIC, INIT_HESSIAN_ALL)              \
  (__VA_ARGS__)

#define INIT_ERROR_ESTIMATION(fn)                                             \
  auto fn##_err = clad::estimate_error(fn);

#define TEST_GRADIENT(fn, numOfDerivativeArgs, ...)                            \
  test_utils::run_gradient<numOfDerivativeArgs>(fn##_grad, __VA_ARGS__);

#define TEST_DIFFERENTIATE(fn, ...)                                            \
  test_utils::run_differentiate(fn##_diff, __VA_ARGS__);

#define TEST_JACOBIAN(fn, numOfDerivativeArgs, size, ...)                      \
  test_utils::run_jacobian<numOfDerivativeArgs, size>(fn##_jac, __VA_ARGS__);

#define TEST_HESSIAN(fn, numOfDerivativeArgs, ...)                             \
  test_utils::run_hessian<numOfDerivativeArgs>(fn##_hessian, __VA_ARGS__);

#define TEST_ERROR_ESTIMATION(fn, PrecissionType, ...)                    \
  test_utils::run_error_estimation<PrecissionType>(fn##_err, __VA_ARGS__);

#endif
} // namespace test_utils