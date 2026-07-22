#ifndef TEST_TEST_UTILS_H
#define TEST_TEST_UTILS_H
#include "clad/Differentiator/ArrayRef.h"

#include <cstdio>
#include <type_traits>
#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
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

template <typename T>
T get_tolerance() {
    if (std::is_same<T,float>::value) return T(1e-3);
    else if (std::is_same<T,long double>::value) return T(1e-10);
    else return T(1e-6);
}

// step for numerical derivative
template <typename T>
T get_h() {
    if (std::is_same<T,float>::value) return T(1e-3);
    else if (std::is_same<T,long double>::value) return T(1e-8);
    else return T(1e-6);
}

// Central finite difference approximation
template<typename T>
T numerical_derivative(T (*f)(T), T x) {
    const T h = get_h<T>();
    return (f(x + h) - f(x - h)) / (2 * h);
}

// 1D Test function for comparing Forward vs Reverse vs Numerical
template<typename T>
void test_func(const std::string &name,
               T (*f)(T), const clad::CladFunction<T(*)(T)> &clad_fwd,
               const decltype(clad::gradient((T (*)(T))nullptr)) &clad_rev,
               const T (&test_points)[7] = {-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0}) {
    const T tol = get_tolerance<T>();
    std::cerr << std::setprecision(10);
    for (T x : test_points) {
        T dfwd_val = clad_fwd.execute(x);
        T drev_val = T();
        clad_rev.execute(x, &drev_val);
        T df_num = numerical_derivative(f, x);
        if (dfwd_val != drev_val || std::abs(dfwd_val - df_num) > tol) {
            std::cerr << "FAIL: " << name << " at x=" << x
                      << ": dfwd=" << dfwd_val
                      << ", drev=" << drev_val
                      << ", numerical=" << df_num << "\n";
        } else {
            std::cout << "PASS: " << name << " at x=" << x
                      << " dfwd=" << dfwd_val << ", drev=" << drev_val << "\n";
        }
    }
}

inline double seed_component(unsigned i) {
    unsigned s = (i * 2654435761u) ^ 0x9e3779b9u;
    s = s * 1103515245u + 12345u;
    return ((s >> 8) & 0xffff) / 32767.5 - 1.0;
}

inline double dot(const double* a, const double* b, int n) {
    double r = 0;
    for (int i = 0; i < n; ++i) r += a[i] * b[i];
    return r;
}

inline double max_abs_diff(const double* a, const double* b, int n) {
    double m = 0;
    for (int i = 0; i < n; ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

inline bool almost_equal(double a, double b,
                         double rel = 1e-6, double abs_tol = 1e-9) {
    double diff = std::fabs(a - b);
    if (diff <= abs_tol) return true;
    double scale = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    return diff <= rel * scale;
}

template<class F>
void fd_gradient(F f, const double* x, int n, double* g, double h = 1e-6) {
    double* xp = new double[n];
    for (int i = 0; i < n; ++i) xp[i] = x[i];
    for (int i = 0; i < n; ++i) {
        double xi = xp[i];
        xp[i] = xi + h; double fp = f(xp, n);
        xp[i] = xi - h; double fm = f(xp, n);
        xp[i] = xi;
        g[i] = (fp - fm) / (2 * h);
    }
    delete[] xp;
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