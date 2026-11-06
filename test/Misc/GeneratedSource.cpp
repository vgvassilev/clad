// -fgenerate-source-file writes the derivatives clad generates for the
// translation unit to a file, once, at its end. A custom derivative, already
// in the user's source, is not written, and nothing is written after an error.
//
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerate-source-file=%t.derivatives.cpp > /dev/null 2>&1
// RUN: FileCheck --input-file=%t.derivatives.cpp \
// RUN:   --implicit-check-not=userWrote_darg0 %s
// RUN: rm -f %t.error.cpp
// RUN: not %cladclang %s -I%S/../../include -fsyntax-only -DERROR \
// RUN:   -Xclang -plugin-arg-clad -Xclang -fgenerate-source-file=%t.error.cpp \
// RUN:   > /dev/null 2>&1
// RUN: not ls %t.error.cpp
// Without a file name, the derivatives go to Derivatives.cpp.
// RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir && %cladclang %s \
// RUN:   -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerate-source-file > /dev/null 2>&1
// RUN: FileCheck --input-file=%t.dir/Derivatives.cpp \
// RUN:   --implicit-check-not=userWrote_darg0 %s
// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -plugin-arg-clad \
// RUN:   -Xclang -fgenerate-source-file=%t.missing/out.cpp 2>&1 \
// RUN:   | FileCheck --check-prefix=NOWRITE %s
// NOWRITE: warning: could not write the derivatives to '{{.*}}missing{{.}}out.cpp'

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

namespace geo {
namespace detail {
inline double norm2(double x, double y) { return x * x + y * y; }
} // namespace detail
} // namespace geo

namespace lit {
inline double operator""_k(long double v) {
  return static_cast<double>(v) * 1000;
}
} // namespace lit
namespace m {
inline double sq(double x) { return x * x; }
} // namespace m

namespace {
inline double scaled(double k, double x) { return k * x; }
} // namespace

// The derivatives clad generates for f are written.
double f(double x, double y) {
  using namespace lit;
  using m::sq;
  return geo::detail::norm2(x, y) + sq(x) * 2.0_k + scaled(3, y);
}

// CHECK: f_darg0(double x, double y)
// CHECK: f_grad(double x, double y, double *_d_x, double *_d_y)

// A custom derivative is the user's, already in their source: not written
// again.
double userWrote(double x) { return x * x; }
namespace clad {
namespace custom_derivatives {
double userWrote_darg0(double x) { return 2 * x; }
} // namespace custom_derivatives
} // namespace clad

// The adjoint of a global is written as a declaration.
double scale = 2;
double timesScale(double x) { return scale * x; }

// CHECK: double _d_scale = 0.;

#ifdef ERROR
double broken(double x) { return undeclared(x); }
#endif

auto d_f = clad::differentiate(f, "x");
auto g_f = clad::gradient(f);
auto d_userWrote = clad::differentiate(userWrote, "x");
auto g_timesScale = clad::gradient(timesScale);
