#include "clad/Differentiator/Differentiator.h"

#include <iostream>
#include <regex>
#include <string>

#include "gtest/gtest.h"

double foo(double x, double alpha, double theta, double x0 = 0);

double wrapper1(double* params) {
  const double ix = 1 + params[0];
  return foo(10., ix, 1.0);
}

TEST(CallDeclOnly, CheckNumDiff) {
  auto grad = clad::gradient(wrapper1, "params");
  // Collect output of grad.dump() into a string as it ouputs using llvm::outs()
  std::string actual;
  testing::internal::CaptureStdout();
  grad.dump();
  actual = testing::internal::GetCapturedStdout();

  // Check the generated code from grad.dump()
  std::string expected = R"(The code is: 
void wrapper1_grad(double *params, double *_d_params) {
    double _d_ix = 0.;
    const double ix = 1 + params[0];
    {
        double _grad0[4] = {0};
        numerical_diff::central_difference<double (*)(double, double, double, double)>(foo, _grad0, 0, 10., ix, 1., 0);
        double _r0 = 1 * _grad0[0];
        double _r1 = 1 * _grad0[1];
        double _r2 = 1 * _grad0[2];
        double _r3 = 1 * _grad0[3];
        _d_ix += _r1;
    }
    _d_params[0] += _d_ix;
}

)";
  EXPECT_EQ(actual, expected);
}

namespace M {
using Real = double;
// Instantiated in Defs.cpp.
template <typename T> Real cube(T x);
} // namespace M

double wrapper3(double x) { return M::cube(x); }

// Defined in Defs.cpp. An inline namespace makes Ratio a name of the file
// scope, and the using declaration makes M::Real one, so both stay as written;
// before clang 16 the latter prints with the scope of what it names.
inline namespace V {
using Ratio = double;
} // namespace V
using M::Real;
Ratio scale(Real x);

double wrapper4(double x) { return scale(x); }

TEST(CallDeclOnly, CheckNumDiffNames) {
  auto d = clad::differentiate(wrapper4, "x");
  testing::internal::CaptureStdout();
  d.dump();
  std::string actual = testing::internal::GetCapturedStdout();
  std::regex expected(
      R"(forward_central_difference<Ratio \(\*\)\((M::)?Real\)>\(scale, x)");
  EXPECT_TRUE(std::regex_search(actual, expected)) << actual;
  EXPECT_NEAR(d.execute(2.0), 3.0, 1e-6);
}

TEST(CallDeclOnly, CheckNumDiffTemplate) {
  auto d = clad::differentiate(wrapper3, "x");
  testing::internal::CaptureStdout();
  d.dump();
  std::string actual = testing::internal::GetCapturedStdout();
  // The template argument picks the specialization `M::cube` alone does not
  // name, and spells `Real` as code outside of `M` can. Before clang 22 the
  // instantiated type prints with its scope; from 22 on it prints as written,
  // and is spelled as what it names.
  std::regex expected(
      R"(return numerical_diff::forward_central_difference<)"
      R"((M::Real|double) \(\*\)\(double\)>\(M::cube, x, 0, 0, x\))");
  EXPECT_TRUE(std::regex_search(actual, expected)) << actual;
  EXPECT_NEAR(d.execute(2.0), 12.0, 1e-6);
}

namespace clad {
namespace custom_derivatives {
// Custom pushforward for the square function but definition will be linked from
// another file.
clad::ValueAndPushforward<double, double> sq_pushforward(double x, double _d_x);
} // namespace custom_derivatives
} // namespace clad

double sq(double x) { return x * x; }

double wrapper2(double* params) { return sq(params[0]); }

TEST(CallDeclOnly, CheckCustomDiff) {
  auto hess = clad::hessian(wrapper2, "params[0]");
  double x = 4.0;
  double dx = 0.0;
  hess.execute(&x, &dx);
  EXPECT_DOUBLE_EQ(dx, 2.0);
}

namespace clad {
namespace custom_derivatives {
float custom_fn_darg0(float x, float y);

void custom_fn_darg0_grad(float x, float y, float* d_x, float* d_y);

float custom_fn_darg1(float x, float y) { return exp(y); }
} // namespace custom_derivatives
} // namespace clad

float custom_fn(float x, float y) {
  // This is to test that Clad actual doesn't generate a derivative for sin(x)
  // as it is commented out, but use the user provided derivatives, which
  // assumes function is sin(x) + exp(y).
  return /*sin(x)*/ +exp(y);
}

TEST(CallDeclOnly, CheckCustomDiff2) {
  auto hessian = clad::hessian(custom_fn);
  float result[4] = {0.0, 0.0, 0.0, 0.0};
  float x = 1.0;
  float y = 2.0;
  hessian.execute(x, y, result);
  EXPECT_FLOAT_EQ(result[0], -sin(x));
  EXPECT_FLOAT_EQ(result[1], 0.0);
  EXPECT_FLOAT_EQ(result[2], 0.0);
  EXPECT_FLOAT_EQ(result[3], exp(y));
}