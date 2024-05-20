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
    double _d_ix = 0;
    const double ix = 1 + params[0];
    goto _label0;
  _label0:
    {
        double _r0 = 0;
        double _r1 = 0;
        double _r2 = 0;
        double _r3 = 0;
        double _grad0[4] = {0};
        numerical_diff::central_difference(foo, _grad0, 0, 10., ix, 1., 0);
        _r0 += 1 * _grad0[0];
        _r1 += 1 * _grad0[1];
        _r2 += 1 * _grad0[2];
        _r3 += 1 * _grad0[3];
        _d_ix += _r1;
    }
    _d_params[0] += _d_ix;
}

)";
  EXPECT_EQ(actual, expected);
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
  auto grad = clad::hessian(wrapper2, "params[0]");
  double x = 4.0;
  double dx = 0.0;
  grad.execute(&x, &dx);
  EXPECT_DOUBLE_EQ(dx, 2.0);
}