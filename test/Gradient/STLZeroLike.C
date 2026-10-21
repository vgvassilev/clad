// RUN: %cladclang %s -I%S/../../include -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -o %t14
// RUN: %t14 | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -o %t20
// RUN: %t20 | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include "clad/Differentiator/STLBuiltins.h"

extern "C" int printf(const char*, ...);

namespace pair_test {
struct Config {
  double value;
  int extent;

  explicit Config(double value, int extent) : value(value), extent(extent) {}
};

Config zero_like(const Config& value) { return Config(0., value.extent); }
} // namespace pair_test

struct CopyFallback {
  double value;

  explicit CopyFallback(double value) : value(value) {}
};

template <class T>
auto has_zero_like(int)
    -> decltype(clad::zero_like(std::declval<const T&>()), std::true_type{});
template <class T> std::false_type has_zero_like(...);

static_assert(!decltype(has_zero_like<std::pair<CopyFallback, double>>(0))::value,
              "Unsupported members must preserve the copy-and-zero fallback");
static_assert(!decltype(has_zero_like<std::pair<double&, double&>>(0))::value,
              "A new adjoint must not borrow the primal's storage");

double replace_pair(std::pair<double, double> values, double x) {
  values.first = x;
  return values.first * values.second;
}

// CHECK: void replace_pair_grad_1(std::pair<double, double> values, double x, double *_d_x) {
// CHECK-NEXT:     std::pair<double, double> _d_values(clad::zero_like(values));

double replace_fallback(std::pair<CopyFallback, double> values, double x) {
  values.first.value = x;
  return values.first.value * values.first.value;
}

// CHECK: void replace_fallback_grad_1(std::pair<CopyFallback, double> values, double x, double *_d_x) {
// CHECK-NEXT:     std::pair<CopyFallback, double> _d_values(values);
// CHECK-NEXT:     clad::zero_init(_d_values);

int main() {
  std::pair<std::pair<double, double>, std::vector<double>> nested{
      {3., 4.}, {5., 6.}};
  auto d_nested = clad::zero_like(nested);
  printf("%.1f %.1f %zu %.1f %.1f\n", d_nested.first.first,
         d_nested.first.second, d_nested.second.size(), d_nested.second[0],
         d_nested.second[1]);
  d_nested.second[0] = 9.;
  printf("%.1f\n", nested.second[0]);

  std::pair<pair_test::Config, double> config{pair_test::Config(7., 11), 5.};
  auto d_config = clad::zero_like(config);
  printf("%.1f %d %.1f\n", d_config.first.value, d_config.first.extent,
         d_config.second);

  double first = 3., second = 4.;
  std::pair<double&, double&> references{first, second};
  clad::zero_init(references);
  printf("%.1f %.1f\n", first, second);

  std::pair<double, double> values{10., 4.};
  double d_x = 0.;
  clad::gradient(replace_pair, "x").execute(values, 3., &d_x);
  printf("%.1f\n", d_x);

  std::pair<CopyFallback, double> fallback{CopyFallback(10.), 4.};
  double d_fallback = 0.;
  clad::gradient(replace_fallback, "x").execute(fallback, 3., &d_fallback);
  printf("%.1f\n", d_fallback);
}

// CHECK-EXEC: 0.0 0.0 2 0.0 0.0
// CHECK-EXEC-NEXT: 5.0
// CHECK-EXEC-NEXT: 0.0 11 0.0
// CHECK-EXEC-NEXT: 0.0 0.0
// CHECK-EXEC-NEXT: 4.0
// CHECK-EXEC-NEXT: 6.0
