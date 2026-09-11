// RUN: %cladclang %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -disable-va -o %t.no-analysis
// RUN: %t.no-analysis | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -enable-va -o %t.va
// RUN: %t.va | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

double mixed_captures(double x, double y) {
  auto inner = [x, &y](double scale) { return x * y * scale; };
  x = 7;
  return inner(2);
}

double nested_captures(double x, double y) {
  auto outer = [&] {
    auto inner = [&] { return x * y; };
    return inner();
  };
  return outer();
}

double early_return(double x) {
  auto inner = [&](double y) {
    if (y < 0)
      return x * y;
    return x * x;
  };
  return inner(x);
}

double renamed_capture(double x) {
  double _d_x = x;
  auto inner = [_d_x] { return _d_x * _d_x; };
  return inner();
}

double initialized_capture(double x) {
  auto inner = [y = x++] { return y * y; };
  return inner() + x;
}

double loop_capture(double x) {
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    auto inner = [x] { return x * x; };
    result += inner();
    x += 1;
  }
  return result;
}

double loop_reference_capture(double x) {
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    auto inner = [&] { return x * x; };
    result += inner();
    x += 1;
  }
  return result;
}

double loop_initialized_capture(double x) {
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    auto inner = [value = x * x] { return value * value; };
    result += inner();
    x += 1;
  }
  return result;
}

double conditional_capture(double x) {
  double result = 0;
  if (x > 0) {
    auto inner = [x] { return x * x; };
    result = inner();
  }
  return result;
}

struct CopyValue {
  double value;
  CopyValue() : value(0) {}
  CopyValue(double x) : value(x) {}
  CopyValue(const CopyValue& other) : value(other.value * 2) {}
};

double mutable_capture(double x) {
  auto inner = [x]() mutable { x *= 2; return x * x; };
  double first = inner();
  return first + inner();
}

double nested_conditional_capture(double x) {
  auto outer = [&] {
    double result = 0;
    if (x > 0) {
      auto inner = [&] { return x * x; };
      result = inner();
    }
    return result;
  };
  return outer();
}

double nested_loop_capture(double x) {
  auto outer = [&] {
    double result = 0;
    for (int i = 0; i < 2; ++i) {
      auto inner = [x] { return x * x; };
      result += inner();
    }
    return result;
  };
  return outer();
}

double mutating_reference_capture(double x) {
  auto inner = [&y = x] { y *= 2; return y * y; };
  double value = inner();
  return value + x;
}

double mutable_initialized_capture(double x) {
  auto inner = [y = x]() mutable { y *= 2; return y * y; };
  double first = inner();
  return first + inner();
}

double unused_capture_initializer(double x) {
  auto inner = [unused = (x *= 2)] { return 1.0; };
  return inner() + x;
}

double nested_early_return_capture(double x) {
  auto outer = [&] {
    if (x < 0) return x * x;
    auto inner = [&](double y) { return x * y; };
    return inner(x);
  };
  return outer();
}

double elided_copy_capture(double x) {
  auto inner = [v = CopyValue(x)] { return v.value * v.value; };
  double value = inner();
  return value * value;
}

double repeated_loop_calls(double x) {
  auto inner = [x]() mutable { x *= 2; return x * x; };
  double result = 0;
  for (int i = 0; i < 3; ++i)
    result += inner();
  return result;
}

double loop_reference_initializer(double x) {
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    auto inner = [&y = x] { y *= 2; return y * y; };
    result += inner();
  }
  return result;
}

double default_and_initialized_capture(double x) {
  auto inner = [&, value = x]() mutable { value *= 2; return value * value; };
  double first = inner();
  return first + inner();
}

double nested_if_initializer(double x) {
  auto outer = [&] {
    if (double y = x * x; y > 0) {
      auto inner = [&] { return y * x; };
      return inner();
    }
    return 0.0;
  };
  return outer();
}

double unused_mutating_call(double x) {
  auto inner = [&] { x *= 2; };
  inner();
  return x * x;
}

int main() {
  double dx = 0, dy = 0;
  clad::gradient(mixed_captures).execute(3, 4, &dx, &dy);
  std::printf("%.1f %.1f\n", dx, dy);
  // CHECK-EXEC: 8.0 6.0
  dx = dy = 0;
  clad::gradient(nested_captures).execute(3, 4, &dx, &dy);
  std::printf("%.1f %.1f\n", dx, dy);
  // CHECK-EXEC-NEXT: 4.0 3.0
  dx = 0;
  clad::gradient(early_return).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -4.0
  dx = 0;
  clad::gradient(early_return).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 4.0
  dx = 0;
  clad::gradient(renamed_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 4.0
  dx = 0;
  clad::gradient(initialized_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 5.0
  dx = 0;
  clad::gradient(loop_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 18.0
  dx = 0;
  clad::gradient(loop_reference_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 18.0
  dx = 0;
  clad::gradient(loop_initialized_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 396.0
  dx = 0;
  clad::gradient(conditional_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 4.0
  dx = 0;
  clad::gradient(conditional_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 0.0
  dx = 0;
  clad::gradient(mutable_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 80.0
  dx = 0;
  clad::gradient(mutable_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -80.0
  dx = 0;
  clad::gradient(nested_conditional_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 4.0
  dx = 0;
  clad::gradient(nested_conditional_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 0.0
  dx = 0;
  clad::gradient(nested_loop_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 8.0
  dx = 0;
  clad::gradient(nested_loop_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -8.0
  dx = 0;
  clad::gradient(mutating_reference_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 18.0
  dx = 0;
  clad::gradient(mutating_reference_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -14.0
  dx = 0;
  clad::gradient(mutable_initialized_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 80.0
  dx = 0;
  clad::gradient(mutable_initialized_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -80.0
  dx = 0;
  clad::gradient(unused_capture_initializer).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 2.0
  dx = 0;
  clad::gradient(unused_capture_initializer).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 2.0
  dx = 0;
  clad::gradient(nested_early_return_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 4.0
  dx = 0;
  clad::gradient(nested_early_return_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -4.0
  dx = 0;
  clad::gradient(elided_copy_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 32.0
  dx = 0;
  clad::gradient(elided_copy_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -32.0
  dx = 0;
  clad::gradient(repeated_loop_calls).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 336.0
  dx = 0;
  clad::gradient(repeated_loop_calls).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -336.0
  dx = 0;
  clad::gradient(loop_reference_initializer).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 336.0
  dx = 0;
  clad::gradient(loop_reference_initializer).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -336.0
  dx = 0;
  clad::gradient(default_and_initialized_capture).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 80.0
  dx = 0;
  clad::gradient(default_and_initialized_capture).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -80.0
  dx = 0;
  clad::gradient(nested_if_initializer).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 12.0
  dx = 0;
  clad::gradient(nested_if_initializer).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 12.0
  dx = 0;
  clad::gradient(unused_mutating_call).execute(2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: 16.0
  dx = 0;
  clad::gradient(unused_mutating_call).execute(-2, &dx);
  std::printf("%.1f\n", dx);
  // CHECK-EXEC-NEXT: -16.0
}
