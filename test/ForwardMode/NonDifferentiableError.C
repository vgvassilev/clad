// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#define non_differentiable __attribute__((annotate("non_differentiable")))

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char* fmt, ...);

class non_differentiable SimpleFunctions2 {
public:
  SimpleFunctions2() noexcept : x(0), y(0) {}
  SimpleFunctions2(double p_x, double p_y) noexcept : x(p_x), y(p_y) {}
  double x;
  double y;
  double mem_fn(double i, double j) { return (x + y) * i + i * j * j; } // expected-error {{attempted differentiation of method 'mem_fn' in class 'SimpleFunctions2', which is marked as non-differentiable}}
};

non_differentiable double fn_s2_mem_fn(double i, double j) {
  SimpleFunctions2 obj(2, 3);
  return obj.mem_fn(i, j) + i * j;
}

#define INIT_EXPR(classname)                                                   \
  classname expr_1(2, 3);                                                      \
  classname expr_2(3, 5);

#define TEST_CLASS(classname, name, i, j)                                      \
  auto d_##name = clad::differentiate(&classname::name, "i");                  \
  printf("%.2f\n", d_##name.execute(expr_1, i, j));                            \
  printf("%.2f\n", d_##name.execute(expr_2, i, j));                            \
  printf("\n");

#define TEST_FUNC(name, i, j)                                                  \
  auto d_##name = clad::differentiate(&name, "i");                             \
  printf("%.2f\n", d_##name.execute(i, j));                                    \
  printf("\n");

int main() {
  INIT_EXPR(SimpleFunctions2);
  TEST_CLASS(SimpleFunctions2, mem_fn, 3, 5);
  TEST_FUNC(fn_s2_mem_fn, 3, 5);  // expected-error {{attempted differentiation of function 'fn_s2_mem_fn', which is marked as non-differentiable}}
}