// RUN: %cladclang %s -fno-exceptions -I%S/../../include -oVirtualFunctions.out 2>&1 | %filecheck %s
// RUN: ./VirtualFunctions.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"

extern "C" int printf(const char*, ...);

class BaseShape {
public:
  virtual double area(double scale) const {
    return scale * 10.0;
  }
};

class Square : public BaseShape {
public:
  double area(double scale) const override {
    return scale * scale * 5.0;
  }
};

int main() {
  BaseShape base;
  Square sq;
  BaseShape* ptr = &sq;

  auto df_base = clad::differentiate(&BaseShape::area);
  auto df_sq = clad::differentiate(&Square::area);

  double res_base = df_base.execute(base, 2.0);
  double res_sq = df_sq.execute(sq, 2.0);
  double res_vtable = df_base.execute(*ptr, 2.0);

  printf("base_df = %.1f, sq_df = %.1f, vtable_df = %.1f\n", res_base, res_sq, res_vtable);
  // CHECK-EXEC: base_df = 10.0, sq_df = 20.0, vtable_df = 20.0
  return 0;
}
