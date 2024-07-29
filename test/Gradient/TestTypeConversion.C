// RUN: %cladnumdiffclang %s  -I%S/../../include -oTestTypeConversion.out 2>&1 | %filecheck %s
// RUN: ./TestTypeConversion.out | %filecheck_exec %s
// RUN: %cladnumdiffclang -Xclang -plugin-arg-clad -Xclang -enable-tbr %s  -I%S/../../include -oTestTypeConversion.out
// RUN: ./TestTypeConversion.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

#include "../TestUtils.h"

float fn_type_conversion(float z, int a) {
  for (int i = 1; i < a; i++){
    z = z * a;
  }
  return z;
}

void fn_type_conversion_grad(float z, int a, float *_d_z, int *_d_a);
// CHECK: void fn_type_conversion_grad(float z, int a, float *_d_z, int *_d_a) {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     int i = 0;
// CHECK-NEXT:     clad::tape<float> _t1 = {};
// CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
// CHECK-NEXT:     for (i = 1; ; i++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(i < a))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, z);
// CHECK-NEXT:         z = z * a;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_z += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         i--;
// CHECK-NEXT:         {
// CHECK-NEXT:             z = clad::pop(_t1);
// CHECK-NEXT:             float _r_d0 = *_d_z;
// CHECK-NEXT:             *_d_z = 0;
// CHECK-NEXT:             *_d_z += _r_d0 * a;
// CHECK-NEXT:             *_d_a += z * _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }

#define TEST(F, x, y)                                                 \
  {                                                                   \
    result_0 = 0;                                                     \
    result_1 = 0;                                                     \
    clad::gradient(F);                                                \
    F##_grad(x, y, &result_0, &result_1);                             \
    printf("Result is = {%.2f, %.2f}\n", result_0, (float)result_1);  \
  }

int main() {
  float result_0;
  int result_1;
  INIT_GRADIENT(fn_type_conversion);
  TEST_GRADIENT(fn_type_conversion, /*numOfDerivativeArgs=*/2, 4, 3, &result_0, &result_1); // CHECK-EXEC: {9.00, 24}
}
