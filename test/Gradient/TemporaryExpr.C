// RUN: %cladclang %s -fno-exceptions -I%S/../../include -oTemporaryExpr.out 2>&1 | %filecheck %s
// RUN: ./TemporaryExpr.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}
#include "clad/Differentiator/Differentiator.h"

class SimpleFunctions {
public:
  SimpleFunctions(double p_x = 0, double p_y = 0) : x(p_x), y(p_y) {}
  double x, y;
  double mem_fn(double i, double j) { return (x + y) * i + i * j; }

  SimpleFunctions operator*(SimpleFunctions& rhs) {
      return SimpleFunctions(this->x * rhs.x, this->y * rhs.y);
  }
};

double fn1(double i, double j) {
    SimpleFunctions sf(3, 5);

    return sf.mem_fn(i, j);
}

double fn2(double i, double j) {
    SimpleFunctions sf(3 * i, 5 * j);

    return sf.mem_fn(i, j);
}

double fn3(double i, double j) {
    SimpleFunctions sf1 (3, 5);
    SimpleFunctions sf2 (i, j);

    SimpleFunctions r = sf1 * sf2;
    return r.mem_fn(i, j);

    /*return (sf1 * sf2).mem_fn(i, j);*/
}

int main() {
    double result1[2] = {};
    auto fn1_grad = clad::gradient(fn1);
    fn1_grad.execute(4, 5, &result1[0], &result1[1]);
    printf("%f %f\n", result1[0], result1[1]); // CHECK-EXEC: 13.000000 4.000000

    double result2[2] = {};
    auto fn2_grad = clad::gradient(fn2);
    fn2_grad.execute(4, 5, &result2[0], &result2[1]);
    printf("%f %f\n", result2[0], result2[1]); // CHECK-EXEC: 54.000000 24.000000

    double result3[2] = {};
    auto fn3_grad = clad::gradient(fn3);
    fn3_grad.execute(4, 5, &result3[0], &result3[1]);
    printf("%f %f\n", result3[0], result3[1]); // CHECK-EXEC: 54.000000 24.000000
}
