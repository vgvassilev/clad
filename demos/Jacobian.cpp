#include "clad/Differentiator/Differentiator.h"

double f_square_add(double a, double b) {
  double check[2] = {a * a, b * b};
  double d = check[0];
  double e = check[1];
  return d + e;
}

double f_cubed_add(double a, double b) {
  return a * a * a + b * b * b;
}

double ads;

void f_0(double a, double b, double c, double output[]) {
    ads = a * a * a + b * b * b;
    output[0] = ads;
    double f =  c * c - a * a * a;
    output[1] = f;
    double g = b / (a * a * a);
    output[2] = g;
}


void jac_check(double input[], double output[]) {
  output[0] = 2 * input[0] + input[1] * input[1];
  output[1] = 5 * input[2] + input[0] / input[1];
  output[3] = 10 * input[1] + input[2] - input[0];
}


int main() {
  auto g = clad::jacobian(f_0);
  g.dump();
  // double input_test[] = {5};
  // double result[10];
  // g.execute(1, 10, input_test);
  // printf("%.2f %.2f %.2f %.2f\n", result[0], result[1], result[2], result[3]);
}
