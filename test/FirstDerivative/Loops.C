// RUN: %cladclang %s -lm -I%S/../../include -oLoops.out 2>&1 | FileCheck %s
// RUN: ./Loops.out | FileCheck -check-prefix=CHECK-EXEC %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include <cmath>

double f1(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; i = i + 1)
        r = r * x;
    return r;
} // = pow(x, y);

double f1_darg0(double x, int y);
// CHECK: double f1_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   double _d_r = 0.;
// CHECK-NEXT:   double r = 1.;
// CHECK-NEXT:   {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     for (int i = 0; i < y; (_d_i = _d_i + 0) , (i = i + 1)) {
// CHECK-NEXT:       _d_r = _d_r * x + r * _d_x;
// CHECK-NEXT:       r = r * x;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return _d_r;
// CHECK-NEXT: }
// _d_r(i) = _d_r(i - 1) * x + r(i - 1) 
// r(i) = r(i - 1) * x = pow(x, i)
// ->
// _d_r(i) = _d_r(i - 1) * x + pow(x, i - 1) = i * pow(x, i - 1)
// _d_r(y) = y * pow(x, y - 1);

double f1_inc(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; i++)
        r *= x;
    return r;
} // = pow(x, y);

double f1_inc_darg0(double x, int y);
//CHECK:   double f1_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       double _d_r = 0.;
//CHECK-NEXT:       double r = 1.;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < y; i++) {
//CHECK-NEXT:               _d_r = _d_r * x + r * _d_x;
//CHECK-NEXT:               r *= x;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_r;
//CHECK-NEXT:   }


double f2(double x, int y) {
    for (int i = 0; i < y; i = i + 1)
        x = x * x;
    return x;
} // = pow(x, pow(2, y));

double f2_darg0(double x, int y);
// CHECK: double f2_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     for (int i = 0; i < y; (_d_i = _d_i + 0) , (i = i + 1)) {
// CHECK-NEXT:       _d_x = _d_x * x + x * _d_x;
// CHECK-NEXT:       x = x * x;
// CHECK-NEXT:     } 
// CHECK-NEXT:  }
// CHECK-NEXT:  return _d_x;
// CHECK-NEXT: }
// _d_x(i) = 2 * _d_x(i - 1) * x(i - 1) 
// x(i) = x(i - 1) * x(i - 1) = pow(x, pow(2, i))
// ->
// _d_x(i) = 2 * _d_x(i - 1) * pow(x, pow(2, i - 1)) = pow(2, i) * pow(x, sum(j = 0...i-1, pow(2, j))) 
// = pow(2, i) * pow(x, pow(2, i) - 1)
// _d_x(y) = pow(2, i) * pow(x, pow(2, y) - 1)

double f2_inc(double x, int y) {
    for (int i = 0; i < y; i++)
        x *= x;
    return x;
} // = pow(x, pow(2, y));

double f2_inc_darg0(double x, int y);
//CHECK:   double f2_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < y; i++) {
//CHECK-NEXT:               _d_x = _d_x * x + x * _d_x;
//CHECK-NEXT:               x *= x;
//CHECK-NEXT:           }
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_x;
//CHECK-NEXT:   }


double f3(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; r = r * x)
        i = i + 1;
    return r;
} // = pow(x, y);

double f3_darg0(double x, int y);
// CHECK: double f3_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   double _d_r = 0.;
// CHECK-NEXT:   double r = 1.;
// CHECK-NEXT:   {
// CHECK-NEXT:     int _d_i = 0;
// CHECK-NEXT:     for (int i = 0; i < y; (_d_r = _d_r * x + r * _d_x) , (r = r * x)) {
// CHECK-NEXT:       _d_i = _d_i + 0;
// CHECK-NEXT:       i = i + 1;
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return _d_r;
// CHECK-NEXT: }
// = y * pow(x, y-1)

double f3_inc(double x, int y) {
    double r = 1.0;
    for (int i = 0; i < y; r *= x)
        i++;
    return r;
} // = pow(x, y);

double f3_inc_darg0(double x, int y);
//CHECK:   double f3_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       double _d_r = 0.;
//CHECK-NEXT:       double r = 1.;
//CHECK-NEXT:       {
//CHECK-NEXT:           int _d_i = 0;
//CHECK-NEXT:           for (int i = 0; i < y; (_d_r = _d_r * x + r * _d_x) , (r *= x))
//CHECK-NEXT:               i++;
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_r;
//CHECK-NEXT:   }


double f4(double x, int y) {
    double r = 1;
    int i;
    for (i = 0; i < y; r = r * std::sin(x))
        i = i + 1;
    return r;
} // = pow(sin(x), y)

double f4_darg0(double x, int y);
// CHECK: double f4_darg0(double x, int y) {
// CHECK-NEXT:   double _d_x = 1;
// CHECK-NEXT:   int _d_y = 0;
// CHECK-NEXT:   double _d_r = 0;
// CHECK-NEXT:   double r = 1;
// CHECK-NEXT:   int _d_i;
// CHECK-NEXT:   int i;
// CHECK-NEXT:   {
// CHECK-NEXT:     _d_i = 0;
// CHECK-NEXT:     for (i = 0; i < y; [&]             {
// CHECK-NEXT:       double _t1 = std::sin(x);
// CHECK-NEXT:      _d_r = _d_r * _t1 + r * clad::custom_derivatives::sin_pushforward(x, _d_x);
// CHECK-NEXT:      r = r * _t1;
// CHECK:        }
// CHECK:        ()) {
// CHECK-NEXT:          _d_i = _d_i + 0;
// CHECK-NEXT:          i = i + 1;
// CHECK-NEXT:        }
// CHECK-NEXT:     }
// CHECK-NEXT: return _d_r;
// CHECK-NEXT: }
// = y * cos(x) * pow(sin(x), y-1) = y/2.0 * sin(2*x) * pow(sin(x), y-2)

double f4_inc(double x, int y) {
    double r = 1;
    int i;
    for (i = 0; i < y; r *= std::sin(x))
        i++;
    return r;
} // = pow(sin(x), y)

double f4_inc_darg0(double x, int y);
//CHECK:   double f4_inc_darg0(double x, int y) {
//CHECK-NEXT:       double _d_x = 1;
//CHECK-NEXT:       int _d_y = 0;
//CHECK-NEXT:       double _d_r = 0;
//CHECK-NEXT:       double r = 1;
//CHECK-NEXT:       int _d_i;
//CHECK-NEXT:       int i;
//CHECK-NEXT:       {
//CHECK-NEXT:           _d_i = 0;
//CHECK-NEXT:           for (i = 0; i < y; [&]             {
//CHECK-NEXT:               double _t2 = clad::custom_derivatives::sin_pushforward(x, _d_x);
//CHECK-NEXT:               double _t3 = std::sin(x);
//CHECK-NEXT:               _d_r = _d_r * _t3 + r * _t2;
//CHECK-NEXT:               r *= _t3;
//CHECK:        }
//CHECK:        ())
//CHECK-NEXT:           i++;
//CHECK-NEXT:       }
//CHECK-NEXT:       return _d_r;
//CHECK-NEXT:   }

double fn5(double i, double j) {
  int b = 3;
  double a = 0;
  while (b) {
    a += i;
    b -= 1;
  }
  return a;
}

// CHECK: double fn5_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_b = 0;
// CHECK-NEXT:     int b = 3;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     while (b)
// CHECK-NEXT:         {
// CHECK-NEXT:             _d_a += _d_i;
// CHECK-NEXT:             a += i;
// CHECK-NEXT:             _d_b -= 0;
// CHECK-NEXT:             b -= 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     return _d_a;
// CHECK-NEXT: }

double fn6(double i, double j) {
  int b = 3;
  double a = 0;
  do {
    a += i;
    b -= 1;
  } while (b);
  return a;
}

// CHECK: double fn6_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_b = 0;
// CHECK-NEXT:     int b = 3;
// CHECK-NEXT:     double _d_a = 0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     do {
// CHECK-NEXT:         _d_a += _d_i;
// CHECK-NEXT:         a += i;
// CHECK-NEXT:         _d_b -= 0;
// CHECK-NEXT:         b -= 1;
// CHECK-NEXT:     } while (b);
// CHECK-NEXT:     return _d_a;
// CHECK-NEXT: }

double fn7(double i, double j) {
  int b = 3;
  double res = 0;
  while (double a = b) {
    a += i;
    res += a;
    b -= 1;
  }
  b = 1;
  while (b)
    b -= 1;
  return res;
}

// CHECK: double fn7_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_b = 0;
// CHECK-NEXT:     int b = 3;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double res = 0;
// CHECK-NEXT:     while (double a = b)
// CHECK-NEXT:         {
// CHECK-NEXT:             double _d_a = _d_b;
// CHECK-NEXT:             _d_a += _d_i;
// CHECK-NEXT:             a += i;
// CHECK-NEXT:             _d_res += _d_a;
// CHECK-NEXT:             res += a;
// CHECK-NEXT:             _d_b -= 0;
// CHECK-NEXT:             b -= 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     _d_b = 0;
// CHECK-NEXT:     b = 1;
// CHECK-NEXT:     while (b)
// CHECK-NEXT:         {
// CHECK-NEXT:             _d_b -= 0;
// CHECK-NEXT:             b -= 1;
// CHECK-NEXT:         }
// CHECK-NEXT:     return _d_res;
// CHECK-NEXT: }

double fn8(double i, double j) {
  do
  continue;
  while(0);
  return i*i;
}

// CHECK: double fn8_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     do {
// CHECK-NEXT:         continue;
// CHECK-NEXT:     } while (0);
// CHECK-NEXT:     return _d_i * i + i * _d_i;
// CHECK-NEXT: }

double fn9(double i, double j) {
  int counter = 4;
  double a = i*j;
  while (int num = counter) {
    counter-=1;
    if (num == 2)
      continue;
    a += i*i;
  }
  return a;
}

// CHECK: double fn9_darg0(double i, double j) {
// CHECK-NEXT:     double _d_i = 1;
// CHECK-NEXT:     double _d_j = 0;
// CHECK-NEXT:     int _d_counter = 0;
// CHECK-NEXT:     int counter = 4;
// CHECK-NEXT:     double _d_a = _d_i * j + i * _d_j;
// CHECK-NEXT:     double a = i * j;
// CHECK-NEXT:     while (int num = counter)
// CHECK-NEXT:         {
// CHECK-NEXT:             int _d_num = _d_counter;
// CHECK-NEXT:             _d_counter -= 0;
// CHECK-NEXT:             counter -= 1;
// CHECK-NEXT:             if (num == 2)
// CHECK-NEXT:                 continue;
// CHECK-NEXT:             _d_a += _d_i * i + i * _d_i;
// CHECK-NEXT:             a += i * i;
// CHECK-NEXT:         }
// CHECK-NEXT:     return _d_a;
// CHECK-NEXT: }

#define TEST(fn)\
auto d_##fn = clad::differentiate(fn, "i");\
printf("%.2f\n", d_##fn.execute(3, 5));

int main() {
  clad::differentiate(f1, 0);
  printf("Result is = %.2f\n", f1_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f1_inc, 0);
  printf("Result is = %.2f\n", f1_inc_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f2, 0);
  printf("Result is = %.2f\n", f2_darg0(10, 2)); // CHECK-EXEC: Result is = 4000.00

  clad::differentiate(f2_inc, 0);
  printf("Result is = %.2f\n", f2_inc_darg0(10, 2)); // CHECK-EXEC: Result is = 4000.00

  clad::differentiate(f3, 0);
  printf("Result is = %.2f\n", f3_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f3_inc, 0);
  printf("Result is = %.2f\n", f3_inc_darg0(10, 2)); // CHECK-EXEC: Result is = 20.00

  clad::differentiate(f4, 0);
  printf("Result is = %.2f\n", f4_darg0(M_PI/4, 8)); // CHECK-EXEC: Result is = 0.50

  clad::differentiate(f4_inc, 0);
  printf("Result is = %.2f\n", f4_inc_darg0(M_PI/4, 8)); // CHECK-EXEC: Result is = 0.50

  TEST(fn5);  // CHECK-EXEC: 3.00
  TEST(fn6);  // CHECK-EXEC: 3.00
  TEST(fn7);  // CHECK-EXEC: 3.00
  TEST(fn8);  // CHECK-EXEC: 6.00
  TEST(fn9);  // CHECK-EXEC: 23.00
}
