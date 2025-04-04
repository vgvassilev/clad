// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oActivity.out 2>&1 | %filecheck %s
// RUN: ./Activity.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-va -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oActivity.out
// RUN: ./Activity.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

double f1(double x){
  double a = x*x;
  double b = 1;
  b = b*b;
  return a;
}

//CHECK: void f1_grad(double x, double *_d_x) {
//CHECK-NEXT:     double _d_a = 0.;
//CHECK-NEXT:     double a = x * x;
//CHECK-NEXT:     double b = 1;
//CHECK-NEXT:     double _t0 = b;
//CHECK-NEXT:     b = b * b;
//CHECK-NEXT:     _d_a += 1;
//CHECK-NEXT:     b = _t0;
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_a * x;
//CHECK-NEXT:         *_d_x += x * _d_a;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f2(double x){
  double a = x*x;
  double b = 1;
  double g;
  if(a)
    b=x;
  else if(b)
    double d = b;
  else
    g = a;
  return a;
}

//CHECK: void f2_grad(double x, double *_d_x) {
//CHECK-NEXT:     bool _cond0;
//CHECK-NEXT:     double _t0;
//CHECK-NEXT:     bool _cond1;
//CHECK-NEXT:     double d = 0.;
//CHECK-NEXT:     double _t1;
//CHECK-NEXT:     double _d_a = 0.;
//CHECK-NEXT:     double a = x * x;
//CHECK-NEXT:     double _d_b = 0.;
//CHECK-NEXT:     double b = 1;
//CHECK-NEXT:     double _d_g = 0.;
//CHECK-NEXT:     double g;
//CHECK-NEXT:     {
//CHECK-NEXT:         _cond0 = a;
//CHECK-NEXT:         if (_cond0) {
//CHECK-NEXT:             _t0 = b;
//CHECK-NEXT:             b = x;
//CHECK-NEXT:         } else {
//CHECK-NEXT:             _cond1 = b;
//CHECK-NEXT:             if (_cond1)
//CHECK-NEXT:                 d = b;
//CHECK-NEXT:             else {
//CHECK-NEXT:                 _t1 = g;
//CHECK-NEXT:                 g = a;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_a += 1;
//CHECK-NEXT:     if (_cond0) {
//CHECK-NEXT:         b = _t0;
//CHECK-NEXT:         double _r_d0 = _d_b;
//CHECK-NEXT:         _d_b = 0.;
//CHECK-NEXT:         *_d_x += _r_d0;
//CHECK-NEXT:     } else if (!_cond1) {
//CHECK-NEXT:         g = _t1;
//CHECK-NEXT:         double _r_d1 = _d_g;
//CHECK-NEXT:         _d_g = 0.;
//CHECK-NEXT:         _d_a += _r_d1;
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         *_d_x += _d_a * x;
//CHECK-NEXT:         *_d_x += x * _d_a;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f3(double x){
  double x1, x2, x3, x4, x5 = 0;
  while(!x3){
    x5 = x4;
    x4 = x3;
    x3 = x2;
    x2 = x1;
    x1 = x;
  }
  return x5;
}

//CHECK: void f3_grad(double x, double *_d_x) {
//CHECK-NEXT:     clad::tape<double> _t1 = {};
//CHECK-NEXT:     clad::tape<double> _t2 = {};
//CHECK-NEXT:     clad::tape<double> _t3 = {};
//CHECK-NEXT:     clad::tape<double> _t4 = {};
//CHECK-NEXT:     clad::tape<double> _t5 = {};
//CHECK-NEXT:     double _d_x1 = 0., _d_x2 = 0., _d_x3 = 0., _d_x4 = 0., _d_x5 = 0.;
//CHECK-NEXT:     double x1, x2, x3, x4, x5 = 0;
//CHECK-NEXT:     unsigned {{int|long}} _t0 = {{0U|0UL}};
//CHECK-NEXT:     while (!x3) 
//CHECK-NEXT:      {
//CHECK-NEXT:         _t0++;
//CHECK-NEXT:         clad::push(_t1, x5);
//CHECK-NEXT:         x5 = x4;
//CHECK-NEXT:         clad::push(_t2, x4);
//CHECK-NEXT:         x4 = x3;
//CHECK-NEXT:         clad::push(_t3, x3);
//CHECK-NEXT:         x3 = x2;
//CHECK-NEXT:         clad::push(_t4, x2);
//CHECK-NEXT:         x2 = x1;
//CHECK-NEXT:         clad::push(_t5, x1);
//CHECK-NEXT:         x1 = x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _d_x5 += 1;
//CHECK-NEXT:     while (_t0) 
//CHECK-NEXT:      {
//CHECK-NEXT:         {
//CHECK-NEXT:             {
//CHECK-NEXT:                 x1 = clad::pop(_t5);
//CHECK-NEXT:                 double _r_d4 = _d_x1;
//CHECK-NEXT:                 _d_x1 = 0.;
//CHECK-NEXT:                 *_d_x += _r_d4;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x2 = clad::pop(_t4);
//CHECK-NEXT:                 double _r_d3 = _d_x2;
//CHECK-NEXT:                 _d_x2 = 0.;
//CHECK-NEXT:                 _d_x1 += _r_d3;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x3 = clad::pop(_t3);
//CHECK-NEXT:                 double _r_d2 = _d_x3;
//CHECK-NEXT:                 _d_x3 = 0.;
//CHECK-NEXT:                 _d_x2 += _r_d2;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x4 = clad::pop(_t2);
//CHECK-NEXT:                 double _r_d1 = _d_x4;
//CHECK-NEXT:                 _d_x4 = 0.;
//CHECK-NEXT:                 _d_x3 += _r_d1;
//CHECK-NEXT:             }
//CHECK-NEXT:             {
//CHECK-NEXT:                 x5 = clad::pop(_t1);
//CHECK-NEXT:                 double _r_d0 = _d_x5;
//CHECK-NEXT:                 _d_x5 = 0.;
//CHECK-NEXT:                 _d_x4 += _r_d0;
//CHECK-NEXT:             }
//CHECK-NEXT:         }
//CHECK-NEXT:         _t0--;
//CHECK-NEXT:     }
//CHECK-NEXT: }

double f4_1(double v, double u){
  double k = 2*u;
  double n = 2*v;
  return n*k;
}
double f4(double x){
  double c = f4_1(x, 1);
  return c;
}
// CHECK: void f4_1_pullback(double v, double u, double _d_y, double *_d_v, double *_d_u) {  
// CHECK-NEXT:     double k = 2 * u;  
// CHECK-NEXT:     double _d_n = 0.;  
// CHECK-NEXT:     double n = 2 * v;  
// CHECK-NEXT:     _d_n += _d_y * k;  
// CHECK-NEXT:     *_d_v += 2 * _d_n;  
// CHECK-NEXT: }  

// CHECK-NEXT: void f4_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = f4_1(x, 1);
// CHECK-NEXT:     _d_c += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         f4_1_pullback(x, 1, _d_c, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f5(double x){
  double g = x ? 1 : 2;
  return g;
}
// CHECK: void f5_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _cond0 = x;
// CHECK-NEXT:     double _d_g = 0.;
// CHECK-NEXT:     double g = _cond0 ? 1 : 2;
// CHECK-NEXT:     _d_g += 1;
// CHECK-NEXT: }

double f6(double x){
  double a = 0;
  if(0){
    a = x;
  }
  return a;
}

// CHECK: void f6_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double a = 0;
// CHECK-NEXT:     if (0) {
// CHECK-NEXT:         _t0 = a;
// CHECK-NEXT:         a = x;
// CHECK-NEXT:     }
// CHECK-NEXT:     if (0) {
// CHECK-NEXT:         a = _t0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f7(double x){
  double &a = x;
  double* b = &a;
  double arr[3] = {1,2,3};
  double c = arr[0]*(*b)+arr[1]*a+arr[2]*x; 
  return a;
}

// CHECK: void f7_grad(double x, double *_d_x) {
// CHECK-NEXT:     double &_d_a = *_d_x;
// CHECK-NEXT:     double &a = x;
// CHECK-NEXT:     double *_d_b = &_d_a;
// CHECK-NEXT:     double *b = &a;
// CHECK-NEXT:     double _d_arr[3] = {0};
// CHECK-NEXT:     double arr[3] = {1, 2, 3};
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = arr[0] * *b + arr[1] * a + arr[2] * x;
// CHECK-NEXT:     _d_a += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_arr[0] += _d_c * *b;
// CHECK-NEXT:         *_d_b += arr[0] * _d_c;
// CHECK-NEXT:         _d_arr[1] += _d_c * a;
// CHECK-NEXT:         _d_a += arr[1] * _d_c;
// CHECK-NEXT:         _d_arr[2] += _d_c * x;
// CHECK-NEXT:         *_d_x += arr[2] * _d_c;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f8_1(double v, double u){
  return v;
}
double f8(double x){
  double c = f8_1(1, 1);
  double f = f8_1(x, 1);
  return f;
}
// CHECK: void f8_1_pullback(double v, double u, double _d_y, double *_d_v, double *_d_u) {
// CHECK-NEXT:     *_d_v += _d_y;
// CHECK-NEXT: }

// CHECK-NEXT: void f8_grad(double x, double *_d_x) {
// CHECK-NEXT:     double c = f8_1(1, 1);
// CHECK-NEXT:     double _d_f = 0.;
// CHECK-NEXT:     double f = f8_1(x, 1);
// CHECK-NEXT:     _d_f += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         double _r1 = 0.;
// CHECK-NEXT:         f8_1_pullback(x, 1, _d_f, &_r0, &_r1);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f9(double x, double const *obs)
{
   double res = 0.0;
   for (int loopIdx0 = 0; loopIdx0 < 2; loopIdx0++) {
      res += std::lgamma(obs[2 + loopIdx0] + 1) + x;
   }
   return res;
}

// CHECK: void f9_grad_0(double x, const double *obs, double *_d_x) {
// CHECK-NEXT:     int loopIdx0 = 0;
// CHECK-NEXT:     clad::tape<double> _t1 = {};
// CHECK-NEXT:     double _d_res = 0.;
// CHECK-NEXT:     double res = 0.;
// CHECK-NEXT:     unsigned {{int|long|long long}} _t0 = {{0U|0UL|0ULL}};
// CHECK-NEXT:     for (loopIdx0 = 0; ; loopIdx0++) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!(loopIdx0 < 2))
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         clad::push(_t1, res);
// CHECK-NEXT:         res += std::lgamma(obs[2 + loopIdx0] + 1) + x;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     for (;; _t0--) {
// CHECK-NEXT:         {
// CHECK-NEXT:             if (!_t0)
// CHECK-NEXT:                 break;
// CHECK-NEXT:         }
// CHECK-NEXT:         loopIdx0--;
// CHECK-NEXT:         {
// CHECK-NEXT:             res = clad::pop(_t1);
// CHECK-NEXT:             double _r_d0 = _d_res;
// CHECK-NEXT:             *_d_x += _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT: }


void f10_1(double x, double* t){
    t[0] = x;
}

double f10(double x){
    double t[3];
    f10_1(x, t);
    return t[0];
}


// CHECK: void f10_1_pullback(double x, double *t, double *_d_x, double *_d_t) {
// CHECK-NEXT:     double _t0 = t[0];
// CHECK-NEXT:     t[0] = x;
// CHECK-NEXT:     {
// CHECK-NEXT:         t[0] = _t0;
// CHECK-NEXT:         double _r_d0 = _d_t[0];
// CHECK-NEXT:         _d_t[0] = 0.;
// CHECK-NEXT:         *_d_x += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK-NEXT: void f10_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _d_t[3] = {0};
// CHECK-NEXT:     double t[3];
// CHECK-NEXT:     f10_1(x, t);
// CHECK-NEXT:     _d_t[0] += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         f10_1_pullback(x, t, &_r0, _d_t);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double f11_1(double v, double& u){
  u = v;
  return u;
}

double f11(double x){
  double y;
  double c = f11_1(x, y);
  return y;
}

// CHECK: void f11_1_pullback(double v, double &u, double _d_y, double *_d_v, double *_d_u) {
// CHECK-NEXT:     double _t0 = u;
// CHECK-NEXT:     u = v;
// CHECK-NEXT:     *_d_u += _d_y;
// CHECK-NEXT:     {
// CHECK-NEXT:         u = _t0;
// CHECK-NEXT:         double _r_d0 = *_d_u;
// CHECK-NEXT:         *_d_u = 0.;
// CHECK-NEXT:         *_d_v += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }
// CHECK-NEXT: void f11_grad(double x, double *_d_x) {
// CHECK-NEXT:     double _d_y = 0.;
// CHECK-NEXT:     double y;
// CHECK-NEXT:     double _t0 = y;
// CHECK-NEXT:     double _d_c = 0.;
// CHECK-NEXT:     double c = f11_1(x, y);
// CHECK-NEXT:     _d_y += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         y = _t0;
// CHECK-NEXT:         double _r0 = 0.;
// CHECK-NEXT:         f11_1_pullback(x, _t0, _d_c, &_r0, &_d_y);
// CHECK-NEXT:         *_d_x += _r0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double gaussian(double x, double mean, double sigma)
{
   const double arg = x - mean;
   const double sig = sigma;
   return std::exp(-0.5 * arg * arg / (sig * sig));
}

double f12_1(double a, double b){
  double c3 = gaussian(1, 1, b);
  return c3;
}

double f12(double x, double y){
  double c1 = f12_1(x, y);
  double c2 = gaussian(x, y, 1);
  return c1*c2;
}
// CHECK: void gaussian_pullback(double x, double mean, double sigma, double _d_y, double *_d_x, double *_d_mean, double *_d_sigma) {  
// CHECK-NEXT:     double _d_arg = 0.;  
// CHECK-NEXT:     const double arg = x - mean;  
// CHECK-NEXT:     double _d_sig = 0.;  
// CHECK-NEXT:     const double sig = sigma;  
// CHECK-NEXT:     double _t0 = (sig * sig);  
// CHECK-NEXT:     {  
// CHECK-NEXT:         double _r0 = 0.;  
// CHECK-NEXT:         _r0 += _d_y * clad::custom_derivatives::std::exp_pushforward(-0.5 * arg * arg / _t0, 1.).pushforward;  
// CHECK-NEXT:         _d_arg += -0.5 * _r0 / _t0 * arg;  
// CHECK-NEXT:         _d_arg += -0.5 * arg * _r0 / _t0;  
// CHECK-NEXT:         double _r1 = _r0 * -(-0.5 * arg * arg / (_t0 * _t0));  
// CHECK-NEXT:         _d_sig += _r1 * sig;  
// CHECK-NEXT:         _d_sig += sig * _r1;  
// CHECK-NEXT:     }  
// CHECK-NEXT:     *_d_sigma += _d_sig;  
// CHECK-NEXT:     {  
// CHECK-NEXT:         *_d_x += _d_arg;  
// CHECK-NEXT:         *_d_mean += -_d_arg;  
// CHECK-NEXT:     }  
// CHECK-NEXT: }  
// CHECK-NEXT: void f12_1_pullback(double a, double b, double _d_y, double *_d_a, double *_d_b) {  
// CHECK-NEXT:     double _d_c3 = 0.;  
// CHECK-NEXT:     double c3 = gaussian(1, 1, b);  
// CHECK-NEXT:     _d_c3 += _d_y;  
// CHECK-NEXT:     {  
// CHECK-NEXT:         double _r0 = 0.;  
// CHECK-NEXT:         double _r1 = 0.;  
// CHECK-NEXT:         double _r2 = 0.;  
// CHECK-NEXT:         gaussian_pullback(1, 1, b, _d_c3, &_r0, &_r1, &_r2);  
// CHECK-NEXT:         *_d_b += _r2;  
// CHECK-NEXT:     }  
// CHECK-NEXT: }  
// CHECK-NEXT: void f12_grad(double x, double y, double *_d_x, double *_d_y) {  
// CHECK-NEXT:     double _d_c1 = 0.;  
// CHECK-NEXT:     double c1 = f12_1(x, y);  
// CHECK-NEXT:     double _d_c2 = 0.;  
// CHECK-NEXT:     double c2 = gaussian(x, y, 1);  
// CHECK-NEXT:     {  
// CHECK-NEXT:         _d_c1 += 1 * c2;  
// CHECK-NEXT:         _d_c2 += c1 * 1;  
// CHECK-NEXT:     }  
// CHECK-NEXT:     {  
// CHECK-NEXT:         double _r2 = 0.;  
// CHECK-NEXT:         double _r3 = 0.;  
// CHECK-NEXT:         double _r4 = 0.;  
// CHECK-NEXT:         gaussian_pullback(x, y, 1, _d_c2, &_r2, &_r3, &_r4);  
// CHECK-NEXT:         *_d_x += _r2;  
// CHECK-NEXT:         *_d_y += _r3;  
// CHECK-NEXT:     }  
// CHECK-NEXT:     {  
// CHECK-NEXT:         double _r0 = 0.;  
// CHECK-NEXT:         double _r1 = 0.;  
// CHECK-NEXT:         f12_1_pullback(x, y, _d_c1, &_r0, &_r1);  
// CHECK-NEXT:         *_d_x += _r0;  
// CHECK-NEXT:         *_d_y += _r1;  
// CHECK-NEXT:     }  
// CHECK-NEXT: }  

double f13_1(double low, double const* vals)
{
  return low * vals[0];
}

double f13(double x, const double* obs){
  double g = f13_1(1, obs);
  return g;
}

// CHECK: void f13_grad_0(double x, const double *obs, double *_d_x) {
// CHECK-NEXT:    double g = f13_1(1, obs);
// CHECK-NEXT:}


#define TEST1(F, x) { \
  result[0] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_va>(F);\
  F##grad.execute(x, result);\
  printf("{%.2f}\n", result[0]); \
}

#define TEST2(F, x, y) { \
  result[0] = 0; \
  result[1] = 0; \
  auto F##grad = clad::gradient<clad::opts::enable_va>(F);\
  F##grad.execute(x,y, &result[0], &result[1]);\
  printf("{%.2f, %.2f}\n", result[0], result[1]); \
}

int main(){
    double arr[] = {1,2,3,4,5};
    double darr[] = {0,0,0,0,0};
    double result[3] = {};
    double dx = 0;
    TEST1(f1, 3);// CHECK-EXEC: {6.00}
    TEST1(f2, 3);// CHECK-EXEC: {6.00}
    TEST1(f3, 3);// CHECK-EXEC: {0.00}
    TEST1(f4, 3);// CHECK-EXEC: {4.00}
    TEST1(f5, 3);// CHECK-EXEC: {0.00}
    TEST1(f6, 3);// CHECK-EXEC: {0.00}
    TEST1(f7, 3);// CHECK-EXEC: {1.00}
    TEST1(f8, 3);// CHECK-EXEC: {1.00}
    auto grad9 = clad::gradient<clad::opts::enable_va>(f9, "x");
    grad9.execute(3, arr, &dx, darr);
    printf("{%.2f}\n", dx);// CHECK-EXEC: {2.00}
    TEST1(f10, 3);// CHECK-EXEC: {1.00}
    TEST1(f11, 3);// CHECK-EXEC: {1.00}
    TEST2(f12, 3, 1);// CHECK-EXEC: {-0.27, 0.27}
    dx = 0;
    auto grad13 = clad::gradient<clad::opts::enable_va>(f13, "x");
    grad13.execute(3, arr, &dx, darr);
    printf("{%.2f}\n", dx); // CHECK-EXEC: {0.00}
}