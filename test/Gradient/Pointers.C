// RUN: %cladclang %s -I%S/../../include -oPointers.out 2>&1 | FileCheck %s
// RUN: ./Pointers.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}

// FIXME: This test does not work with enable-tbr flag, because the
// current implementation of TBR analysis doesn't support pointers.
// XFAIL: target={{i586.*}}

#include "clad/Differentiator/Differentiator.h"

double nonMemFn(double i) {
  return i*i;
}
// CHECK: void nonMemFn_grad(double i, double *_d_i) {
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_i += 1 * i;
// CHECK-NEXT:         *_d_i += i * 1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double minimalPointer(double x) {
  double* const p = &x;
  *p = (*p)*(*p);
  return *p; // x*x
}

// CHECK: void minimalPointer_grad(double x, double *_d_x) {
// CHECK-NEXT:     double *_d_p = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     _d_p = &*_d_x;
// CHECK-NEXT:     double *const p = &x;
// CHECK-NEXT:     _t0 = *p;
// CHECK-NEXT:     *p = *p * (*p);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     *_d_p += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         *p = _t0;
// CHECK-NEXT:         double _r_d0 = *_d_p;
// CHECK-NEXT:         *_d_p -= _r_d0;
// CHECK-NEXT:         *_d_p += _r_d0 * (*p);
// CHECK-NEXT:         *_d_p += *p * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double arrayPointer(const double* arr) {
  const double *p = arr;
  p = p + 1;
  double sum = *p;
  p++;
  sum += (*p)*2;
  p += 1;
  sum += (*p)*4;
  ++p;
  sum += (*p)*3;
  p -= 2;
  p = p - 2;
  sum += 5 * (*p);
  return sum; // 5*arr[0] + arr[1] + 2*arr[2] + 4*arr[3] + 3*arr[4]
}

// CHECK: void arrayPointer_grad(const double *arr, double *_d_arr) {
// CHECK-NEXT:     double *_d_p = 0;
// CHECK-NEXT:     const double *_t0;
// CHECK-NEXT:     double *_t1;
// CHECK-NEXT:     double _d_sum = 0;
// CHECK-NEXT:     double _t2;
// CHECK-NEXT:     const double *_t3;
// CHECK-NEXT:     double *_t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     const double *_t7;
// CHECK-NEXT:     double *_t8;
// CHECK-NEXT:     const double *_t9;
// CHECK-NEXT:     double *_t10;
// CHECK-NEXT:     double _t11;
// CHECK-NEXT:     _d_p = _d_arr;
// CHECK-NEXT:     const double *p = arr;
// CHECK-NEXT:     _t0 = p;
// CHECK-NEXT:     _t1 = _d_p;
// CHECK-NEXT:     _d_p = _d_p + 1;
// CHECK-NEXT:     p = p + 1;
// CHECK-NEXT:     double sum = *p;
// CHECK-NEXT:     _d_p++;
// CHECK-NEXT:     p++;
// CHECK-NEXT:     _t2 = sum;
// CHECK-NEXT:     sum += *p * 2;
// CHECK-NEXT:     _t3 = p;
// CHECK-NEXT:     _t4 = _d_p;
// CHECK-NEXT:     _d_p += 1;
// CHECK-NEXT:     p += 1;
// CHECK-NEXT:     _t5 = sum;
// CHECK-NEXT:     sum += *p * 4;
// CHECK-NEXT:     ++_d_p;
// CHECK-NEXT:     ++p;
// CHECK-NEXT:     _t6 = sum;
// CHECK-NEXT:     sum += *p * 3;
// CHECK-NEXT:     _t7 = p;
// CHECK-NEXT:     _t8 = _d_p;
// CHECK-NEXT:     _d_p -= 2;
// CHECK-NEXT:     p -= 2;
// CHECK-NEXT:     _t9 = p;
// CHECK-NEXT:     _t10 = _d_p;
// CHECK-NEXT:     _d_p = _d_p - 2;
// CHECK-NEXT:     p = p - 2;
// CHECK-NEXT:     _t11 = sum;
// CHECK-NEXT:     sum += 5 * (*p);
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t11;
// CHECK-NEXT:         double _r_d3 = _d_sum;
// CHECK-NEXT:         *_d_p += 5 * _r_d3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         p = _t9;
// CHECK-NEXT:         _d_p = _t10;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         p = _t7;
// CHECK-NEXT:         _d_p = _t8;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t6;
// CHECK-NEXT:         double _r_d2 = _d_sum;
// CHECK-NEXT:         *_d_p += _r_d2 * 3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         --p;
// CHECK-NEXT:         --_d_p;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t5;
// CHECK-NEXT:         double _r_d1 = _d_sum;
// CHECK-NEXT:         *_d_p += _r_d1 * 4;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         p = _t3;
// CHECK-NEXT:         _d_p = _t4;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t2;
// CHECK-NEXT:         double _r_d0 = _d_sum;
// CHECK-NEXT:         *_d_p += _r_d0 * 2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         p--;
// CHECK-NEXT:         _d_p--;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_p += _d_sum;
// CHECK-NEXT:     {
// CHECK-NEXT:         p = _t0;
// CHECK-NEXT:         _d_p = _t1;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double pointerParam(const double* arr, size_t n) {
  double sum = 0;
  for (size_t i=0; i < n; ++i) {
    size_t* j = &i;
    sum += arr[0] * (*j);
    arr = arr + 1;
  }
  return sum;
}

// CHECK: void pointerParam_grad_0(const double *arr, size_t n, double *_d_arr) {
// CHECK-NEXT:     size_t _d_n = 0;
// CHECK-NEXT:     double _d_sum = 0;
// CHECK-NEXT:     unsigned {{int|long}} _t0;
// CHECK-NEXT:     size_t _d_i = 0;
// CHECK-NEXT:     size_t i = 0;
// CHECK-NEXT:     clad::tape<size_t *> _t1 = {};
// CHECK-NEXT:     clad::tape<size_t *> _t3 = {};
// CHECK-NEXT:     size_t *_d_j = 0;
// CHECK-NEXT:     size_t *j = 0;
// CHECK-NEXT:     clad::tape<double> _t4 = {};
// CHECK-NEXT:     clad::tape<const double *> _t5 = {};
// CHECK-NEXT:     clad::tape<double *> _t6 = {};
// CHECK-NEXT:     double sum = 0;
// CHECK-NEXT:     _t0 = 0;
// CHECK-NEXT:     for (i = 0; i < n; ++i) {
// CHECK-NEXT:         _t0++;
// CHECK-NEXT:         _d_j = &_d_i;
// CHECK-NEXT:         clad::push(_t1, _d_j);
// CHECK-NEXT:         clad::push(_t3, j) , j = &i;
// CHECK-NEXT:         clad::push(_t4, sum);
// CHECK-NEXT:         sum += arr[0] * (*j);
// CHECK-NEXT:         clad::push(_t5, arr);
// CHECK-NEXT:         clad::push(_t6, _d_arr);
// CHECK-NEXT:         _d_arr = _d_arr + 1;
// CHECK-NEXT:         arr = arr + 1;
// CHECK-NEXT:     }
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     for (; _t0; _t0--) {
// CHECK-NEXT:         --i;
// CHECK-NEXT:         size_t *_t2 = clad::pop(_t1);
// CHECK-NEXT:         {
// CHECK-NEXT:             arr = clad::pop(_t5);
// CHECK-NEXT:             _d_arr = clad::pop(_t6);
// CHECK-NEXT:         }
// CHECK-NEXT:         {
// CHECK-NEXT:             sum = clad::pop(_t4);
// CHECK-NEXT:             double _r_d0 = _d_sum;
// CHECK-NEXT:             _d_arr[0] += _r_d0 * (*j);
// CHECK-NEXT:             *_t2 += arr[0] * _r_d0;
// CHECK-NEXT:         }
// CHECK-NEXT:         j = clad::pop(_t3);
// CHECK-NEXT:     }
// CHECK-NEXT: }

double pointerMultipleParams(const double* a, const double* b) {
  double sum = b[2];
  b = a;
  a = 1+a;
  ++b;
  sum += a[0] + b[0]; // += 2*a[1]
  b++; a++;
  sum += a[0] + b[0]; // += 2*a[2]
  b--; a--;
  sum += a[0] + b[0]; // += 2*a[1]
  --b; --a;
  sum += a[0] + b[0]; // += 2*a[0]
  return sum; // 2*a[0] + 4*a[1] + 2*a[2] + b[2]
}

// CHECK: void pointerMultipleParams_grad(const double *a, const double *b, double *_d_a, double *_d_b) {
// CHECK-NEXT:     double _d_sum = 0;
// CHECK-NEXT:     const double *_t0;
// CHECK-NEXT:     double *_t1;
// CHECK-NEXT:     const double *_t2;
// CHECK-NEXT:     double *_t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     double _t6;
// CHECK-NEXT:     double _t7;
// CHECK-NEXT:     double sum = b[2];
// CHECK-NEXT:     _t0 = b;
// CHECK-NEXT:     _t1 = _d_b;
// CHECK-NEXT:     _d_b = _d_a;
// CHECK-NEXT:     b = a;
// CHECK-NEXT:     _t2 = a;
// CHECK-NEXT:     _t3 = _d_a;
// CHECK-NEXT:     _d_a = 1 + _d_a;
// CHECK-NEXT:     a = 1 + a;
// CHECK-NEXT:     ++_d_b;
// CHECK-NEXT:     ++b;
// CHECK-NEXT:     _t4 = sum;
// CHECK-NEXT:     sum += a[0] + b[0];
// CHECK-NEXT:     _d_b++;
// CHECK-NEXT:     b++;
// CHECK-NEXT:     _d_a++;
// CHECK-NEXT:     a++;
// CHECK-NEXT:     _t5 = sum;
// CHECK-NEXT:     sum += a[0] + b[0];
// CHECK-NEXT:     _d_b--;
// CHECK-NEXT:     b--;
// CHECK-NEXT:     _d_a--;
// CHECK-NEXT:     a--;
// CHECK-NEXT:     _t6 = sum;
// CHECK-NEXT:     sum += a[0] + b[0];
// CHECK-NEXT:     --_d_b;
// CHECK-NEXT:     --b;
// CHECK-NEXT:     --_d_a;
// CHECK-NEXT:     --a;
// CHECK-NEXT:     _t7 = sum;
// CHECK-NEXT:     sum += a[0] + b[0];
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t7;
// CHECK-NEXT:         double _r_d3 = _d_sum;
// CHECK-NEXT:         _d_a[0] += _r_d3;
// CHECK-NEXT:         _d_b[0] += _r_d3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         ++a;
// CHECK-NEXT:         ++_d_a;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         ++b;
// CHECK-NEXT:         ++_d_b;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t6;
// CHECK-NEXT:         double _r_d2 = _d_sum;
// CHECK-NEXT:         _d_a[0] += _r_d2;
// CHECK-NEXT:         _d_b[0] += _r_d2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         a++;
// CHECK-NEXT:         _d_a++;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         b++;
// CHECK-NEXT:         _d_b++;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t5;
// CHECK-NEXT:         double _r_d1 = _d_sum;
// CHECK-NEXT:         _d_a[0] += _r_d1;
// CHECK-NEXT:         _d_b[0] += _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         a--;
// CHECK-NEXT:         _d_a--;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         b--;
// CHECK-NEXT:         _d_b--;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         sum = _t4;
// CHECK-NEXT:         double _r_d0 = _d_sum;
// CHECK-NEXT:         _d_a[0] += _r_d0;
// CHECK-NEXT:         _d_b[0] += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         --b;
// CHECK-NEXT:         --_d_b;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         a = _t2;
// CHECK-NEXT:         _d_a = _t3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         b = _t0;
// CHECK-NEXT:         _d_b = _t1;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_b[2] += _d_sum;
// CHECK-NEXT: }

double newAndDeletePointer(double i, double j) {
  double *p = new double(i);
  double *q = new double(j);
  double *r = new double[2];
  r[0] = i + j;
  r[1] = i*j;
  double sum = *p + *q + r[0] + r[1];
  delete p;
  delete q;
  delete [] r;
  return sum;
}

// CHECK: void newAndDeletePointer_grad(double i, double j, double *_d_i, double *_d_j) {
// CHECK-NEXT:     double *_d_p = 0;
// CHECK-NEXT:     double *_d_q = 0;
// CHECK-NEXT:     double *_d_r = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _d_sum = 0;
// CHECK-NEXT:     _d_p = new double(*_d_i);
// CHECK-NEXT:     double *p = new double(i);
// CHECK-NEXT:     _d_q = new double(*_d_j);
// CHECK-NEXT:     double *q = new double(j);
// CHECK-NEXT:     _d_r = new double [2](/*implicit*/(double{{[ ]?}}[2])0);
// CHECK-NEXT:     double *r = new double [2];
// CHECK-NEXT:     _t0 = r[0];
// CHECK-NEXT:     r[0] = i + j;
// CHECK-NEXT:     _t1 = r[1];
// CHECK-NEXT:     r[1] = i * j;
// CHECK-NEXT:     double sum = *p + *q + r[0] + r[1];
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_sum += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_p += _d_sum;
// CHECK-NEXT:         *_d_q += _d_sum;
// CHECK-NEXT:         _d_r[0] += _d_sum;
// CHECK-NEXT:         _d_r[1] += _d_sum;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         r[1] = _t1;
// CHECK-NEXT:         double _r_d1 = _d_r[1];
// CHECK-NEXT:         _d_r[1] -= _r_d1;
// CHECK-NEXT:         *_d_i += _r_d1 * j;
// CHECK-NEXT:         *_d_j += i * _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         r[0] = _t0;
// CHECK-NEXT:         double _r_d0 = _d_r[0];
// CHECK-NEXT:         _d_r[0] -= _r_d0;
// CHECK-NEXT:         *_d_i += _r_d0;
// CHECK-NEXT:         *_d_j += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     *_d_j += *_d_q;
// CHECK-NEXT:     *_d_i += *_d_p;
// CHECK-NEXT:     delete p;
// CHECK-NEXT:     delete _d_p;
// CHECK-NEXT:     delete q;
// CHECK-NEXT:     delete _d_q;
// CHECK-NEXT:     delete [] r;
// CHECK-NEXT:     delete [] _d_r;
// CHECK-NEXT: }

struct T {
  double x;
  int y;
};

double structPointer (double x) {
  T* t = new T{x};
  double res = t->x;
  delete t;
  return res;
}

// CHECK: void structPointer_grad(double x, double *_d_x) {
// CHECK-NEXT:     T *_d_t = 0;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     _d_t = new T();
// CHECK-NEXT:     T *t = new T({x, /*implicit*/(int)0});
// CHECK-NEXT:     double res = t->x;
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     _d_t->x += _d_res;
// CHECK-NEXT:     *_d_x += *_d_t.x;
// CHECK-NEXT:     delete t;
// CHECK-NEXT:     delete _d_t;
// CHECK-NEXT: }

double cStyleMemoryAlloc(double x, size_t n) {
  T* t = (T*)malloc(n * sizeof(T));
  memset(t, 0, n * sizeof(T));
  t->x = x;
  double* p = (double*)calloc(1, sizeof(double));
  *p = x;
  double res = t->x + *p;
  p = (double*)realloc(p, 2*sizeof(double));
  p[1] = 2*x;
  res += p[1];
  free(p);
  free(t);
  return res;
}

// CHECK: void cStyleMemoryAlloc_grad_0(double x, size_t n, double *_d_x) {
// CHECK-NEXT:     size_t _d_n = 0;
// CHECK-NEXT:     T *_d_t = 0;
// CHECK-NEXT:     double _t0;
// CHECK-NEXT:     double *_d_p = 0;
// CHECK-NEXT:     double _t1;
// CHECK-NEXT:     double _d_res = 0;
// CHECK-NEXT:     double *_t2;
// CHECK-NEXT:     double *_t3;
// CHECK-NEXT:     double _t4;
// CHECK-NEXT:     double _t5;
// CHECK-NEXT:     _d_t = (T *)malloc(n * sizeof(T));
// CHECK-NEXT:     T *t = (T *)malloc(n * sizeof(T));
// CHECK-NEXT:     memset(_d_t, 0, n * sizeof(T));
// CHECK-NEXT:     memset(t, 0, n * sizeof(T));
// CHECK-NEXT:     _t0 = t->x;
// CHECK-NEXT:     t->x = x;
// CHECK-NEXT:     _d_p = (double *)calloc(1, sizeof(double));
// CHECK-NEXT:     double *p = (double *)calloc(1, sizeof(double));
// CHECK-NEXT:     _t1 = *p;
// CHECK-NEXT:     *p = x;
// CHECK-NEXT:     double res = t->x + *p;
// CHECK-NEXT:     _t2 = p;
// CHECK-NEXT:     _t3 = _d_p;
// CHECK-NEXT:     _d_p = (double *)realloc(_d_p, 2 * sizeof(double));
// CHECK-NEXT:     p = (double *)realloc(p, 2 * sizeof(double));
// CHECK-NEXT:     _t4 = p[1];
// CHECK-NEXT:     p[1] = 2 * x;
// CHECK-NEXT:     _t5 = res;
// CHECK-NEXT:     res += p[1];
// CHECK-NEXT:     goto _label0;
// CHECK-NEXT:   _label0:
// CHECK-NEXT:     _d_res += 1;
// CHECK-NEXT:     {
// CHECK-NEXT:         res = _t5;
// CHECK-NEXT:         double _r_d3 = _d_res;
// CHECK-NEXT:         _d_p[1] += _r_d3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         p[1] = _t4;
// CHECK-NEXT:         double _r_d2 = _d_p[1];
// CHECK-NEXT:         _d_p[1] -= _r_d2;
// CHECK-NEXT:         *_d_x += 2 * _r_d2;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         p = _t2;
// CHECK-NEXT:         _d_p = _t3;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_t->x += _d_res;
// CHECK-NEXT:         *_d_p += _d_res;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         *p = _t1;
// CHECK-NEXT:         double _r_d1 = *_d_p;
// CHECK-NEXT:         *_d_p -= _r_d1;
// CHECK-NEXT:         *_d_x += _r_d1;
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK-NEXT:         t->x = _t0;
// CHECK-NEXT:         double _r_d0 = _d_t->x;
// CHECK-NEXT:         _d_t->x -= _r_d0;
// CHECK-NEXT:         *_d_x += _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT:     free(p);
// CHECK-NEXT:     free(_d_p);
// CHECK-NEXT:     free(t);
// CHECK-NEXT:     free(_d_t);
// CHECK-NEXT: }

#define NON_MEM_FN_TEST(var)\
res[0]=0;\
var.execute(5,res);\
printf("%.2f\n", res[0]);

int main() {
  auto nonMemFnPtr = &nonMemFn;
  auto nonMemFnPtrToPtr = &nonMemFnPtr;
  auto nonMemFnPtrToPtrToPtr = &nonMemFnPtrToPtr;
  auto nonMemFnIndirectPtr = nonMemFnPtr;
  auto nonMemFnIndirectIndirectPtr = nonMemFnIndirectPtr;

  double res[2];

  auto d_nonMemFn = clad::gradient(nonMemFn, "i");
  auto d_nonMemFnPar = clad::gradient((nonMemFn), "i");
  auto d_nonMemFnPtr = clad::gradient(nonMemFnPtr, "i");
  auto d_nonMemFnPtrToPtr = clad::gradient(*nonMemFnPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrPar = clad::gradient((*(nonMemFnPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtr_1 = clad::gradient(**&nonMemFnPtrToPtr, "i");
  auto d_nonMemFnPtrToPtr_1Par = clad::gradient(**(&nonMemFnPtrToPtr), "i");
  auto d_nonMemFnPtrToPtr_1ParPar = clad::gradient(*(*(&nonMemFnPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtrToPtr = clad::gradient(**nonMemFnPtrToPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrToPtr_1 = clad::gradient(***&nonMemFnPtrToPtrToPtr, "i");
  auto d_nonMemFnPtrToPtrToPtr_1Par = clad::gradient(***(&nonMemFnPtrToPtrToPtr), "i");
  auto d_nonMemFnPtrToPtrToPtr_1ParPar = clad::gradient(*(**(&nonMemFnPtrToPtrToPtr)), "i");
  auto d_nonMemFnPtrToPtrToPtr_1ParParPar = clad::gradient((*(**((&nonMemFnPtrToPtrToPtr)))), "i");
  auto d_nonMemFnIndirectPtr = clad::gradient(nonMemFnIndirectPtr, "i");
  auto d_nonMemFnIndirectIndirectPtr = clad::gradient(nonMemFnIndirectIndirectPtr, "i");
  auto d_nonMemFnStaticCast = clad::gradient(static_cast<decltype(&nonMemFn)>(nonMemFn), "i");
  auto d_nonMemFnReinterpretCast = clad::gradient(reinterpret_cast<decltype(&nonMemFn)>(nonMemFn), "i");
  auto d_nonMemFnCStyleCast = clad::gradient((decltype(&nonMemFn))(nonMemFn), "i");


  NON_MEM_FN_TEST(d_nonMemFn); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1Par); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtr_1ParPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1Par); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnPtrToPtrToPtr_1ParParPar); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnIndirectPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnIndirectIndirectPtr); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnStaticCast); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnReinterpretCast); // CHECK-EXEC: 10.00

  NON_MEM_FN_TEST(d_nonMemFnCStyleCast); // CHECK-EXEC: 10.00

  // Pointer operation tests.
  auto d_minimalPointer = clad::gradient(minimalPointer, "x");
  NON_MEM_FN_TEST(d_minimalPointer); // CHECK-EXEC: 10.00

  auto d_arrayPointer = clad::gradient(arrayPointer, "arr");
  double arr[5] = {1, 2, 3, 4, 5};
  double d_arr[5] = {0, 0, 0, 0, 0};
  d_arrayPointer.execute(arr, d_arr);
  printf("%.2f %.2f %.2f %.2f %.2f\n", d_arr[0], d_arr[1], d_arr[2], d_arr[3], d_arr[4]); // CHECK-EXEC: 5.00 1.00 2.00 4.00 3.00

  auto d_pointerParam = clad::gradient(pointerParam, "arr");
  d_arr[0] = d_arr[1] = d_arr[2] = d_arr[3] = d_arr[4] = 0;
  d_pointerParam.execute(arr, 5, d_arr);
  printf("%.2f %.2f %.2f %.2f %.2f\n", d_arr[0], d_arr[1], d_arr[2], d_arr[3], d_arr[4]); // CHECK-EXEC: 0.00 1.00 2.00 3.00 4.00

  auto d_pointerMultipleParams = clad::gradient(pointerMultipleParams);
  double b_arr[5] = {1, 2, 3, 4, 5};
  double d_b_arr[5] = {0, 0, 0, 0, 0};
  d_arr[0] = d_arr[1] = d_arr[2] = d_arr[3] = d_arr[4] = 0;
  d_pointerMultipleParams.execute(arr, b_arr, d_arr, d_b_arr);
  printf("%.2f %.2f %.2f %.2f %.2f\n", d_arr[0], d_arr[1], d_arr[2], d_arr[3], d_arr[4]); // CHECK-EXEC: 2.00 4.00 2.00 0.00 0.00
  printf("%.2f %.2f %.2f %.2f %.2f\n", d_b_arr[0], d_b_arr[1], d_b_arr[2], d_b_arr[3], d_b_arr[4]); // CHECK-EXEC: 0.00 0.00 1.00 0.00 0.00

  auto d_newAndDeletePointer = clad::gradient(newAndDeletePointer);
  double d_i = 0, d_j = 0;
  d_newAndDeletePointer.execute(5, 7, &d_i, &d_j);
  printf("%.2f %.2f\n", d_i, d_j); // CHECK-EXEC: 9.00 7.00

  auto d_structPointer = clad::gradient(structPointer);
  double d_x = 0;
  d_structPointer.execute(5, &d_x);
  printf("%.2f\n", d_x); // CHECK-EXEC: 1.00

  auto d_cStyleMemoryAlloc = clad::gradient<clad::opts::disable_tbr>(cStyleMemoryAlloc, "x");
  d_x = 0;
  d_cStyleMemoryAlloc.execute(5, 7, &d_x);
  printf("%.2f\n", d_x); // CHECK-EXEC: 4.00
}
