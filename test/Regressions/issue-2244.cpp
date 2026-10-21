// RUN: %cladclang %s -I%S/../../include -Xclang -verify -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -DTYPE_DIAGNOSTICS -fsyntax-only -Xclang -verify

#include "clad/Differentiator/Differentiator.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

// A: issue reproducer, decltype of an inactive parameter as a loop variable.
double declA(double x, std::size_t n) {
  double r = 0;
  for (decltype(n) i = 0; i < n; ++i)
    r += x;
  return r;
}

// B: decltype of an active parameter, function scope (no promotion).
double declB(double x) {
  decltype(x) y = x * x;
  return y;
}

// C: decltype of a local inside a loop body (declaration is promoted).
double declC(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    double t = x * i;
    decltype(t) u = t * x;
    r += u;
  }
  return r;
}

// D: nested in pointer / reference / const.
double declD(double x) {
  const decltype(x)& ref = x;
  const decltype(x)* p = &x;
  return ref * *p;
}

// E: GNU typeof.
double declE(double x, int n) {
  double r = 0;
  for (__typeof__(n) i = 0; i < n; ++i)
    r += x * x;
  return r;
}

// F: local typedef of decltype.
double declF(double x, std::size_t n) {
  typedef decltype(n) idx_t;
  double r = 0;
  for (idx_t i = 0; i < n; ++i)
    r += x;
  return r;
}

// G: reached as a pullback from another function.
double declG(double x, std::size_t n) { return declA(x, n) * 2; }

double declI(double x, int n) {
  // Exercise the template argument without invoking a vector constructor.
  std::vector<decltype(n)>* idx = nullptr;
  (void)idx;
  return x * x;
}

double declArray(double x) {
  decltype(x) values[2] = {x, x};
  decltype(x) (*p)[2] = &values;
  return (*p)[0] * (*p)[1];
}

double declRvalue(double x) {
  decltype(x)&& ref = static_cast<decltype(x)&&>(x);
  return ref * ref;
}

double declUsing(double x) {
  using value_t = decltype(x);
  value_t y = x * x;
  return y;
}

double declCast(double x) {
  return (decltype(x))x * decltype(x)(x);
}

double declTypeOfType(double x) {
  __typeof__(decltype(x)) y = x * x;
  return y;
}

double declFunction(double x) {
  using function_t = decltype(x)(double);
  function_t* p = nullptr;
  (void)p;
  return x * x;
}

struct Box {
  using value_type = double;
  double value;
};

double declQualifier(double x) {
  Box box{x};
  decltype(box)::value_type y = box.value * x;
  return y;
}

double declH(double x, decltype(x) y) { return x * y; }

double declVLA(double x, int n) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  double values[n];
  double (*p)[n] = nullptr;
#pragma clang diagnostic pop
  (void)p;
  values[0] = x;
  return values[0] * n;
}

double declNew(double x) {
  auto* p = new decltype(x)(x * x);
  double result = *p;
  delete p;
  return result;
}

double declIncomplete(double x) {
  decltype(x) (*p)[] = nullptr;
  (void)p;
  return x * x;
}

double declNonTypeArgument(double x) {
  std::array<decltype(x), 2>* p = nullptr;
  (void)p;
  return x * x;
}

double declFunctionVLA(double x, int n) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wvla-cxx-extension"
  using Fn = void (*)(double (*)[n]);
#pragma clang diagnostic pop
  Fn p = nullptr;
  (void)p;
  return x;
}

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wvla"
double declVLATypedef(double x, int n) { // expected-note {{declared here}}
  typedef double row_t[n]; // expected-warning {{variable length arrays}} expected-note {{function parameter 'n'}}
  row_t* p = nullptr;
  (void)p;
  return x * x;
}

#pragma clang diagnostic pop

#ifdef TYPE_DIAGNOSTICS
double declAtomicVLA(double x, int n) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"
  _Atomic(double (*)[n])* p = nullptr;
#pragma clang diagnostic pop
  (void)p;
  return x;
}
// expected-error@* {{clad cannot clone expression-bearing type}}
#endif

int main() {
  double dx = 0;
  auto gA = clad::gradient(declA, "x");
  dx = 0;
  gA.execute(3, 4, &dx);
  printf("A %.1f\n", dx); // CHECK-EXEC: A 4.0
  auto gB = clad::gradient(declB);
  dx = 0;
  gB.execute(3, &dx);
  printf("B %.1f\n", dx); // CHECK-EXEC: B 6.0
  auto gC = clad::gradient(declC, "x");
  dx = 0;
  gC.execute(3, 3, &dx);
  printf("C %.1f\n", dx); // CHECK-EXEC: C 18.0
  auto gD = clad::gradient(declD);
  dx = 0;
  gD.execute(3, &dx);
  printf("D %.1f\n", dx); // CHECK-EXEC: D 6.0
  auto gE = clad::gradient(declE, "x");
  dx = 0;
  gE.execute(3, 2, &dx);
  printf("E %.1f\n", dx); // CHECK-EXEC: E 12.0
  auto gF = clad::gradient(declF, "x");
  dx = 0;
  gF.execute(3, 4, &dx);
  printf("F %.1f\n", dx); // CHECK-EXEC: F 4.0
  auto gG = clad::gradient(declG, "x");
  dx = 0;
  gG.execute(3, 4, &dx);
  printf("G %.1f\n", dx); // CHECK-EXEC: G 8.0
  auto gI = clad::gradient(declI, "x");
  dx = 0;
  gI.execute(3, 2, &dx);
  printf("I %.1f\n", dx); // CHECK-EXEC: I 6.0
  auto gArray = clad::gradient(declArray);
  dx = 0;
  gArray.execute(3, &dx);
  printf("Array %.1f\n", dx); // CHECK-EXEC: Array 6.0
  auto fRvalue = clad::differentiate(declRvalue);
  printf("Rvalue %.1f\n", fRvalue.execute(3)); // CHECK-EXEC: Rvalue 6.0
  auto gUsing = clad::gradient(declUsing);
  dx = 0;
  gUsing.execute(3, &dx);
  printf("Using %.1f\n", dx); // CHECK-EXEC: Using 6.0
  auto gCast = clad::gradient(declCast);
  dx = 0;
  gCast.execute(3, &dx);
  printf("Cast %.1f\n", dx); // CHECK-EXEC: Cast 6.0
  auto fCast = clad::differentiate(declCast);
  printf("fwdCast %.1f\n", fCast.execute(3)); // CHECK-EXEC: fwdCast 6.0
  auto gTypeOfType = clad::gradient(declTypeOfType);
  dx = 0;
  gTypeOfType.execute(3, &dx);
  printf("TypeOfType %.1f\n", dx); // CHECK-EXEC: TypeOfType 6.0
  auto fFunction = clad::differentiate(declFunction);
  printf("Function %.1f\n", fFunction.execute(3)); // CHECK-EXEC: Function 6.0
  auto vB = clad::differentiate<clad::opts::vector_mode>(declB);
  dx = 0;
  vB.execute(3, &dx);
  printf("Vector %.1f\n", dx); // CHECK-EXEC: Vector 6.0
  auto gQualifier = clad::gradient(declQualifier);
  dx = 0;
  gQualifier.execute(3, &dx);
  printf("Qualifier %.1f\n", dx); // CHECK-EXEC: Qualifier 6.0
  auto gH = clad::gradient(declH);
  double dy = 0;
  dx = 0;
  gH.execute(3, 4, &dx, &dy);
  printf("H %.1f %.1f\n", dx, dy); // CHECK-EXEC: H 4.0 3.0
  auto fVLA = clad::differentiate(declVLA, "x");
  printf("VLA %.1f\n", fVLA.execute(3, 2)); // CHECK-EXEC: VLA 2.0
  auto fNew = clad::differentiate(declNew);
  printf("New %.1f\n", fNew.execute(3)); // CHECK-EXEC: New 6.0
  auto fIncomplete = clad::differentiate(declIncomplete);
  printf("Incomplete %.1f\n", fIncomplete.execute(3)); // CHECK-EXEC: Incomplete 6.0
  auto fNonTypeArgument = clad::differentiate(declNonTypeArgument);
  printf("NonTypeArgument %.1f\n", fNonTypeArgument.execute(3)); // CHECK-EXEC: NonTypeArgument 6.0
  auto fFunctionVLA = clad::differentiate(declFunctionVLA, "x");
  printf("FunctionVLA %.1f\n", fFunctionVLA.execute(3, 2)); // CHECK-EXEC: FunctionVLA 1.0
  auto fVLATypedef = clad::differentiate(declVLATypedef, "x");
  printf("VLATypedef %.1f\n", fVLATypedef.execute(3, 2)); // CHECK-EXEC: VLATypedef 6.0
#ifdef TYPE_DIAGNOSTICS
  clad::differentiate(declAtomicVLA, "x");
#endif
  auto fA = clad::differentiate(declA, "x");
  printf("fwdA %.1f\n", fA.execute(3, 4)); // CHECK-EXEC: fwdA 4.0
  auto fC = clad::differentiate(declC, "x");
  printf("fwdC %.1f\n", fC.execute(3, 3)); // CHECK-EXEC: fwdC 18.0
}

// CHECK: void declA_grad_0(double x, std::size_t n, double *_d_x) {
// CHECK-NEXT:     std::size_t _d_n = {{0U|0UL|0ULL}};
// CHECK-NEXT:     decltype(n) _d_i = {{0U|0UL|0ULL}};
// CHECK-NEXT:     decltype(n) i = {{0U|0UL|0ULL}};
// CHECK-NEXT:     double _d_r = 0.;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {{std::size_t|unsigned int|unsigned long|unsigned long long}} _t0;
// CHECK-NEXT:     for (i = 0; i < n; ++i) {
// CHECK-NEXT:         r += x;
// CHECK-NEXT:     }
// CHECK-NEXT:     _d_r += 1;
// CHECK-NEXT:     for (_t0 = n > 0 ? ({{std::size_t|unsigned int|unsigned long|unsigned long long}})n : {{0U|0UL|0ULL}}; _t0; _t0--) {
// CHECK-NEXT:         *_d_x += _d_r;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: double declNonTypeArgument_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     std::array<double, 2> *_d_p = nullptr;
// CHECK-NEXT:     std::array<double, 2> *p = nullptr;
// CHECK-NEXT:     return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// CHECK: double declVLATypedef_darg0(double x, int n) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     int _d_n = 0;
// CHECK-NEXT:     typedef double row_t[n];
// CHECK-NEXT:     double (*_d_p)[n] = nullptr;
// CHECK-NEXT:     double (*p)[n] = nullptr;
// CHECK-NEXT:     return _d_x * x + x * _d_x;
// CHECK-NEXT: }

// CHECK: double declA_darg0(double x, std::size_t n) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     std::size_t _d_n = 0;
// CHECK-NEXT:     double _d_r = 0;
// CHECK-NEXT:     double r = 0;
// CHECK-NEXT:     {
// CHECK-NEXT:         decltype(n) _d_i = 0;
// CHECK-NEXT:         for (decltype(n) i = 0; i < n; ++i) {
// CHECK-NEXT:             _d_r += _d_x;
// CHECK-NEXT:             r += x;
// CHECK-NEXT:         }
// CHECK-NEXT:     }
// CHECK-NEXT:     return _d_r;
// CHECK-NEXT: }
