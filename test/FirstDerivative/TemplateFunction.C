// RUN: %cladclang %s -I%S/../../include  -Xclang -verify 2>&1 | FileCheck %s
//CHECK-NOT: {{.*error:.*}}
#include "clad/Differentiator/Differentiator.h"

template<typename T>
T simple_return(T x) {
  return x;
};

template<typename T>
T addition(T x) {
  return x + x;
}

template<typename T>
T multiplication(T x) {
  return x * x;
}

int main () { // expected-no-diagnostics
  int x;

  diff(simple_return<int>, 1);
  // CHECK: int simple_return_derived_x(int x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  diff(simple_return<float>, 1);
  // CHECK: float simple_return_derived_x(float x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  diff(simple_return<double>, 1);
  // CHECK: double simple_return_derived_x(double x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  diff(addition<int>, 1);
  // CHECK: int addition_derived_x(int x) {
  // CHECK-NEXT: return 1 + (1);
  // CHECK-NEXT: }

  diff(addition<float>, 1);
  // CHECK: float addition_derived_x(float x) {
  // CHECK-NEXT: return 1 + (1);
  // CHECK-NEXT: }

  diff(addition<double>, 1);
  // CHECK: double addition_derived_x(double x) {
  // CHECK-NEXT: return 1 + (1);
  // CHECK-NEXT: }

  diff(multiplication<int>, 1);
  // CHECK: int multiplication_derived_x(int x) {
  // CHECK-NEXT: return (1 * x + x * 1);
  // CHECK-NEXT: }

  diff(multiplication<float>, 1);
  // CHECK: float multiplication_derived_x(float x) {
  // CHECK-NEXT: return (1 * x + x * 1);
  // CHECK-NEXT: }

  diff(multiplication<double>, 1);
  // CHECK: double multiplication_derived_x(double x) {
  // CHECK-NEXT: return (1 * x + x * 1);
  // CHECK-NEXT: }

  return 0;
}
