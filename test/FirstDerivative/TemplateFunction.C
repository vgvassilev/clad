// RUN: %cladclang %s -I%S/../../include 
//CHECK-NOT: {{.*error|warning|note:.*}}

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

int main () {
  int x;

  clad::differentiate(simple_return<int>, 0);
  // CHECK: int simple_return_darg0(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  clad::differentiate(simple_return<float>, 0);
  // CHECK: float simple_return_darg0(float x) {
  // CHECK-NEXT: float _d_x = 1;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  clad::differentiate(simple_return<double>, 0);
  // CHECK: double simple_return_darg0(double x) {
  // CHECK-NEXT: double _d_x = 1;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  clad::differentiate(addition<int>, 0);
  // CHECK: int addition_darg0(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: return _d_x + _d_x;
  // CHECK-NEXT: }

  clad::differentiate(addition<float>, 0);
  // CHECK: float addition_darg0(float x) {
  // CHECK-NEXT: float _d_x = 1;
  // CHECK-NEXT: return _d_x + _d_x;
  // CHECK-NEXT: }

  clad::differentiate(addition<double>, 0);
  // CHECK: double addition_darg0(double x) {
  // CHECK-NEXT: double _d_x = 1;
  // CHECK-NEXT: return _d_x + _d_x;
  // CHECK-NEXT: }

  clad::differentiate(multiplication<int>, 0);
  // CHECK: int multiplication_darg0(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: return _d_x * x + x * _d_x;
  // CHECK-NEXT: }

  clad::differentiate(multiplication<float>, 0);
  // CHECK: float multiplication_darg0(float x) {
  // CHECK-NEXT: float _d_x = 1;
  // CHECK-NEXT: return _d_x * x + x * _d_x;
  // CHECK-NEXT: }

  clad::differentiate(multiplication<double>, 0);
  // CHECK: double multiplication_darg0(double x) {
  // CHECK: double _d_x = 1;
  // CHECK-NEXT: return _d_x * x + x * _d_x;
  // CHECK-NEXT: }

  return 0;
}
