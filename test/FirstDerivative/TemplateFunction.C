// RUN: %cladclang %s -I%S/../../include 2>&1 | %filecheck %s

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
  // CHECK: template<> int simple_return_darg0<int>(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  clad::differentiate(simple_return<float>, 0);
  // CHECK: template<> float simple_return_darg0<float>(float x) {
  // CHECK-NEXT: float _d_x = 1;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  clad::differentiate(simple_return<double>, 0);
  // CHECK: template<> double simple_return_darg0<double>(double x) {
  // CHECK-NEXT: double _d_x = 1;
  // CHECK-NEXT: return _d_x;
  // CHECK-NEXT: }

  clad::differentiate(addition<int>, 0);
  // CHECK: template<> int addition_darg0<int>(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: return _d_x + _d_x;
  // CHECK-NEXT: }

  clad::differentiate(addition<float>, 0);
  // CHECK: template<> float addition_darg0<float>(float x) {
  // CHECK-NEXT: float _d_x = 1;
  // CHECK-NEXT: return _d_x + _d_x;
  // CHECK-NEXT: }

  clad::differentiate(addition<double>, 0);
  // CHECK: template<> double addition_darg0<double>(double x) {
  // CHECK-NEXT: double _d_x = 1;
  // CHECK-NEXT: return _d_x + _d_x;
  // CHECK-NEXT: }

  clad::differentiate(multiplication<int>, 0);
  // CHECK: template<> int multiplication_darg0<int>(int x) {
  // CHECK-NEXT: int _d_x = 1;
  // CHECK-NEXT: return _d_x * x + x * _d_x;
  // CHECK-NEXT: }

  clad::differentiate(multiplication<float>, 0);
  // CHECK: template<> float multiplication_darg0<float>(float x) {
  // CHECK-NEXT: float _d_x = 1;
  // CHECK-NEXT: return _d_x * x + x * _d_x;
  // CHECK-NEXT: }

  clad::differentiate(multiplication<double>, 0);
  // CHECK: template<> double multiplication_darg0<double>(double x) {
  // CHECK: double _d_x = 1;
  // CHECK-NEXT: return _d_x * x + x * _d_x;
  // CHECK-NEXT: }

  return 0;
}
