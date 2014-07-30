// RUN: %cladclang %s -I%S/../../include 2>&1 | FileCheck %s
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
  // CHECK: int simple_return_dx(int x) {
  // CHECK-NEXT: return 1;
  // CHECK-NEXT: }

  clad::differentiate(simple_return<float>, 0);
  // CHECK: float simple_return_dx(float x) {
  // CHECK-NEXT: return 1.F;
  // CHECK-NEXT: }

  clad::differentiate(simple_return<double>, 0);
  // CHECK: double simple_return_dx(double x) {
  // CHECK-NEXT: return 1.;
  // CHECK-NEXT: }

  clad::differentiate(addition<int>, 0);
  // CHECK: int addition_dx(int x) {
  // CHECK-NEXT: return 1 + (1);
  // CHECK-NEXT: }

  clad::differentiate(addition<float>, 0);
  // CHECK: float addition_dx(float x) {
  // CHECK-NEXT: return 1.F + (1.F);
  // CHECK-NEXT: }

  clad::differentiate(addition<double>, 0);
  // CHECK: double addition_dx(double x) {
  // CHECK-NEXT: return 1. + (1.);
  // CHECK-NEXT: }

  clad::differentiate(multiplication<int>, 0);
  // CHECK: int multiplication_dx(int x) {
  // CHECK-NEXT: return (1 * x + x * 1);
  // CHECK-NEXT: }

  clad::differentiate(multiplication<float>, 0);
  // CHECK: float multiplication_dx(float x) {
  // CHECK-NEXT: return (1.F * x + x * 1.F);
  // CHECK-NEXT: }

  clad::differentiate(multiplication<double>, 0);
  // CHECK: double multiplication_dx(double x) {
  // CHECK-NEXT: return (1. * x + x * 1.);
  // CHECK-NEXT: }

  return 0;
}
