// RUN: %autodiff %s -I%S/../../include -fsyntax-only 2>&1 | FileCheck %s

#include "autodiff/Differentiator/Differentiator.h"

int f_1(int x) {
  return x * x;
}

int f_2(int x) {
  return (1 * x + x * 1);
}

int f_3(int x) {
  return 1 * x;
}

int f_4(int x) {
  return (0 * x + 1 * 1);
}

int main () {
  int x = 4;
  diff(f_1, 1);
  diff(f_2, 1);
  diff(f_3, 1);
  diff(f_4, 1);
  return 0;
}