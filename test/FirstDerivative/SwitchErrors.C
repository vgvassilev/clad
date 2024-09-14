// RUN: %cladclang %s -I%S/../../include -fsyntax-only -Xclang -verify 2>&1

#include "clad/Differentiator/Differentiator.h"

double fn1(double i, double j, int choice) {
  double a = 0;
  switch (choice) {
    {
      case 1: // expected-error {{Differentiating switch case label contained in a compound statement, other than the switch statement compound statement, is not supported.}}
        a = i*i;
    }
    a += i;
    {
      {
        case 2: // expected-error {{Differentiating switch case label contained in a compound statement, other than the switch statement compound statement, is not supported.}}
          a = 3 * i;
          break;
        case 3: a = 4 * i;
      }
    }
    a += i;
    break;
  }
  return a;
}

#define INIT(fn, args)\
auto d_##fn = clad::differentiate(fn, args);

int main() {
  INIT(fn1, "i");
}
