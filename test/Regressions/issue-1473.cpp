// RUN: %cladclang -I%S/../../include %s

#include "clad/Differentiator/Differentiator.h"

union b {
};

b c(int d, int e, int f) {
  b g;
  g = {};
  return g;
}
int h;
void f1(int j) {
  int d;
  int f;
  c(d, h, f);
}

union U {
  float f;
  int i;
};

double f2(float j) {
  U u{1.f};
  u.f = j;
  return u.f;
}

int main() {
   clad::gradient(f1);
   clad::gradient(f2);
}
