// RUN: %cladclang -fsyntax-only -Xclang -verify %s

double fn_incorrect_checkpoint(double x, double y) {
  double sum = 0;
  #pragma clad checkpoint  // expected-error {{expected 'loop' after 'checkpoint' in #pragma clad}}
  for (int i = 0; i < 100; ++i) {
    double t1 = x;
    double t2 = y / t1;
    sum += t2;
  }
  #pragma clad checkpoint other  // expected-error {{expected 'loop' after 'checkpoint' in #pragma clad}}
  while (false) {}

  return sum;
}