// RUN: %cladclang %s -I%S/../../include -Xclang -verify -c

#include "clad/Differentiator/Differentiator.h"

double fn_dangling_checkpoint(double x, double y) {
  #pragma clad checkpoint loop  // expected-error {{'#pragma clad checkpoint loop' is only allowed before a loop}}
  return x + 1;
}

double fn_mixed_checkpoint(double x, double y) {
  #pragma clad checkpoint loop
  for (int i = 0; i < 2; ++i)
    x += y;

  #pragma clad checkpoint loop  // expected-error {{'#pragma clad checkpoint loop' is only allowed before a loop}}
  return x;
}

// The reverse sweep of a checkpointed loop recomputes each iteration from
// the state the previous reverse iteration left; a value one iteration hands
// the next is not there to recompute from.
double fn_carried(double x) {
  double t = 1, s = 0;
  #pragma clad checkpoint loop
  for (int i = 0; i < 3; i++) { // expected-error {{'#pragma clad checkpoint loop' recomputes each iteration of this loop from the values before it, but 't' carries a value from one iteration into the next; the gradient would be wrong}}
    t = t * x; // expected-note {{'t' is read here before the iteration assigns it}}
    s += t;
  }
  return s;
}

// A product's adjoint reads the running product, so it carries too.
double fn_product(double x) {
  double p = 1;
  #pragma clad checkpoint loop
  for (int i = 0; i < 3; i++) // expected-error {{'#pragma clad checkpoint loop' recomputes each iteration of this loop from the values before it, but 'p' carries a value from one iteration into the next; the gradient would be wrong}}
    p *= x; // expected-note {{'p' is read here before the iteration assigns it}}
  return p;
}

// A `while` is not a counted loop, so it has no trip count to prove, but it
// carries values the same way and the analysis says so.
double fn_carried_while(double x) {
  double t = 1, s = 0;
  int i = 0;
  #pragma clad checkpoint loop
  while (i < 3) { // expected-error {{'#pragma clad checkpoint loop' recomputes each iteration of this loop from the values before it, but 't' carries a value from one iteration into the next; the gradient would be wrong}}
    t = t * x; // expected-note {{'t' is read here before the iteration assigns it}}
    s += t;
    i++;
  }
  return s;
}

// A `do` carries the same way, and its body is the loop's own statement
// rather than a compound one.
double fn_carried_do(double x) {
  double t = 1, s = 0;
  int i = 0;
  #pragma clad checkpoint loop
  do { // expected-error {{'#pragma clad checkpoint loop' recomputes each iteration of this loop from the values before it, but 't' carries a value from one iteration into the next; the gradient would be wrong}}
    t = t * x; // expected-note {{'t' is read here before the iteration assigns it}}
    s += t;
    i++;
  } while (i < 3);
  return s;
}

// Recomputable: a sum is never read by its own adjoint, a value assigned at
// the top of the body is the iteration's own, and so is one declared there;
// the index is the loop's.
double fn_recomputable(double x) {
  double t = 0, s = 0;
  #pragma clad checkpoint loop
  for (int i = 0; i < 3; i++) {
    t = x * i;
    double u = t + x;
    s += t * u;
  }
  return s;
}

int main() {
  // Hits duplicate pragma-diagnosis suppression path.
  clad::gradient(fn_dangling_checkpoint);
  clad::hessian(fn_dangling_checkpoint, "x");

  // Hits reverse loop checkpoint scan with one invalid entry in map.
  clad::gradient(fn_mixed_checkpoint);

  clad::gradient(fn_carried);
  clad::gradient(fn_product);
  clad::gradient(fn_carried_while);
  clad::gradient(fn_carried_do);
  clad::gradient(fn_recomputable);
}
