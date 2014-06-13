#include "clad/Differentiator/Differentiator.h"

struct S {
  float g(int x) { return x * x * x; }
};

int pow2(int x) { return x * x; }

int main() {
  // Differentiate pow2. Clad will define a function named pow2_derived_x(int)
  // with the derivative, ready to be called.
  auto cladPow2 = clad::differentiate(pow2, 0);
  // Get the 1st derivative of pow2 and call it with 12.
  int pow1stOrderDerivative = clad::differentiate(pow2, 0).execute(12);

  // Or we can call like this:
  pow1stOrderDerivative = cladPow.execute();

  auto fnPtr = clad::differentiate<5>(pow2, 0).getFunctionPtr();
  fnPtr(1);

  auto cladSPow2 = clad::differentiate(&S::g, 1);


  // IfStmts

  // Loops

  // Params

  // Lambda

  return 0;
}
