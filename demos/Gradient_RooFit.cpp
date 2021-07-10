//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to calculate gradient/normal vector of
// implicit function.
//
// author:  Alexander Penev <alexander_penev-at-yahoo.com>
// author:  Patrick Bos <p.bos-at-esciencecenter.nl>
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -I $ROOTSYS/include -x c++ -std=c++11 \
// Gradient_RooFit.cpp -o Gradient_RooFit
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -I $ROOTSYS/include -x c++ -std=c++11 Gradient_RooFit.cpp \
// -o Gradient_RooFit

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooGaussian.h"

// Implicit function for sphere
float roofit_gauss(float _x) {
  RooRealVar x("x", "x", _x);
  RooConstVar mean("mean", "Mean of Gaussian", 0) ;
  RooConstVar sigma("sigma","Width of Gaussian",3) ;
  RooGaussian gauss("gauss","gauss(x,mean,sigma)", x, mean, sigma) ;

  return gauss.GetVal();
}

int main() {
  // Differentiate implicit sphere function. Clad will produce the three derivatives
  // of function roofit_gauss
  auto roofit_gauss_dx = clad::differentiate(roofit_gauss, 0);

  // Calculate gradient in point P (Normal vector)
  float grad_0 = roofit_gauss_dx.execute(0);
  printf("Results are %f, \n", grad_0);
}
