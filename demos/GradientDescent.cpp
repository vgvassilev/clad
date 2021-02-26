//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Optimization problem demo using simple iterative Gradient Descent method.
// This method requires gradient definition of the target function,
// it is obtained with clad.
//
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -x c++ -lstdc++ -lm GradientDescent.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -x c++ -lstdc++ -lm GradientDescent.cpp
//

#include <iostream>
#include <cmath>

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

struct Point3D {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  Point3D() : x(0.), y(0.), z(0.) {}
  Point3D(double x, double y, double z) : x(x), y(y), z(z) {}
};

typedef double (*targetT) (double, double, double);
typedef void (*targetGradT) (double, double, double, double*);

class GradientDescent {
public:
  GradientDescent(targetT target, targetGradT targetGrad):
  target(target), targetGrad(targetGrad){
    setParams(Point3D(0, 0, 0));
  }

  bool hasConverged() {
    if (currentStep == 0)
      return false;

    return abs(grad.x) <= tol && abs(grad.y) <= tol && abs(grad.z) <= tol;
  }

  void setParams(Point3D newParams) {
    params = newParams;
    updateGrad();
  }

  Point3D optimize(unsigned maxSteps, bool printInfo) {
    unsigned i = 0;
    while(!hasConverged() && i++ < maxSteps) {
      iterate();

      if (printInfo)
        print();
    }

    return params;
  }

private:
  targetT target;
  targetGradT targetGrad;

  Point3D grad;
  Point3D params;
  unsigned currentStep = 0;

  const double tol = 1e-4;
  const double learningRate = 1e-2;

  void updateGrad() {
    double gradArr[3];
    targetGrad(params.x, params.y, params.z, gradArr);

    grad = Point3D(gradArr[0], gradArr[1], gradArr[2]);
  }

  void iterate() {
    if (hasConverged())
      return;

    setParams(Point3D(
        params.x - learningRate * grad.x,
        params.y - learningRate * grad.y,
        params.z - learningRate * grad.z
    ));

    currentStep += 1;
  }

  void print(){
    double value = target(params.x, params.y, params.z);

    std::cout << "Step #" << currentStep <<
              " x: " << params.x << " y: " << params.y << " z: " << params.z <<
              " value: " << value << std::endl;
  }
};

double f(double x, double y, double z) {
  return (x-1)*(x-1) + (y-2)*(y-2) + (z-3)*(z-3)*(z-3)*(z-3);
}
void f_grad(double, double, double, double*);

int main() {
  clad::gradient(f);

  GradientDescent gd = GradientDescent(f, f_grad);
  auto point = gd.optimize(10000, true);

  std::cout << "Result: " << "(" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;

  return 0;
}
