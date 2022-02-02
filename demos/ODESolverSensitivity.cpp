//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Sensitivity evaluation of Runge-Kutta 4th order solver.
// In order to obtain sensitivity, we differentiate ODE solver
// along with RHS of the equation.
// Using clad we can switch a solver or modify an equation without having
// to recalculate their derivatives manually.
//
// http://kitchingroup.cheme.cmu.edu/blog/2018/10/11/A-differentiable-ODE-integrator-for-sensitivity-analysis/
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -x c++ -lstdc++ -lm ODESolverSensitivity.cpp
//
// A typical invocation would be:
// ../../../../obj/Debug+Asserts/bin/clang  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../include/ -x c++ -lstdc++ -lm ODESolverSensitivity.cpp
//
// To plot the results install gnuplot and type:
// gnuplot -e "plot 'sens.dat' using 1:2 with lines; pause -1"

#include <fstream>
#include <cmath>

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

// dx/dy = f
double f(double x, double y, double a, double b, double c) {
  return -b * x + c * (a - y);
}

double rungeKutta(double x0, double y0, double x, double h, double a, double b, double c)
{
  int n = (int)((x - x0) / h);

  double k1, k2, k3, k4;

  double y = y0;
  for (unsigned i=1; i<=n; i++)
  {
    k1 = h*f(x0, y, a, b, c);
    k2 = h*f(x0 + 0.5*h, y + 0.5*k1, a, b, c);
    k3 = h*f(x0 + 0.5*h, y + 0.5*k2, a, b, c);
    k4 = h*f(x0 + h, y + k3, a, b, c);

    y = y + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4);;

    x0 = x0 + h;
  }

  return y;
}

double solution(double a, double b, double c, double x) {
  return rungeKutta(0, 0, x, 0.001, a, b, c);
}

// Evaluate b-sensitivity over x domain
double bSensitivity(double x) {
  // Using reverse mode since forward mode doesn't support
  // differentiation of multi-arg calls
  // FIXME: switch to clad::differentiate after fixing vgvassilev/clad#168
  auto h = clad::gradient(solution);

  double grad[4];
  h.execute(1.0, 3.0, 3.0, x, &grad[0], &grad[1], &grad[2], &grad[3]);

  return grad[1];
}

int main() {
  double x0 = 0;
  double x1 = 0.5;
  double h = 0.001;
  unsigned steps = (unsigned)((x1 - x0) / h);

  // Write tab-separated data to plot results e.g. with gnuplot
  std::ofstream out("sens.dat");
  for (unsigned i = 0; i < steps; i++) {
    double x = x0 + h * i;
    double db = bSensitivity(x);

    out << x << "\t" << std::abs(db) << std::endl;
  }
  out.close();

  return 0;
}
