#include "clad/Differentiator/Differentiator.h"
#include <Kokkos_Core.hpp>
#include "functor_for.hpp"
#include "lambda_reduction.hpp"
#include "lambda_reduction_subview.hpp"

//#define use_generated_file
//#define use_forward_mode

#ifdef use_generated_file
#include "generated/Derivatives.cpp"
#endif

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double **> A("A", 10, 10);
    Kokkos::View<double *> x("x", 10);
    Kokkos::View<double *> y("y", 10);

    Kokkos::deep_copy(A, 3);
    Kokkos::deep_copy(x, 2);
    Kokkos::deep_copy(y, 4);

    std::cout << f(3.,4.) << std::endl;
    std::cout << weightedDotProduct_1(A, x, y) << std::endl;
    std::cout << weightedDotProduct_2(A, x, y) << std::endl;

    double epsilon = 1e-6;

    double f_pe = f(3.+epsilon,4.);
    double f_me = f(3.-epsilon,4.);
    double dx_f_FD = (f_pe-f_me) / (2 * epsilon);

    double tolerance = 1e-6;

    std::cout << "dx_f_FD: " << dx_f_FD << std::endl;

    double dx = 0, dy = 0;
    double dx_f;

#ifndef use_generated_file
  #ifdef use_forward_mode
    auto f_dx_exe = clad::differentiate(f, "x");
  #endif
    auto f_grad_exe = clad::gradient(f);
    // Any of the two below will generate an "error: Attempted differentiation w.r.t. member 'x' which is not of real type."
    //auto weightedDotProduct_1_dx = clad::differentiate(weightedDotProduct_1<typeof(A),typeof(x),typeof(y)>, "x");
    //auto weightedDotProduct_2_dx = clad::differentiate(weightedDotProduct_2<typeof(A),typeof(x),typeof(y)>, "x");
  #ifdef use_forward_mode
    dx_f = f_dx_exe.execute(3.,4.);
  #endif
    // After this call, dx and dy will store the derivatives of x and y respectively.
    f_grad_exe.execute(3., 4., &dx, &dy);
#else
  #ifdef use_forward_mode
    dx_f = f_darg0(3.,4.);
  #endif
    f_grad(3., 4., &dx, &dy);
#endif

  #ifdef use_forward_mode
    std::cout << "dx: " << dx_f << std::endl;
  #endif
    std::cout << "dx: " << dx << ' ' << "dy: " << dy << std::endl;
  #ifdef use_forward_mode
    assert(dx==dx_f && "error");
  #endif
    assert(std::abs(dx-dx_f_FD)<tolerance && "error");
  }
  Kokkos::finalize();

}