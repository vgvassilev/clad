#include "clad/Differentiator/Differentiator.h"
#include <Kokkos_Core.hpp>
#include "functor_for.hpp"
#include "lambda_reduction.hpp"
#include "lambda_reduction_subview.hpp"

//#define use_generated_file

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

#ifndef use_generated_file
    auto f_dx_exe = clad::differentiate(f, "x");
    auto f_grad_exe = clad::gradient(f);
    // Any of the two below will generate an "error: Attempted differentiation w.r.t. member 'x' which is not of real type."
    //auto weightedDotProduct_1_dx = clad::differentiate(weightedDotProduct_1<typeof(A),typeof(x),typeof(y)>, "x");
    //auto weightedDotProduct_2_dx = clad::differentiate(weightedDotProduct_2<typeof(A),typeof(x),typeof(y)>, "x");

    double dx_f = f_dx_exe.execute(3.,4.);
    std::cout << "dx: " << dx_f << std::endl;

    double dx = 0, dy = 0;
    // After this call, dx and dy will store the derivatives of x and y respectively.
    f_grad_exe.execute(3., 4., &dx, &dy);
    std::cout << "dx: " << dx << ' ' << "dy: " << dy << std::endl;
    assert(dx==dx_f && "error");
#else
    double dx = 0, dy = 0;
    double dx_f = f_darg0(3.,4.);
    std::cout << "dx: " << dx_f << std::endl;
    f_grad(3., 4., &dx, &dy);
    std::cout << "dx: " << dx << ' ' << "dy: " << dy << std::endl;
    assert(dx==dx_f && "error");
#endif
  }
  Kokkos::finalize();

}