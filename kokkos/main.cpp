#include "clad/Differentiator/Differentiator.h"
#include <Kokkos_Core.hpp>
#include "functor_for.hpp"
#include "lambda_reduction.hpp"
#include "lambda_reduction_subview.hpp"

//#define use_generated_file
#define use_forward_mode

#ifdef use_generated_file
#include "generated/Derivatives.hpp"
#endif

template <typename ViewtypeA, typename CladFunctionType>
typename ViewtypeA::value_type solve(ViewtypeA A, typename ViewtypeA::value_type (*objective)(ViewtypeA), CladFunctionType gradient) {
  ViewtypeA gradA("gradA", A.extent(0), A.extent(1));
  ViewtypeA tmp("tmp", A.extent(0), A.extent(1));

  std::vector<typename ViewtypeA::value_type> objective_history;

  int n_iterations = 10;
  int n_line_search = 10;

  double epsilon_min = 0.;
  double epsilon_tmp = 0.;
  double epsilon_max = 1000.;
  double epsilon_delta = (epsilon_max-epsilon_min)/n_line_search;

  typename ViewtypeA::value_type obj_min = objective(A);

  objective_history.push_back(obj_min);

  for (int i = 0; i < n_iterations; ++i) {

    gradient.execute(A, &gradA);

    epsilon_min = 0.;

    for (int j = 0; j < n_line_search; ++j) {
      epsilon_tmp = epsilon_delta * (j+1);
      Kokkos::parallel_for( A.extent(0), KOKKOS_LAMBDA ( int i) {
        
        for ( int j = 0; j < A.extent(1); ++j ) {
          tmp( i, j ) = A( i, j ) - epsilon_tmp * gradA( i, j );
        }
      });

      typename ViewtypeA::value_type obj_tmp = objective(tmp);

      if ( obj_tmp < obj_min) {
        obj_min = obj_tmp;
        epsilon_min = epsilon_tmp;
      }
    }

    Kokkos::parallel_for( A.extent(0), KOKKOS_LAMBDA ( int i) {
      
      for ( int j = 0; j < A.extent(1); ++j ) {
        A( i, j ) -= epsilon_min * gradA( i, j );
      }
    });

    objective_history.push_back(obj_min);
  }


  for (int i = 0; i < n_iterations + 1; ++i) {
    std::cout << "Objective value " << objective_history[i] << " iteration " << i << std::endl;
  }
  return obj_min;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    Kokkos::View<double **> A("A", 10, 10);
    Kokkos::View<double **> dA("dA", 10, 10);
    Kokkos::View<double *> x("x", 10);
    Kokkos::View<double *> y("y", 10);

    Kokkos::deep_copy(A, 3);
    Kokkos::deep_copy(x, 2);
    Kokkos::deep_copy(y, 4);

    std::cout << f(3.,4.) << std::endl;
    std::cout << weightedDotProduct_1(A, x, y) << std::endl;
    std::cout << weightedDotProduct_2(A, x, y) << std::endl;

    std::cout << f_view(A) << std::endl;

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
    // Any of the two below will generate an "error: Attempted differentiation w.r.t. member 'x' which is not of real type."
    //auto weightedDotProduct_1_dx = clad::differentiate(weightedDotProduct_1<typeof(A),typeof(x),typeof(y)>, "x");
    //auto weightedDotProduct_2_dx = clad::differentiate(weightedDotProduct_2<typeof(A),typeof(x),typeof(y)>, "x");
  #endif
    auto f_grad_exe = clad::gradient(f);
    auto f_view_grad_exe = clad::gradient(f_view<Kokkos::View<double **>>);
  #ifdef use_forward_mode
    dx_f = f_dx_exe.execute(3.,4.);
  #endif
    // After this call, dx and dy will store the derivatives of x and y respectively.
    f_grad_exe.execute(3., 4., &dx, &dy);
    f_view_grad_exe.execute(A, &dA);

    solve(A, &f_view, f_view_grad_exe);
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