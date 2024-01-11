template <typename VT>
struct ParallelFunctor {
  VT a;
  double x, y;

  ParallelFunctor(VT _a, double _x, double _y) : a(_a), x(_x), y(_y) {}

  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    for (size_t j =0; j<a.extent(1); ++j)
      a(i,j) += x * x + y;
  }
};


double f2(double x, double y) {
 return x; 
}

template <typename ViewtypeA>
typename ViewtypeA::value_type f_view(ViewtypeA a) {
  typename ViewtypeA::value_type sum;
  auto a_row_0 = Kokkos::subview( a, Kokkos::make_pair(0, 2), Kokkos::ALL );
  //auto h_a_row_0 = Kokkos::create_mirror_view(a_row_0); //create_mirror_view_and_copy
  //Kokkos::deep_copy(h_a_row_0, a_row_0);
  //sum = h_a_row_0(0,0);
  //kokkos_builtin_derivative::parallel_sum(sum,  Kokkos::subview(a_row_0,0,0));
  //sum = 10 * sum * sum * sum;
  kokkos_builtin_derivative::parallel_sum(sum, a_row_0);
  return 1e-6*sum*sum;
}


template <typename ViewtypeX>
typename ViewtypeX::value_type f_multilevel(ViewtypeX x) {
  typename ViewtypeX::value_type sum;
  //kokkos_builtin_derivative::parallel_sum(mean_x, x);

  ViewtypeX y("y", x.extent(0));

  Kokkos::parallel_for( x.extent(0), KOKKOS_LAMBDA ( const int j0) {
    x(j0) = 3*x(j0);
    //x(j0) = 3*x(j0) - mean_x;
  });

  Kokkos::parallel_for( x.extent(0)-1, KOKKOS_LAMBDA ( const int j1) {
    if (j1 != x.extent(0)-1)
      y(j1+1) = 2.6*x(j1)*x(j1);
    else
      y(j1) = 2.6*x(0)*x(0);
  });

  const int n_max = 10;
  const int n = x.extent(0) > n_max ? n_max : x.extent(0);

  auto y_n_rows = Kokkos::subview( y, Kokkos::make_pair(0, n));
  kokkos_builtin_derivative::parallel_sum(sum, y_n_rows);
  return sum;
}


template <typename ViewtypeA>
void f_view_2(ViewtypeA a, double tmp) {
  Kokkos::deep_copy(a, tmp);
}

template <typename T>
KOKKOS_INLINE_FUNCTION 
T pow(T a) {
  return a*a;
}

double f(double x, double y) {

  const int N1 = 4;
  constexpr int N2 = 4;

  Kokkos::View<double *[N2], Kokkos::LayoutLeft> a("a", N1);
  Kokkos::View<double *[N2], Kokkos::LayoutLeft> b("b", N1);

  double tmp = x * x + y;

  // These 2 lines do not work. Is it because nothing is returned by f_view_2?
  //f_view_2(a, tmp); 
  //return f_view(a);

  Kokkos::deep_copy(a, tmp);

  Kokkos::deep_copy(a, x);
  Kokkos::deep_copy(b, x * x + y);
  //Kokkos::deep_copy(a, b);

  Kokkos::parallel_for( b.extent(0), KOKKOS_LAMBDA ( const int j0) {
    b(j0,0) += 3.53;
  });

  Kokkos::parallel_for( a.extent(0)-1, KOKKOS_LAMBDA ( const int j1) {
    a(j1,0) += b(j1+1,0)*6.89 + b(j1,1) + pow(b(j1,1) + a(j1,1) + b(j1+1,0) * b(j1+1,0) * a(j1,2));
  });

  double sum;
  auto a_row_0 = Kokkos::subview( a, Kokkos::make_pair(0, 2), Kokkos::ALL );
  
  return f_view(a_row_0);
}