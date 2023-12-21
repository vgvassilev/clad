
namespace kokkos_builtin_derivative {

template <typename ViewtypeA>
void parallel_sum(typename ViewtypeA::value_type &sum, const ViewtypeA A) {
  double tmp_sum = sum;
  sum = 0.;
  //to be updated to be rank independent
  Kokkos::parallel_reduce( A.extent(0), KOKKOS_LAMBDA ( int i, typename ViewtypeA::value_type &update ) {
    
    for ( int j = 0; j < A.extent(1); ++j ) {
      update += A( i, j );
    }
  }, sum );
  sum += tmp_sum;
}

template <typename ViewtypeA>
void parallel_sum(ViewtypeA A, const typename ViewtypeA::value_type b) {

  Kokkos::parallel_for( A.extent(0), KOKKOS_LAMBDA ( int i) {
    
    for ( int j = 0; j < A.extent(1); ++j ) {
      A( i, j ) += b;
    }
  });
}

}

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
KOKKOS_INLINE_FUNCTION
double f_view(ViewtypeA a) {
  double sum;
  auto a_row_0 = Kokkos::subview( a, Kokkos::make_pair(0, 2), Kokkos::ALL );
  
  sum = a_row_0(0,0);
  kokkos_builtin_derivative::parallel_sum(sum, a_row_0);
  return 1e-6*sum*sum;
}

template <typename ViewtypeA>
KOKKOS_INLINE_FUNCTION
void f_view_2(ViewtypeA a, double tmp) {
  Kokkos::deep_copy(a, tmp);
}


KOKKOS_INLINE_FUNCTION
double f(double x, double y) {

  const int N1 = 4;
  constexpr int N2 = 4;

  Kokkos::View<double *[N2], Kokkos::LayoutLeft> a("a", N1);
  Kokkos::View<double *[N2], Kokkos::LayoutLeft> b("b", N1);

  double tmp = x * x + y;

  const int i = 0;
  const int j = 0;

  // These 2 lines do not work. Is it because nothing is returned by f_view_2?
  //f_view_2(a, tmp); 
  //return f_view(a);

  Kokkos::deep_copy(a, tmp);

  Kokkos::deep_copy(a, x);
  Kokkos::deep_copy(b, x * x + y);
  Kokkos::deep_copy(a, b);

  double sum;
  auto a_row_0 = Kokkos::subview( a, Kokkos::make_pair(0, 2), Kokkos::ALL );
  
  sum = a_row_0(0,0);

  return sum*sum;
}