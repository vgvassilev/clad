
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

KOKKOS_INLINE_FUNCTION
double f(double x, double y) {

  const int N1 = 4;
  constexpr int N2 = 4;

  Kokkos::View<double *[N2], Kokkos::LayoutLeft> a("a", N1);
  Kokkos::View<double *[N2], Kokkos::LayoutLeft> b("b", N1);

  double tmp = x * x + y;

  const int i = 0;
  const int j = 0;

  double zero = 0.;
  //Kokkos::deep_copy(a, tmp);
  //auto a_row_0 = Kokkos::subview( a, 0, Kokkos::ALL );

  //b(i,j) = tmp;

  //Kokkos::deep_copy(a, tmp);
  //Kokkos::deep_copy(a, tmp);

  //Kokkos::deep_copy(a, x);
  Kokkos::deep_copy(b, x * x + y);
  Kokkos::deep_copy(a, b);

  //a(i,j) = x;
  //a(i,j) = x * x + y;

  //size_t N1n = a.extent(0);

  //ParallelFunctor functor(a,x,y);

  //Kokkos::parallel_for(N1n, functor);

 // double sum = f2(x, y);

  return a(i,j);
}