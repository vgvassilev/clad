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


KOKKOS_INLINE_FUNCTION
double f(double x, double y) {

  const int N1 = 4;
  constexpr int N2 = 4;

  Kokkos::View<double *[N2], Kokkos::LayoutLeft> a("a", N1);

  double tmp = x * x + y;

  a(0,0) = tmp;

  //Kokkos::deep_copy(a, 0);

  size_t N1n = a.extent(0);

  //ParallelFunctor functor(a,x,y);

  //Kokkos::parallel_for(N1n, functor);

  return tmp;
}