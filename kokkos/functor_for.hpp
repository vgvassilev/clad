
namespace kokkos_builtin_derivative {

template <typename ViewtypeA>
void parallel_sum(typename ViewtypeA::value_type &sum, const ViewtypeA A) {
  double tmp_sum = sum;
  sum = 0.;

  const int s0 = A.stride(0);
  const int s1 = A.stride(1);
  const int s2 = A.stride(2);
  const int s3 = A.stride(3);
  const int s4 = A.stride(4);
  const int s5 = A.stride(5);
  const int s6 = A.stride(6);
  const int s7 = A.stride(7);

  const int e0 = A.extent_int(0);
  const int e1 = A.extent_int(1);
  const int e2 = A.extent_int(2);
  const int e3 = A.extent_int(3);
  const int e4 = A.extent_int(4);
  const int e5 = A.extent_int(5);
  const int e6 = A.extent_int(6);
  const int e7 = A.extent_int(7);  

  Kokkos::Array<int, 6> begins = {0, 0, 0, 0, 0, 0};
  Kokkos::Array<int, 6> ends = {e0, e1, e2, e3, e4, e5};

  Kokkos::parallel_reduce(Kokkos::MDRangePolicy< Kokkos::Rank<6> > (begins, ends),
    KOKKOS_LAMBDA (const int i0, 
                   const int i1, 
                   const int i2, 
                   const int i3,
                   const int i4,
                   const int i5,
                   typename ViewtypeA::value_type& update) {
    const int offset = i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5;
    for ( int i6 = 0; i6 < e6; ++i6 ) {
      for ( int i7 = 0; i7 < e7; ++i7 ) {
        update += A.data()[offset + i6*s6 + i7*s7];
      }
    }
  }, sum );
  sum += tmp_sum;
}

template <typename ViewtypeA>
void parallel_sum(ViewtypeA A, const typename ViewtypeA::value_type b) {

  const int s0 = A.stride(0);
  const int s1 = A.stride(1);
  const int s2 = A.stride(2);
  const int s3 = A.stride(3);
  const int s4 = A.stride(4);
  const int s5 = A.stride(5);
  const int s6 = A.stride(6);
  const int s7 = A.stride(7);

  const int e0 = A.extent_int(0);
  const int e1 = A.extent_int(1);
  const int e2 = A.extent_int(2);
  const int e3 = A.extent_int(3);
  const int e4 = A.extent_int(4);
  const int e5 = A.extent_int(5);
  const int e6 = A.extent_int(6);
  const int e7 = A.extent_int(7);  

  Kokkos::Array<int, 6> begins = {0, 0, 0, 0, 0, 0};
  Kokkos::Array<int, 6> ends = {e0, e1, e2, e3, e4, e5};

  Kokkos::parallel_for(Kokkos::MDRangePolicy< Kokkos::Rank<6> > (begins, ends),
    KOKKOS_LAMBDA (const int i0, 
                   const int i1, 
                   const int i2, 
                   const int i3,
                   const int i4,
                   const int i5) {
    const int offset = i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5;
    for ( int i6 = 0; i6 < e6; ++i6 ) {
      for ( int i7 = 0; i7 < e7; ++i7 ) {
        A.data()[offset + i6*s6 + i7*s7] += b;
      }
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
double f_view(ViewtypeA a) {
  double sum;
  auto a_row_0 = Kokkos::subview( a, Kokkos::make_pair(0, 2), Kokkos::ALL );
  //auto h_a_row_0 = Kokkos::create_mirror_view(a_row_0); //create_mirror_view_and_copy
  //Kokkos::deep_copy(h_a_row_0, a_row_0);
  //sum = h_a_row_0(0,0);
  kokkos_builtin_derivative::parallel_sum(sum, a_row_0);
  return 1e-6*sum*sum;
}

template <typename ViewtypeA>
void f_view_2(ViewtypeA a, double tmp) {
  Kokkos::deep_copy(a, tmp);
}


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