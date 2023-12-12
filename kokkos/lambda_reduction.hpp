
template <typename ViewtypeA, typename Viewtypex, typename Viewtypey>
typename ViewtypeA::value_type weightedDotProduct_2(ViewtypeA A, Viewtypex x, Viewtypey y) {
    // Application: <y,Ax> = y^T*A*x

    typename ViewtypeA::value_type result = 0;

    Kokkos::parallel_reduce( A.extent(0), KOKKOS_LAMBDA ( int j, typename ViewtypeA::value_type &update ) {
      typename ViewtypeA::value_type temp2 = 0;

      for ( int i = 0; i < A.extent(1); ++i ) {
        temp2 += A( j, i ) * x( i );
      }

      update += y( j ) * temp2;
    }, result );

    return result;
}