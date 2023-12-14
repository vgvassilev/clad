inline double f_darg0(double x, double y) {
    double _d_x = 1;
    double _d_y = 0;
    const int _d_N1 = 0;
    const int N1 = 4;
    const int _d_N2 = 0;
    const int N2 = 4;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_a("_d_a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> a("a", N1);
    Kokkos::deep_copy(_d_a, 0, nullptr);
    Kokkos::deep_copy(a, 0, nullptr);
    size_t _d_N1n;
    size_t N1n = a.extent(0);
    ParallelFunctor<Kokkos::View<double *[4], Kokkos::LayoutLeft> > _d_functor(_d_a, _d_x, _d_y);
    ParallelFunctor<Kokkos::View<double *[4], Kokkos::LayoutLeft> > functor(a, x, y);
    Kokkos::parallel_for(N1n, _d_functor);
    Kokkos::parallel_for(N1n, functor);
    return _d_a(0, 0);
}
