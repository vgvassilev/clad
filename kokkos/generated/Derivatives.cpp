inline double f_darg0(double x, double y) {
    double _d_x = 1;
    double _d_y = 0;
    const int _d_N1 = 0;
    const int N1 = 4;
    const int _d_N2 = 0;
    const int N2 = 4;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_a("_d_a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> a("a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_b("_d_b", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> b("b", N1);
    double _d_tmp = _d_x * x + x * _d_x + _d_y;
    double tmp = x * x + y;
    const int _d_i = 0;
    const int i = 0;
    const int _d_j = 0;
    const int j = 0;
    double _d_zero = 0.;
    double zero = 0.;
    Kokkos::deep_copy(_d_b, _d_x * x + x * _d_x + _d_y, nullptr);
    Kokkos::deep_copy(b, x * x + y, nullptr);
    Kokkos::deep_copy(_d_a, _d_b, nullptr);
    Kokkos::deep_copy(a, b, nullptr);
    return _d_a(i, j);
}
inline void f_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
    int _d_N1 = 0;
    int _d_N2 = 0;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_a("_d_a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_b("_d_b", N1);
    double _t0;
    double _t1;
    double _d_tmp = 0;
    int _d_i = 0;
    int _d_j = 0;
    double _d_zero = 0;
    double _t2;
    double _t3;
    double _t4;
    const int N1 = 4;
    const int N2 = 4;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> a("a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> b("b", N1);
    _t1 = x;
    _t0 = x;
    double tmp = _t1 * _t0 + y;
    const int i = 0;
    const int j = 0;
    double zero = 0.;
    _t2 = x;
    _t4 = x;
    _t3 = x;
    Kokkos::deep_copy(b, x * _t2 + y, nullptr);
    Kokkos::deep_copy(a, b, nullptr);
    goto _label0;
  _label0:
    _d_a(i, j) += 1;
    Kokkos::deep_copy(_d_b, _d_a, nullptr);
    {
        double _grad0 = 0.;
        kokkos_builtin_derivative::parallel_sum(_grad0, _d_b);
        double _r2 = _grad0;
        double _r3 = _r2 * _t3;
        * _d_x += _r3;
        double _r4 = _t4 * _r2;
        * _d_x += _r4;
        * _d_y += _r2;
    }
    {
        double _r0 = _d_tmp * _t0;
        * _d_x += _r0;
        double _r1 = _t1 * _d_tmp;
        * _d_x += _r1;
        * _d_y += _d_tmp;
    }
}
