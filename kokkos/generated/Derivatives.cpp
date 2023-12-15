inline double f_darg0(double x, double y) {
    double _d_x = 1;
    double _d_y = 0;
    const int _d_N1 = 0;
    const int N1 = 4;
    const int _d_N2 = 0;
    const int N2 = 4;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_a("_d_a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> a("a", N1);
    double _d_tmp = _d_x * x + x * _d_x + _d_y;
    double tmp = x * x + y;
    _d_a(0, 0) = _d_tmp;
    a(0, 0) = tmp;
    size_t _d_N1n;
    size_t N1n = a.extent(0);
    return _d_a(0, 0);
}
inline void f_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
    int _d_N1 = 0;
    int _d_N2 = 0;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_a("_d_a", N1);
    double _t0;
    double _t1;
    double _d_tmp = 0;
    size_t _d_N1n = 0;
    const int N1 = 4;
    const int N2 = 4;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> a("a", N1);
    _t1 = x;
    _t0 = x;
    double tmp = _t1 * _t0 + y;
    a(0, 0) = tmp;
    size_t N1n = a.extent(0);
    goto _label0;
  _label0:
    _d_a(0, 0) += 1;
    {
        double _r_d0 = _d_a(0, 0);
        _d_tmp += _r_d0;
        _d_a(0, 0) -= _r_d0;
        _d_a(0, 0);
    }
    {
        double _r0 = _d_tmp * _t0;
        * _d_x += _r0;
        double _r1 = _t1 * _d_tmp;
        * _d_x += _r1;
        * _d_y += _d_tmp;
    }
}
