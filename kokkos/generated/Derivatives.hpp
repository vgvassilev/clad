template <typename type_a> 
void f_view_pullback(type_a a, typename type_a::value_type _d_y, type_a _d_a) {
    typename type_a::value_type _d_sum = 0;
    auto _d_a_row_0 = Kokkos::subview(_d_a, Kokkos::make_pair(0, 2), Kokkos::ALL);
    typename type_a::value_type _t0;
    typename type_a::value_type _t1;
    double _t2;
    typename type_a::value_type sum;
    auto a_row_0 = Kokkos::subview(a, Kokkos::make_pair(0, 2), Kokkos::ALL);
    kokkos_builtin_derivative::parallel_sum(sum, a_row_0);
    _t1 = sum;
    _t2 = 9.9999999999999995E-7 * _t1;
    _t0 = sum;
    goto _label0;
  _label0:
    {
        double _r0 = _d_y * _t0;
        double _r1 = _r0 * _t1;
        double _r2 = 9.9999999999999995E-7 * _r0;
        _d_sum += _r2;
        double _r3 = _t2 * _d_y;
        _d_sum += _r3;
    }
    kokkos_builtin_derivative::parallel_sum(_d_a_row_0, _d_sum);
}
void f_grad(double x, double y, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
    const int N1 = 4;
    int _d_N1 = 0;
    int _d_N2 = 0;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_a("_d_a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> _d_b("_d_b", N1);
    double _t0;
    double _t1;
    double _d_tmp = 0;
    int _d_i = 0;
    int _d_j = 0;
    double _t2;
    double _t3;
    double _t4;
    double _d_sum = 0;
    auto _d_a_row_0 = Kokkos::subview(_d_a, Kokkos::make_pair(0, 2), Kokkos::ALL);
    Kokkos::View<double *[4], Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<0> > _t5;
    const int N2 = 4;
    Kokkos::View<double *[4], Kokkos::LayoutLeft> a("a", N1);
    Kokkos::View<double *[4], Kokkos::LayoutLeft> b("b", N1);
    _t1 = x;
    _t0 = x;
    double tmp = _t1 * _t0 + y;
    const int i = 0;
    const int j = 0;
    Kokkos::deep_copy(a, tmp);
    Kokkos::deep_copy(a, x);
    _t2 = x;
    _t4 = x;
    _t3 = x;
    Kokkos::deep_copy(b, x * _t2 + y);
    Kokkos::deep_copy(a, b);
    double sum;
    auto a_row_0 = Kokkos::subview(a, Kokkos::make_pair(0, 2), Kokkos::ALL);
    _t5 = a_row_0;
    goto _label0;
  _label0:
    {
        f_view_pullback(_t5, 1, &_d_a_row_0);
        Kokkos::View<double *[4], Kokkos::LayoutLeft, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Kokkos::MemoryTraits<0> > _r5 = _d_a_row_0;
    }
    {
        Kokkos::deep_copy(_d_b, _d_a);
        Kokkos::deep_copy(_d_a, 0.);
    }
    {
        double _grad0 = 0.;
        kokkos_builtin_derivative::parallel_sum(_grad0, _d_b);
        Kokkos::deep_copy(_d_b, 0.);
        double _r2 = _grad0;
        double _r3 = _r2 * _t3;
        * _d_x += _r3;
        double _r4 = _t4 * _r2;
        * _d_x += _r4;
        * _d_y += _r2;
    }
    {
        kokkos_builtin_derivative::parallel_sum(* _d_x, _d_a);
        Kokkos::deep_copy(_d_a, 0.);
    }
    {
        kokkos_builtin_derivative::parallel_sum(_d_tmp, _d_a);
        Kokkos::deep_copy(_d_a, 0.);
    }
    {
        double _r0 = _d_tmp * _t0;
        * _d_x += _r0;
        double _r1 = _t1 * _d_tmp;
        * _d_x += _r1;
        * _d_y += _d_tmp;
    }
}
template <typename type_a> 
void f_view_grad(type_a a, type_a _d_a) {
    typename type_a::value_type _d_sum = 0;
    auto _d_a_row_0 = Kokkos::subview(_d_a, Kokkos::make_pair(0, 2), Kokkos::ALL);
    typename type_a::value_type _t0;
    typename type_a::value_type _t1;
    double _t2;
    typename type_a::value_type sum;
    auto a_row_0 = Kokkos::subview(a, Kokkos::make_pair(0, 2), Kokkos::ALL);
    kokkos_builtin_derivative::parallel_sum(sum, a_row_0);
    _t1 = sum;
    _t2 = 9.9999999999999995E-7 * _t1;
    _t0 = sum;
    goto _label0;
  _label0:
    {
        double _r0 = 1 * _t0;
        double _r1 = _r0 * _t1;
        double _r2 = 9.9999999999999995E-7 * _r0;
        _d_sum += _r2;
        double _r3 = _t2 * 1;
        _d_sum += _r3;
    }
    kokkos_builtin_derivative::parallel_sum(_d_a_row_0, _d_sum);
}
