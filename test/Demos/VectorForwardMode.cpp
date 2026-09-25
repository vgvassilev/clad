// Both array arguments differentiated in one pass rather than one pass
// each. The arrays' length is only known at run time, which is what the
// generated code has to cope with.
//
// RUN: %cladclang %S/../../demos/VectorForwardMode.cpp -I%S/../../include -o%t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// CHECK: void weighted_sum_dvec_0_1(double *arr, double *weights, int n, clad::array_ref<double> _d_arr, clad::array_ref<double> _d_weights) {
// CHECK-NEXT:    unsigned {{int|long|long long}} indepVarCount = _d_arr.size() + _d_weights.size();
// CHECK-NEXT:    clad::matrix<double> _d_vector_arr = clad::identity_matrix(_d_arr.size(), indepVarCount, {{0U|0UL|0ULL}});
// CHECK-NEXT:    clad::matrix<double> _d_vector_weights = clad::identity_matrix(_d_weights.size(), indepVarCount, _d_arr.size());
// CHECK-NEXT:    clad::array<int> _d_vector_n = clad::zero_vector(indepVarCount);
// CHECK-NEXT:    clad::array<double> _d_vector_res(clad::zero_vector(indepVarCount));
// CHECK-NEXT:    double res = 0;
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::array<int> _d_vector_i(clad::zero_vector(indepVarCount));
// CHECK-NEXT:        for (int i = 0; i < n; ++i) {
// CHECK-NEXT:            _d_vector_res += _d_vector_weights[i] * arr[i] + weights[i] * _d_vector_arr[i];
// CHECK-NEXT:            res += weights[i] * arr[i];
// CHECK-NEXT:        }
// CHECK-NEXT:    }
// CHECK-NEXT:    {
// CHECK-NEXT:        clad::array<double> _d_vector_return(_d_vector_res);
// CHECK-NEXT:        _d_arr = _d_vector_return.slice({{0U|0UL|0ULL}}, _d_arr.size());
// CHECK-NEXT:        _d_weights = _d_vector_return.slice(_d_arr.size(), _d_weights.size());
// CHECK-NEXT:        return;
// CHECK-NEXT:    }
// CHECK-NEXT: }

// CHECK-EXEC: Vector forward mode w.r.t. all:
// CHECK-EXEC:  darr = {0.5, 0.7, 0.9}
// CHECK-EXEC:  dweights = {3, 4, 5}
