// Error estimation over a summation loop. Where the estimate is
// accumulated is a property of the generated code, and running the demo
// writes gnuplot data and fifty iterations of output, so this reads the
// code clad wrote and stops there.
//
// RUN: %cladclang %S/../../demos/ErrorEstimation/FloatSum.cpp -I%S/../../include -o%t 2>&1 | %filecheck %s

// CHECK: void vanillaSum_grad(float x, unsigned int n, float *_d_x, unsigned int *_d_n, double &_final_error) {
// CHECK:    unsigned int _d_i = 0U;
// CHECK:    unsigned int i = 0U;
// CHECK:    clad::tape<float> _t1 = {};
// CHECK:    float _d_sum = 0.F;
// CHECK:    float sum = 0.;
// CHECK:    unsigned {{int|long|long long}} _t0;
// CHECK:    for (i = 0; i < n; i++) {
// CHECK:        clad::push(_t1, sum);
// CHECK:        sum = sum + x;
// CHECK:    }
// CHECK:    _d_sum += 1;
// CHECK:    for (_t0 = n > 0 ? (unsigned {{int|long|long long}})n : 0{{U|UL|ULL}}; _t0; _t0--) {
// CHECK:        i--;
// CHECK:        {
// CHECK:            _final_error += std::abs(_d_sum * sum * 1.1920928955078125E-7);
// CHECK:            sum = clad::pop(_t1);
// CHECK:            float _r_d0 = _d_sum;
// CHECK:            _d_sum = 0.F;
// CHECK:            _d_sum += _r_d0;
// CHECK:            *_d_x += _r_d0;
// CHECK:        }
// CHECK:    }
// CHECK:    _final_error += std::abs(_d_sum * sum * 1.1920928955078125E-7);
// CHECK:    _final_error += std::abs(*_d_x * x * 1.1920928955078125E-7);
// CHECK:}
