// RUN: %cladclang_cuda -Xclang -plugin-arg-clad -Xclang -disable-tbr -I%S/../../include -fsyntax-only \
// RUN:     --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath  -Xclang -verify \
// RUN:     %s 2>&1 | %filecheck %s
//
// RUN: %cladclang_cuda -Xclang -plugin-arg-clad -Xclang -disable-tbr -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oGradientKernels.out %s
//
// RUN: ./GradientKernels.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

__global__ void kernel(int *a) {
  *a *= *a;
}

// CHECK:    void kernel_grad(int *a, int *_d_a) {
//CHECK-NEXT:    int _t0 = *a;
//CHECK-NEXT:    *a *= *a;
//CHECK-NEXT:    {
//CHECK-NEXT:        *a = _t0;
//CHECK-NEXT:        int _r_d0 = *_d_a;
//CHECK-NEXT:        *_d_a = 0;
//CHECK-NEXT:        *_d_a += _r_d0 * *a;
//CHECK-NEXT:        atomicAdd(_d_a, *a * _r_d0);
//CHECK-NEXT:    }
//CHECK-NEXT: }

void fake_kernel(int *a) {
  *a *= *a;
}

__global__ void add_kernel(int *out, int *in) {
  int index = threadIdx.x;
  out[index] += in[index];
}

// CHECK:    void add_kernel_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:     int index0 = threadIdx.x;
//CHECK-NEXT:     int _t0 = out[index0];
//CHECK-NEXT:     out[index0] += in[index0];
//CHECK-NEXT:     {
//CHECK-NEXT:         out[index0] = _t0;
//CHECK-NEXT:         int _r_d0 = _d_out[index0];
//CHECK-NEXT:         atomicAdd(&_d_in[index0], _r_d0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void add_kernel_2(int *out, int *in) {
  out[threadIdx.x] += in[threadIdx.x];
}

// CHECK:    void add_kernel_2_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:     int _t0 = out[threadIdx.x];
//CHECK-NEXT:     out[threadIdx.x] += in[threadIdx.x];
//CHECK-NEXT:     {
//CHECK-NEXT:         out[threadIdx.x] = _t0;
//CHECK-NEXT:         int _r_d0 = _d_out[threadIdx.x];
//CHECK-NEXT:         atomicAdd(&_d_in[threadIdx.x], _r_d0);
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void add_kernel_3(int *out, int *in) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  out[index] += in[index];
}

// CHECK:    void add_kernel_3_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    int _t2 = out[index0];
//CHECK-NEXT:    out[index0] += in[index0];
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t2;
//CHECK-NEXT:        int _r_d0 = _d_out[index0];
//CHECK-NEXT:        atomicAdd(&_d_in[index0], _r_d0);
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void add_kernel_4(int *out, int *in, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < N) {
    int sum = 0;
    // Each thread sums elements in steps of warpSize
    for (int i = index; i < N; i += warpSize) {
        sum += in[i];
    }
    out[index] = sum;
  }
  return;
}

// CHECK: void add_kernel_4_grad_0_1(int *out, int *in, int N, int *_d_out, int *_d_in) {
//CHECK-NEXT:    int _d_N = 0;
//CHECK-NEXT:    bool _cond0;
//CHECK-NEXT:    int _d_sum = 0;
//CHECK-NEXT:    int sum = 0;
//CHECK-NEXT:    unsigned long _t2;
//CHECK-NEXT:    int _d_i = 0;
//CHECK-NEXT:    int i = 0;
//CHECK-NEXT:    clad::tape<int> _t3 = {};
//CHECK-NEXT:    clad::tape<int> _t4 = {};
//CHECK-NEXT:    int _t5;
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        _cond0 = index0 < N;
//CHECK-NEXT:        if (_cond0) {
//CHECK-NEXT:            sum = 0;
//CHECK-NEXT:            _t2 = 0UL;
//CHECK-NEXT:            for (i = index0; ; clad::push(_t3, i) , (i += warpSize)) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!(i < N))
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                _t2++;
//CHECK-NEXT:                clad::push(_t4, sum);
//CHECK-NEXT:                sum += in[i];
//CHECK-NEXT:            }
//CHECK-NEXT:            _t5 = out[index0];
//CHECK-NEXT:            out[index0] = sum;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        {
//CHECK-NEXT:            out[index0] = _t5;
//CHECK-NEXT:            int _r_d2 = _d_out[index0];
//CHECK-NEXT:            _d_out[index0] = 0;
//CHECK-NEXT:            _d_sum += _r_d2;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            for (;; _t2--) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!_t2)
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    i = clad::pop(_t3);
//CHECK-NEXT:                    int _r_d0 = _d_i;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    sum = clad::pop(_t4);
//CHECK-NEXT:                    int _r_d1 = _d_sum;
//CHECK-NEXT:                    atomicAdd(&_d_in[i], _r_d1);
//CHECK-NEXT:                }
//CHECK-NEXT:            }
//CHECK-NEXT:            _d_index += _d_i;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void add_kernel_5(int *out, int *in, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        int sum = 0;
        // Calculate the total number of threads in the grid
        int totalThreads = blockDim.x * gridDim.x;
        // Each thread sums elements in steps of the total number of threads in the grid
        for (int i = index; i < N; i += totalThreads) {
            sum += in[i];
        }
        out[index] = sum;
    }
}

// CHECK: void add_kernel_5_grad_0_1(int *out, int *in, int N, int *_d_out, int *_d_in) {
//CHECK-NEXT:    int _d_N = 0;
//CHECK-NEXT:    bool _cond0;
//CHECK-NEXT:    int _d_sum = 0;
//CHECK-NEXT:    int sum = 0;
//CHECK-NEXT:    unsigned int _t2;
//CHECK-NEXT:    unsigned int _t3;
//CHECK-NEXT:    int _d_totalThreads = 0;
//CHECK-NEXT:    int totalThreads = 0;
//CHECK-NEXT:    unsigned long _t4;
//CHECK-NEXT:    int _d_i = 0;
//CHECK-NEXT:    int i = 0;
//CHECK-NEXT:    clad::tape<int> _t5 = {};
//CHECK-NEXT:    clad::tape<int> _t6 = {};
//CHECK-NEXT:    int _t7;
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        _cond0 = index0 < N;
//CHECK-NEXT:        if (_cond0) {
//CHECK-NEXT:            sum = 0;
//CHECK-NEXT:            _t3 = blockDim.x;
//CHECK-NEXT:            _t2 = gridDim.x;
//CHECK-NEXT:            totalThreads = _t3 * _t2;
//CHECK-NEXT:            _t4 = 0UL;
//CHECK-NEXT:            for (i = index0; ; clad::push(_t5, i) , (i += totalThreads)) {
//CHECK-NEXT:                {
//CHECK-NEXT:                   if (!(i < N))
//CHECK-NEXT:                       break;
//CHECK-NEXT:                }
//CHECK-NEXT:                _t4++;
//CHECK-NEXT:                clad::push(_t6, sum);
//CHECK-NEXT:                sum += in[i];
//CHECK-NEXT:            }
//CHECK-NEXT:            _t7 = out[index0];
//CHECK-NEXT:            out[index0] = sum;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        {
//CHECK-NEXT:            out[index0] = _t7;
//CHECK-NEXT:            int _r_d2 = _d_out[index0];
//CHECK-NEXT:            _d_out[index0] = 0;
//CHECK-NEXT:            _d_sum += _r_d2;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            for (;; _t4--) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!_t4)
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    i = clad::pop(_t5);
//CHECK-NEXT:                    int _r_d0 = _d_i;
//CHECK-NEXT:                    _d_totalThreads += _r_d0;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    sum = clad::pop(_t6);
//CHECK-NEXT:                    int _r_d1 = _d_sum;
//CHECK-NEXT:                    atomicAdd(&_d_in[i], _r_d1);
//CHECK-NEXT:                }
//CHECK-NEXT:            }
//CHECK-NEXT:            _d_index += _d_i;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void add_kernel_6(int *a, int *b) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  a[index] = *b;
}

// CHECK: void add_kernel_6_grad(int *a, int *b, int *_d_a, int *_d_b) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    int _t2 = a[index0];
//CHECK-NEXT:    a[index0] = *b;
//CHECK-NEXT:    {
//CHECK-NEXT:        a[index0] = _t2;
//CHECK-NEXT:        int _r_d0 = _d_a[index0];
//CHECK-NEXT:        _d_a[index0] = 0;
//CHECK-NEXT:        atomicAdd(_d_b, _r_d0);
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void add_kernel_7(double *a, double *b) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  a[2 * index] = b[0];
  a[2 * index + 1] = b[0];
}

// CHECK: void add_kernel_7_grad(double *a, double *b, double *_d_a, double *_d_b) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    double _t2 = a[2 * index0];
//CHECK-NEXT:    a[2 * index0] = b[0];
//CHECK-NEXT:    double _t3 = a[2 * index0 + 1];
//CHECK-NEXT:    a[2 * index0 + 1] = b[0];
//CHECK-NEXT:    {
//CHECK-NEXT:        a[2 * index0 + 1] = _t3;
//CHECK-NEXT:        double _r_d1 = _d_a[2 * index0 + 1];
//CHECK-NEXT:        _d_a[2 * index0 + 1] = 0.;
//CHECK-NEXT:        atomicAdd(&_d_b[0], _r_d1);
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        a[2 * index0] = _t2;
//CHECK-NEXT:        double _r_d0 = _d_a[2 * index0];
//CHECK-NEXT:        _d_a[2 * index0] = 0.;
//CHECK-NEXT:        atomicAdd(&_d_b[0], _r_d0);
//CHECK-NEXT:    }
//CHECK-NEXT:}

__device__ double device_fn(const double in, double val) {
  return in + val;
}

__global__ void kernel_with_device_call(double *out, const double *in, double val) {
  int index = threadIdx.x;
  out[index] = device_fn(in[index], val);
}

// CHECK: __attribute__((device)) void device_fn_pullback_1(const double in, double val, double _d_y, double *_d_in, double *_d_val) {
//CHECK-NEXT:    {
//CHECK-NEXT:                *_d_in += _d_y;
//CHECK-NEXT:                *_d_val += _d_y;
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: void kernel_with_device_call_grad_0_2(double *out, const double *in, double val, double *_d_out, double *_d_val) {
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x;
//CHECK-NEXT:    double _t0 = out[index0];
//CHECK-NEXT:    out[index0] = device_fn(in[index0], val);
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t0;
//CHECK-NEXT:        double _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_out[index0] = 0.;
//CHECK-NEXT:        double _r0 = 0.;
//CHECK-NEXT:        double _r1 = 0.;
//CHECK-NEXT:        device_fn_pullback_1(in[index0], val, _r_d0, &_r0, &_r1);
//CHECK-NEXT:        atomicAdd(_d_val, _r1);
//CHECK-NEXT:    }
//CHECK-NEXT:}

__device__ double device_fn_2(const double *in, double val) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  return in[index] + val;
}

__global__ void kernel_with_device_call_2(double *out, const double *in, double val) {
  int index = threadIdx.x;
  out[index] = device_fn_2(in, val);
} 

__global__ void dup_kernel_with_device_call_2(double *out, const double *in, double val) {
  int index = threadIdx.x;
  out[index] = device_fn_2(in, val);
} 

//CHECK: __attribute__((device)) void device_fn_2_pullback_1(const double *in, double val, double _d_y, double *_d_in, double *_d_val) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        _d_in[index0] += _d_y;
//CHECK-NEXT:        *_d_val += _d_y;
//CHECK-NEXT:    }
//CHECK-NEXT:}

//CHECK: void device_fn_2_pullback_1(const double *in, double val, double _d_y, double *_d_val) __attribute__((device));

// CHECK: void kernel_with_device_call_2_grad_0_2(double *out, const double *in, double val, double *_d_out, double *_d_val) {
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x;
//CHECK-NEXT:    double _t0 = out[index0];
//CHECK-NEXT:    out[index0] = device_fn_2(in, val);
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t0;
//CHECK-NEXT:        double _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_out[index0] = 0.;
//CHECK-NEXT:        double _r0 = 0.;
//CHECK-NEXT:        device_fn_2_pullback_1(in, val, _r_d0, &_r0);
//CHECK-NEXT:        atomicAdd(_d_val, _r0);
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: __attribute__((device)) void device_fn_2_pullback_0(const double *in, double val, double _d_y, double *_d_in, double *_d_val) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        atomicAdd(&_d_in[index0], _d_y);
//CHECK-NEXT:        *_d_val += _d_y;
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: void kernel_with_device_call_2_grad_0_1(double *out, const double *in, double val, double *_d_out, double *_d_in) {
//CHECK-NEXT:    double _d_val = 0.;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x;
//CHECK-NEXT:    double _t0 = out[index0];
//CHECK-NEXT:    out[index0] = device_fn_2(in, val);
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t0;
//CHECK-NEXT:        double _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_out[index0] = 0.;
//CHECK-NEXT:        double _r0 = 0.;
//CHECK-NEXT:        device_fn_2_pullback_0(in, val, _r_d0, _d_in, &_r0);
//CHECK-NEXT:        _d_val += _r0;
//CHECK-NEXT:    }
//CHECK-NEXT:}

__device__ double device_fn_3(double *in, double *val) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  return in[index] + *val;
}

__global__ void kernel_with_device_call_3(double *out, double *in, double *val) {
  int index = threadIdx.x;
  out[index] = device_fn_3(in, val);
} 

// CHECK: __attribute__((device)) void device_fn_3_pullback_0_1(double *in, double *val, double _d_y, double *_d_in, double *_d_val) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        atomicAdd(&_d_in[index0], _d_y);
//CHECK-NEXT:        atomicAdd(_d_val, _d_y);
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: void kernel_with_device_call_3_grad(double *out, double *in, double *val, double *_d_out, double *_d_in, double *_d_val) {
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x;
//CHECK-NEXT:    double _t0 = out[index0];
//CHECK-NEXT:    out[index0] = device_fn_3(in, val);
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t0;
//CHECK-NEXT:        double _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_out[index0] = 0.;
//CHECK-NEXT:        device_fn_3_pullback_0_1(in, val, _r_d0, _d_in, _d_val);
//CHECK-NEXT:    }
//CHECK-NEXT:}

__device__ double device_fn_4(double *in, double val) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  return in[index] + val;
}

__device__ double device_with_device_call(double *in, double val) {
  return device_fn_4(in, val);
}

__global__ void kernel_with_nested_device_call(double *out, double *in, double val) {
  int index = threadIdx.x;
  out[index] = device_with_device_call(in, val);
}

// CHECK: __attribute__((device)) void device_fn_4_pullback_0_1(double *in, double val, double _d_y, double *_d_in, double *_d_val) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        atomicAdd(&_d_in[index0], _d_y);
//CHECK-NEXT:        *_d_val += _d_y;
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: __attribute__((device)) void device_with_device_call_pullback_0(double *in, double val, double _d_y, double *_d_in, double *_d_val) {
//CHECK-NEXT:    {
//CHECK-NEXT:        double _r0 = 0.;
//CHECK-NEXT:        device_fn_4_pullback_0_1(in, val, _d_y, _d_in, &_r0);
//CHECK-NEXT:        *_d_val += _r0;
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: void kernel_with_nested_device_call_grad_0_1(double *out, double *in, double val, double *_d_out, double *_d_in) {
//CHECK-NEXT:    double _d_val = 0.;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x;
//CHECK-NEXT:    double _t0 = out[index0];
//CHECK-NEXT:    out[index0] = device_with_device_call(in, val);
//CHECK-NEXT:    {
//CHECK-NEXT:       out[index0] = _t0;
//CHECK-NEXT:       double _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_out[index0] = 0.;
//CHECK-NEXT:        double _r0 = 0.;
//CHECK-NEXT:        device_with_device_call_pullback_0(in, val, _r_d0, _d_in, &_r0);
//CHECK-NEXT:        _d_val += _r0;
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void fn1(double *out, const double *in, double val) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  double temp = val;
  out[index] = device_fn(in[index], temp);
}

// CHECK: void fn1_grad_0_2(double *out, const double *in, double val, double *_d_out, double *_d_val) {
// CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
// CHECK-NEXT:    unsigned int _t0 = blockDim.x;
// CHECK-NEXT:    int _d_index = 0;
// CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
// CHECK-NEXT:    double _d_temp = 0.;
// CHECK-NEXT:    double temp = val;
// CHECK-NEXT:    double _t2 = out[index0];
// CHECK-NEXT:    out[index0] = device_fn(in[index0], temp);
// CHECK-NEXT:    {
// CHECK-NEXT:        out[index0] = _t2;
// CHECK-NEXT:        double _r_d0 = _d_out[index0];
// CHECK-NEXT:        _d_out[index0] = 0.;
// CHECK-NEXT:        double _r0 = 0.;
// CHECK-NEXT:        double _r1 = 0.;
// CHECK-NEXT:        device_fn_pullback_1(in[index0], temp, _r_d0, &_r0, &_r1);
// CHECK-NEXT:        _d_temp += _r1;
// CHECK-NEXT:    }
// CHECK-NEXT:    atomicAdd(_d_val, _d_temp);
// CHECK-NEXT: }

__global__ void kernel_call(double *a, double *b) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  a[index] = *b;
}

void fn(double *out, double *in) {
  kernel_call<<<1, 10>>>(out, in);
}

// CHECK: void fn_grad(double *out, double *in, double *_d_out, double *_d_in) {
//CHECK-NEXT:     kernel_call<<<1, 10>>>(out, in);
//CHECK-NEXT:     kernel_call_pullback<<<1, 10>>>(out, in, _d_out, _d_in);
//CHECK-NEXT: }

double fn_memory(double *out, double *in) {
  double *in_dev = nullptr;
  cudaMalloc(&in_dev, 10 * sizeof(double));
  cudaMemcpy(in_dev, in, 10 * sizeof(double), cudaMemcpyHostToDevice);
  kernel_call<<<1, 10>>>(out, in_dev);
  cudaDeviceSynchronize();
  double *out_host = (double *)malloc(10 * sizeof(double));
  cudaMemcpy(out_host, out, 10 * sizeof(double), cudaMemcpyDeviceToHost);
  double res = 0;
  for (int i=0; i < 10; ++i) {
    printf("Writing result of out[%d]\n", i);
    res += out_host[i];
  }
  free(out_host);
  cudaFree(out);
  cudaFree(in_dev);
  return res;
}

// CHECK: void fn_memory_grad(double *out, double *in, double *_d_out, double *_d_in) {
//CHECK-NEXT:    int _d_i = 0;
//CHECK-NEXT:    int i = 0;
//CHECK-NEXT:    clad::tape<double> _t1 = {};
//CHECK-NEXT:    double *_d_in_dev = nullptr;
//CHECK-NEXT:    double *in_dev = nullptr;
//CHECK-NEXT:    cudaMalloc(&_d_in_dev, 10 * sizeof(double));
//CHECK-NEXT:    cudaMemset(_d_in_dev, 0, 10 * sizeof(double));
//CHECK-NEXT:    cudaMalloc(&in_dev, 10 * sizeof(double));
//CHECK-NEXT:    cudaMemcpy(in_dev, in, 10 * sizeof(double), cudaMemcpyHostToDevice);
//CHECK-NEXT:    kernel_call<<<1, 10>>>(out, in_dev);
//CHECK-NEXT:    cudaDeviceSynchronize();
//CHECK-NEXT:    double *_d_out_host = (double *)malloc(10 * sizeof(double));
//CHECK-NEXT:    memset(_d_out_host, 0, 10 * sizeof(double));
//CHECK-NEXT:    double *out_host = (double *)malloc(10 * sizeof(double));
//CHECK-NEXT:    cudaMemcpy(out_host, out, 10 * sizeof(double), cudaMemcpyDeviceToHost);
//CHECK-NEXT:    double _d_res = 0.;
//CHECK-NEXT:    double res = 0;
//CHECK-NEXT:    unsigned long _t0 = 0UL;
//CHECK-NEXT:    for (i = 0; ; ++i) {
//CHECK-NEXT:        {
//CHECK-NEXT:            if (!(i < 10))
//CHECK-NEXT:                break;
//CHECK-NEXT:        }
//CHECK-NEXT:        _t0++;
//CHECK-NEXT:        printf("Writing result of out[%d]\n", i);
//CHECK-NEXT:        clad::push(_t1, res);
//CHECK-NEXT:        res += out_host[i];
//CHECK-NEXT:    }
//CHECK-NEXT:    _d_res += 1;
//CHECK-NEXT:    for (;; _t0--) {
//CHECK-NEXT:        {
//CHECK-NEXT:            if (!_t0)
//CHECK-NEXT:                break;
//CHECK-NEXT:        }
//CHECK-NEXT:        --i;
//CHECK-NEXT:        {
//CHECK-NEXT:            res = clad::pop(_t1);
//CHECK-NEXT:            double _r_d0 = _d_res;
//CHECK-NEXT:            _d_out_host[i] += _r_d0;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        unsigned long _r2 = 0UL;
//CHECK-NEXT:        cudaMemcpyKind _r3 = static_cast<cudaMemcpyKind>(0U);
//CHECK-NEXT:        clad::custom_derivatives::cudaMemcpy_pullback(out_host, out, 10 * sizeof(double), cudaMemcpyDeviceToHost, static_cast<cudaError>(0U), _d_out_host, _d_out, &_r2, &_r3);
//CHECK-NEXT:    }
//CHECK-NEXT:    kernel_call_pullback<<<1, 10>>>(out, in_dev, _d_out, _d_in_dev);
//CHECK-NEXT:    {
//CHECK-NEXT:        unsigned long _r0 = 0UL;
//CHECK-NEXT:        cudaMemcpyKind _r1 = static_cast<cudaMemcpyKind>(0U);
//CHECK-NEXT:        clad::custom_derivatives::cudaMemcpy_pullback(in_dev, in, 10 * sizeof(double), cudaMemcpyHostToDevice, static_cast<cudaError>(0U), _d_in_dev, _d_in, &_r0, &_r1);
//CHECK-NEXT:    }
//CHECK-NEXT:    free(out_host);
//CHECK-NEXT:    free(_d_out_host);
//CHECK-NEXT:    cudaFree(out);
//CHECK-NEXT:    cudaFree(in_dev);
//CHECK-NEXT:    cudaFree(_d_in_dev);
//CHECK-NEXT:}

void launch_add_kernel_4(int *out, int *in, const int N) {
  int *in_dev = nullptr;
  cudaMalloc(&in_dev, N * sizeof(int));
  cudaMemcpy(in_dev, in, N * sizeof(int), cudaMemcpyHostToDevice);
  int *out_dev = nullptr;
  cudaMalloc(&out_dev, N * sizeof(int));
  cudaMemcpy(out_dev, out, N * sizeof(int), cudaMemcpyHostToDevice);

  add_kernel_4<<<1, 5>>>(out_dev, in_dev, N);

  cudaMemcpy(out, out_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(in_dev);
  cudaFree(out_dev);
}

// CHECK: __attribute__((global)) void add_kernel_4_pullback(int *out, int *in, int N, int *_d_out, int *_d_in, int *_d_N) {
//CHECK-NEXT:    bool _cond0;
//CHECK-NEXT:    int _d_sum = 0;
//CHECK-NEXT:    int sum = 0;
//CHECK-NEXT:    unsigned long _t2;
//CHECK-NEXT:    int _d_i = 0;
//CHECK-NEXT:    int i = 0;
//CHECK-NEXT:    clad::tape<int> _t3 = {};
//CHECK-NEXT:    clad::tape<int> _t4 = {};
//CHECK-NEXT:    int _t5;
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        _cond0 = index0 < N;
//CHECK-NEXT:        if (_cond0) {
//CHECK-NEXT:            sum = 0;
//CHECK-NEXT:            _t2 = 0UL;
//CHECK-NEXT:            for (i = index0; ; clad::push(_t3, i) , (i += warpSize)) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!(i < N))
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                _t2++;
//CHECK-NEXT:                clad::push(_t4, sum);
//CHECK-NEXT:                sum += in[i];
//CHECK-NEXT:            }
//CHECK-NEXT:            _t5 = out[index0];
//CHECK-NEXT:            out[index0] = sum;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        {
//CHECK-NEXT:            out[index0] = _t5;
//CHECK-NEXT:            int _r_d2 = _d_out[index0];
//CHECK-NEXT:            _d_out[index0] = 0;
//CHECK-NEXT:            _d_sum += _r_d2;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            for (;; _t2--) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!_t2)
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    i = clad::pop(_t3);
//CHECK-NEXT:                    int _r_d0 = _d_i;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    sum = clad::pop(_t4);
//CHECK-NEXT:                    int _r_d1 = _d_sum;
//CHECK-NEXT:                    atomicAdd(&_d_in[i], _r_d1);
//CHECK-NEXT:                }
//CHECK-NEXT:            }
//CHECK-NEXT:            _d_index += _d_i;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}

// CHECK: void launch_add_kernel_4_grad_0_1(int *out, int *in, const int N, int *_d_out, int *_d_in) {
//CHECK-NEXT:    int _d_N = 0;
//CHECK-NEXT:    int *_d_in_dev = nullptr;
//CHECK-NEXT:    int *in_dev = nullptr;
//CHECK-NEXT:    cudaMalloc(&_d_in_dev, N * sizeof(int));
//CHECK-NEXT:    cudaMemset(_d_in_dev, 0, N * sizeof(int));
//CHECK-NEXT:    cudaMalloc(&in_dev, N * sizeof(int));
//CHECK-NEXT:    cudaMemcpy(in_dev, in, N * sizeof(int), cudaMemcpyHostToDevice);
//CHECK-NEXT:    int *_d_out_dev = nullptr;
//CHECK-NEXT:    int *out_dev = nullptr;
//CHECK-NEXT:    cudaMalloc(&_d_out_dev, N * sizeof(int));
//CHECK-NEXT:    cudaMemset(_d_out_dev, 0, N * sizeof(int));
//CHECK-NEXT:    cudaMalloc(&out_dev, N * sizeof(int));
//CHECK-NEXT:    cudaMemcpy(out_dev, out, N * sizeof(int), cudaMemcpyHostToDevice);
//CHECK-NEXT:    add_kernel_4<<<1, 5>>>(out_dev, in_dev, N);
//CHECK-NEXT:    cudaMemcpy(out, out_dev, N * sizeof(int), cudaMemcpyDeviceToHost);
//CHECK-NEXT:    {
//CHECK-NEXT:        unsigned long _r6 = 0UL;
//CHECK-NEXT:        cudaMemcpyKind _r7 = static_cast<cudaMemcpyKind>(0U);
//CHECK-NEXT:        clad::custom_derivatives::cudaMemcpy_pullback(out, out_dev, N * sizeof(int), cudaMemcpyDeviceToHost, static_cast<cudaError>(0U), _d_out, _d_out_dev, &_r6, &_r7);
//CHECK-NEXT:        _d_N += _r6 * sizeof(int);
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        int _r4 = 0;
//CHECK-NEXT:        int *_r5 = nullptr;
//CHECK-NEXT:        cudaMalloc(&_r5, 4);
//CHECK-NEXT:        cudaMemset(_r5, 0, 4);
//CHECK-NEXT:        add_kernel_4_pullback<<<1, 5>>>(out_dev, in_dev, N, _d_out_dev, _d_in_dev, _r5);
//CHECK-NEXT:        cudaMemcpy(&_r4, _r5, 4, cudaMemcpyDeviceToHost);
//CHECK-NEXT:        cudaFree(_r5);
//CHECK-NEXT:        _d_N += _r4;
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        unsigned long _r2 = 0UL;
//CHECK-NEXT:        cudaMemcpyKind _r3 = static_cast<cudaMemcpyKind>(0U);
//CHECK-NEXT:        clad::custom_derivatives::cudaMemcpy_pullback(out_dev, out, N * sizeof(int), cudaMemcpyHostToDevice, static_cast<cudaError>(0U), _d_out_dev, _d_out, &_r2, &_r3);
//CHECK-NEXT:        _d_N += _r2 * sizeof(int);
//CHECK-NEXT:    }
//CHECK-NEXT:    {
//CHECK-NEXT:        unsigned long _r0 = 0UL;
//CHECK-NEXT:        cudaMemcpyKind _r1 = static_cast<cudaMemcpyKind>(0U);
//CHECK-NEXT:        clad::custom_derivatives::cudaMemcpy_pullback(in_dev, in, N * sizeof(int), cudaMemcpyHostToDevice, static_cast<cudaError>(0U), _d_in_dev, _d_in, &_r0, &_r1);
//CHECK-NEXT:        _d_N += _r0 * sizeof(int);
//CHECK-NEXT:    }
//CHECK-NEXT:    cudaFree(in_dev);
//CHECK-NEXT:    cudaFree(_d_in_dev);
//CHECK-NEXT:    cudaFree(out_dev);
//CHECK-NEXT:    cudaFree(_d_out_dev);
//CHECK-NEXT:}

// CHECK: __attribute__((device)) void device_fn_2_pullback_1(const double *in, double val, double _d_y, double *_d_val) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    *_d_val += _d_y;
//CHECK-NEXT:}

#define TEST(F, grid, block, shared_mem, use_stream, x, dx, N)              \
  {                                                                         \
    int *fives = (int*)malloc(N * sizeof(int));                             \
    for(int i = 0; i < N; i++) {                                            \
      fives[i] = 5;                                                         \
    }                                                                       \
    int *ones = (int*)malloc(N * sizeof(int));                              \
    for(int i = 0; i < N; i++) {                                            \
      ones[i] = 1;                                                          \
    }                                                                       \
    cudaMemcpy(x, fives, N * sizeof(int), cudaMemcpyHostToDevice);          \
    cudaMemcpy(dx, ones, N * sizeof(int), cudaMemcpyHostToDevice);          \
    auto test = clad::gradient(F);                                          \
    if constexpr (use_stream) {                                             \
      cudaStream_t cudaStream;                                              \
      cudaStreamCreate(&cudaStream);                                        \
      test.execute_kernel(grid, block, shared_mem, cudaStream, x, dx);      \
    }                                                                       \
    else {                                                                  \
      test.execute_kernel(grid, block, x, dx);                              \
    }                                                                       \
    int *res = (int*)malloc(N * sizeof(int));                               \
    cudaMemcpy(res, dx, N * sizeof(int), cudaMemcpyDeviceToHost);           \
    for (int i = 0; i < (N - 1); i++) {                                     \
      printf("%d, ", res[i]);                                               \
    }                                                                       \
    printf("%d\n", res[N-1]);                                               \
    free(fives);                                                            \
    free(ones);                                                             \
    free(res);                                                              \
  }


#define TEST_2(F, grid, block, shared_mem, use_stream, args, y, x, dy, dx, N) \
  {                                                                           \
    int *fives = (int*)malloc(N * sizeof(int));                               \
    for(int i = 0; i < N; i++) {                                              \
      fives[i] = 5;                                                           \
    }                                                                         \
    int *zeros = (int*)malloc(N * sizeof(int));                               \
    for(int i = 0; i < N; i++) {                                              \
      zeros[i] = 0;                                                           \
    }                                                                         \
    cudaMemcpy(x, fives, N * sizeof(int), cudaMemcpyHostToDevice);            \
    cudaMemcpy(y, zeros, N * sizeof(int), cudaMemcpyHostToDevice);            \
    cudaMemcpy(dy, fives, N * sizeof(int), cudaMemcpyHostToDevice);           \
    cudaMemcpy(dx, zeros, N * sizeof(int), cudaMemcpyHostToDevice);           \
    auto test = clad::gradient(F, args);                                      \
    if constexpr (use_stream) {                                               \
      cudaStream_t cudaStream;                                                \
      cudaStreamCreate(&cudaStream);                                          \
      test.execute_kernel(grid, block, shared_mem, cudaStream, y, x, dy, dx); \
    }                                                                         \
    else {                                                                    \
      test.execute_kernel(grid, block, y, x, dy, dx);                         \
    }                                                                         \
    int *res = (int*)malloc(N * sizeof(int));                                 \
    cudaMemcpy(res, dx, N * sizeof(int), cudaMemcpyDeviceToHost);             \
    for (int i = 0; i < (N - 1); i++) {                                       \
      printf("%d, ", res[i]);                                                 \
    }                                                                         \
    printf("%d\n", res[N-1]);                                                 \
    free(fives);                                                              \
    free(zeros);                                                              \
    free(res);                                                                \
  }

#define TEST_2_N(F, grid, block, shared_mem, use_stream, args, y, x, dy, dx, N)   \
  {                                                                               \
    int *fives = (int*)malloc(N * sizeof(int));                                   \
    for(int i = 0; i < N; i++) {                                                  \
      fives[i] = 5;                                                               \
    }                                                                             \
    int *zeros = (int*)malloc(N * sizeof(int));                                   \
    for(int i = 0; i < N; i++) {                                                  \
      zeros[i] = 0;                                                               \
    }                                                                             \
    cudaMemcpy(x, fives, N * sizeof(int), cudaMemcpyHostToDevice);                \
    cudaMemcpy(y, zeros, N * sizeof(int), cudaMemcpyHostToDevice);                \
    cudaMemcpy(dy, fives, N * sizeof(int), cudaMemcpyHostToDevice);               \
    cudaMemcpy(dx, zeros, N * sizeof(int), cudaMemcpyHostToDevice);               \
    auto test = clad::gradient(F, args);                                          \
    if constexpr (use_stream) {                                                   \
      cudaStream_t cudaStream;                                                    \
      cudaStreamCreate(&cudaStream);                                              \
      test.execute_kernel(grid, block, shared_mem, cudaStream, y, x, N, dy, dx);  \
    }                                                                             \
    else {                                                                        \
      test.execute_kernel(grid, block, y, x, N, dy, dx);                          \
    }                                                                             \
    int *res = (int*)malloc(N * sizeof(int));                                     \
    cudaMemcpy(res, dx, N * sizeof(int), cudaMemcpyDeviceToHost);                 \
    for (int i = 0; i < (N - 1); i++) {                                           \
      printf("%d, ", res[i]);                                                     \
    }                                                                             \
    printf("%d\n", res[N-1]);                                                     \
    free(fives);                                                                  \
    free(zeros);                                                                  \
    free(res);                                                                    \
  }

#define TEST_2_D(F, grid, block, shared_mem, use_stream, args, y, x, dy, dx, N) \
  {                                                                             \
    double *fives = (double*)malloc(N * sizeof(double));                        \
    for(int i = 0; i < N; i++) {                                                \
      fives[i] = 5;                                                             \
    }                                                                           \
    double *zeros = (double*)malloc(N * sizeof(double));                        \
    for(int i = 0; i < N; i++) {                                                \
      zeros[i] = 0;                                                             \
    }                                                                           \
    cudaMemcpy(x, fives, N * sizeof(double), cudaMemcpyHostToDevice);           \
    cudaMemcpy(y, zeros, N * sizeof(double), cudaMemcpyHostToDevice);           \
    cudaMemcpy(dy, fives, N * sizeof(double), cudaMemcpyHostToDevice);          \
    cudaMemcpy(dx, zeros, N * sizeof(double), cudaMemcpyHostToDevice);          \
    auto test = clad::gradient(F, args);                                        \
    if constexpr (use_stream) {                                                 \
      cudaStream_t cudaStream;                                                  \
      cudaStreamCreate(&cudaStream);                                            \
      test.execute_kernel(grid, block, shared_mem, cudaStream, y, x, dy, dx);   \
    }                                                                           \
    else {                                                                      \
      test.execute_kernel(grid, block, y, x, dy, dx);                           \
    }                                                                           \
    double *res = (double*)malloc(N * sizeof(double));                          \
    cudaMemcpy(res, dx, N * sizeof(double), cudaMemcpyDeviceToHost);            \
    for (int i = 0; i < (N - 1); i++) {                                         \
      printf("%0.2f, ", res[i]);                                                \
    }                                                                           \
    printf("%0.2f\n", res[N-1]);                                                \
    free(fives);                                                                \
    free(zeros);                                                                \
    free(res);                                                                  \
  }

#define INIT(x, y, val, dx, dy, d_val)                                          \
{                                                                               \
  cudaMemcpy(x, fives, 10 * sizeof(double), cudaMemcpyHostToDevice);            \
  cudaMemcpy(y, zeros, 10 * sizeof(double), cudaMemcpyHostToDevice);            \
  cudaMemcpy(val, fives, sizeof(double), cudaMemcpyHostToDevice);               \
  cudaMemcpy(dx, zeros, 10 * sizeof(double), cudaMemcpyHostToDevice);           \
  cudaMemcpy(dy, fives, 10 * sizeof(double), cudaMemcpyHostToDevice);           \
  cudaMemcpy(d_val, zeros, sizeof(double), cudaMemcpyHostToDevice);             \
}

int main(void) {
  int *a, *d_a;
  cudaMalloc(&a, sizeof(int));
  cudaMalloc(&d_a, sizeof(int));

  TEST(kernel, dim3(1), dim3(1), 0, false, a, d_a, 1); // CHECK-EXEC: 10
  TEST(kernel, dim3(1), dim3(1), 0, true, a, d_a, 1); // CHECK-EXEC: 10

  auto error = clad::gradient(fake_kernel); 
  error.execute_kernel(dim3(1), dim3(1), a, d_a); // CHECK-EXEC: Use execute() for non-global CUDA kernels

  auto test = clad::gradient(kernel);
  test.execute(a, d_a); // CHECK-EXEC: Use execute_kernel() for global CUDA kernels

  cudaFree(a);
  cudaFree(d_a);

  int *dummy_in, *dummy_out, *d_out, *d_in;
  cudaMalloc(&dummy_in, 10 * sizeof(int));
  cudaMalloc(&dummy_out, 10 * sizeof(int));
  cudaMalloc(&d_out, 10 * sizeof(int));
  cudaMalloc(&d_in, 10 * sizeof(int));

  TEST_2(add_kernel, dim3(1), dim3(5, 1, 1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_2, dim3(1), dim3(5, 1, 1), 0, true, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_3, dim3(5, 1, 1), dim3(1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2_N(add_kernel_4, dim3(1), dim3(5, 1, 1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2_N(add_kernel_5, dim3(2, 1, 1), dim3(1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_6, dim3(1), dim3(5, 1, 1), 0, false, "a, b", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 25, 0, 0, 0, 0

  cudaFree(dummy_in);
  cudaFree(dummy_out);
  cudaFree(d_out);
  cudaFree(d_in);

  double *dummy_in_double, *dummy_out_double, *d_out_double, *d_in_double;
  cudaMalloc(&dummy_in_double, 10 * sizeof(double));
  cudaMalloc(&dummy_out_double, 10 * sizeof(double));
  cudaMalloc(&d_out_double, 10 * sizeof(double));
  cudaMalloc(&d_in_double, 10 * sizeof(double));

  TEST_2_D(add_kernel_7, dim3(1), dim3(5, 1, 1), 0, false, "a, b", dummy_out_double, dummy_in_double, d_out_double, d_in_double, 10); // CHECK-EXEC: 50.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00

  double *val, *d_val;
  cudaMalloc(&val, sizeof(double));
  cudaMalloc(&d_val, sizeof(double));

  double *fives = (double*)malloc(10 * sizeof(double));
  double *zeros = (double*)malloc(10 * sizeof(double));
  for(int i = 0; i < 10; i++) { fives[i] = 5; zeros[i] = 0; }
  
  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto test_device = clad::gradient(kernel_with_device_call, "out, val");
  test_device.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, 5, d_out_double, d_val);
  double *res = (double*)malloc(10 * sizeof(double));
  cudaMemcpy(res, d_val, sizeof(double), cudaMemcpyDeviceToHost); // no need for synchronization before or after, 
                                                                  // as the cudaMemcpy call is queued after the kernel call 
                                                                  // on the default stream and the cudaMemcpy call is blocking
  printf("%0.2f\n", *res); // CHECK-EXEC: 50.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto test_device_2 = clad::gradient(kernel_with_device_call_2, "out, val");
  test_device_2.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, 5, d_out_double, d_val);
  cudaMemcpy(res, d_val, sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f\n", *res); // CHECK-EXEC: 50.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto check_dup = clad::gradient(dup_kernel_with_device_call_2, "out, val"); // check that the pullback function is not regenerated
  check_dup.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, 5, d_out_double, d_val);
  cudaMemcpy(res, d_val, sizeof(double), cudaMemcpyDeviceToHost);
  printf("%s\n", cudaGetErrorString(cudaGetLastError())); // CHECK-EXEC: no error
  printf("%0.2f\n", *res); // CHECK-EXEC: 50.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto test_device_3 = clad::gradient(kernel_with_device_call_2, "out, in");
  test_device_3.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, 5, d_out_double, d_in_double);
  cudaMemcpy(res, d_in_double, 10 * sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f, %0.2f, %0.2f\n", res[0], res[1], res[2]); // CHECK-EXEC: 5.00, 5.00, 5.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto test_device_4 = clad::gradient(kernel_with_device_call_3);
  test_device_4.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, val, d_out_double, d_in_double, d_val);
  cudaMemcpy(res, d_in_double, 10 * sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f, %0.2f, %0.2f\n", res[0], res[1], res[2]); // CHECK-EXEC: 5.00, 5.00, 5.00
  cudaMemcpy(res, d_val, sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f\n", *res); // CHECK-EXEC: 50.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto local_param = clad::gradient(fn1, "out, val");
  local_param.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, 5, d_out_double, d_val);
  cudaMemcpy(res, d_val, sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f\n", *res); // CHECK-EXEC: 50.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto test_kernel_call = clad::gradient(fn);
  test_kernel_call.execute(dummy_out_double, dummy_in_double, d_out_double, d_in_double);
  cudaMemcpy(res, d_in_double, sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f\n", *res); // CHECK-EXEC: 50.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto nested_device = clad::gradient(kernel_with_nested_device_call, "out, in");
  nested_device.execute_kernel(dim3(1), dim3(10, 1, 1), dummy_out_double, dummy_in_double, 5, d_out_double, d_in_double);
  cudaMemcpy(res, d_in_double, 10 * sizeof(double), cudaMemcpyDeviceToHost);
  printf("%0.2f, %0.2f, %0.2f\n", res[0], res[1], res[2]); // CHECK-EXEC: 5.00, 5.00, 5.00

  INIT(dummy_in_double, dummy_out_double, val, d_in_double, d_out_double, d_val);

  auto test_memory = clad::gradient(fn_memory);
  test_memory.execute(dummy_out_double, fives, d_out_double, zeros);
  printf("%0.2f, %0.2f, %0.2f\n", zeros[0], zeros[1], zeros[2]); // CHECK-EXEC: 60.00, 0.00, 0.00

  auto launch_kernel_4_test = clad::gradient(launch_add_kernel_4, "out, in");
  int *out_res = (int*)malloc(10 * sizeof(int));
  int *in_res = (int*)calloc(10, sizeof(int));
  int *zeros_int = (int*)calloc(10, sizeof(int));
  int *fives_int = (int*)malloc(10 * sizeof(int));
  for(int i = 0; i < 10; i++) { fives_int[i] = 5; out_res[i] = 5; }

  launch_kernel_4_test.execute(zeros_int, fives_int, 10, out_res, in_res);
  printf("%d, %d, %d\n", in_res[0], in_res[1], in_res[2]); // CHECK-EXEC: 5, 5, 5

  free(res);
  free(fives);
  free(zeros);
  free(fives_int);
  free(zeros_int);
  free(out_res);
  free(in_res);
  cudaFree(d_out_double);
  cudaFree(d_in_double);
  cudaFree(val);
  cudaFree(d_val);
  cudaFree(dummy_in_double);

  return 0;
}
