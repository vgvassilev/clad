Automatically compute reverse-mode derivatives of CUDA functions with Clad
******************************************************************************

Clad offers the ability to differentiate CUDA functions in reverse-mode. Specifically,
Clad can differentiate functions that are marked with either the `__device__` or `__global__` specifier.

For the kernels, since they are void functions, the output parameter must be included in the parameter list of `clad::gradient`.
To execute the kernel, the user has to call the `execute_kernel` method of the `CladFunction` object and provide the grid configuration.
The size of the shared memory to dynamically allocate for the kernel and the stream to use for its execution are appointed the default values of `0` and `nulltptr` respectively, 
if not explicitly specified. Note that either none of these two arguments or both of them must be provided to the `execute_kernel` call. 
Clad does not handle cases where only one of the two is provided, even if the order is correct.

.. code-block:: cpp

    #include "clad/Differentiator/Differentiator.h"

    auto kernel_grad = clad::gradient(kernel, "in, out"); // compute the derivative of out w.r.t in
    // Option 1:
    kernel_grad.execute_kernel(gridDim, blockDim, sharedMem, stream, in, out, in_grad, out_grad);
    // Option 2:
    kernel_grad.execute_kernel(gridDim, blockDim, in, out, in_grad, out_grad);


CUDA features supported by Clad
================================================

Clad supports the following CUDA features:
* The commonly used CUDA built-in variables `threadIdx`, `blockIdx`, `blockDim`, `gridDim` and `warpSize` 
* The CUDA host functions `cudaMalloc`, `cudaMemcpy` and `cudaFree`

To use CUDA math functions, the user must define the equivalent pullback function in Clad's CUDA custom derivatives:

.. code-block:: cpp

    // In `clad/include/clad/Differentiator/BuiltinDerivativesCUDA.cuh`

    namespace clad {

    namespace custom_derivatives {

    __device__ inline void __fdividef_pullback(float a, float b, float d_y,
                                            float* d_a, float* d_b) {
    *d_a += (1.F / b) * d_y;
    *d_b += (-a / (b * b)) * d_y;
    }

    }
    }



CUDA features not yet supported by Clad
================================================

The following CUDA features are not yet supported:
* The use of shared memory within the original function
* Synchronization primitives like `__syncthreads()` and `cudaDeviceSynchronize()`
* Other CUDA host functions apart from those listed in the previous section


Demos
================================================

For examples of using Clad with CUDA, see the `clad/demos/CUDA` folder.