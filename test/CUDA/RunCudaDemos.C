// RUN: %cladclang_cuda -I%S/../../include --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath %cudaldflags -o VectorAddition.out %S/../../demos/CUDA/VectorAddition.cu 2>&1 | FileCheck -check-prefix CHECK_VECTOR_ADDITION %s
// REQUIRES: cuda-runtime
// CHECK_VECTOR_ADDITION: void vector_addition_grad(const thrust::device_vector<double> &x, const thrust::device_vector<double> &y, thrust::device_vector<double> &z, thrust::device_vector<double> *_d_x, thrust::device_vector<double> *_d_y, thrust::device_vector<double> *_d_z)
// CHECK_VECTOR_ADDITION: clad::custom_derivatives::thrust::transform_reverse_forw
// CHECK_VECTOR_ADDITION: clad::custom_derivatives::thrust::reduce_pullback
// CHECK_VECTOR_ADDITION: clad::custom_derivatives::thrust::transform_pullback

// RUN: ./VectorAddition.out | FileCheck -check-prefix CHECK_VECTOR_ADDITION_EXEC %s
// CHECK_VECTOR_ADDITION_EXEC: Running vector addition demo.
// CHECK_VECTOR_ADDITION_EXEC: Gradients of sum wrt initial x: 1 1 1 1 1 1 1 1 1 1 

// RUN: %cladclang_cuda -I%S/../../include --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath %cudaldflags -o ParticleSimulation.out %S/../../demos/CUDA/ParticleSimulation.cu 2>&1 | FileCheck -check-prefix CHECK_PARTICLE_SIMULATION %s
// CHECK_PARTICLE_SIMULATION: void run_simulation_grad(thrust::device_vector<double> &x, thrust::device_vector<double> &y, const thrust::device_vector<double> &vx, const thrust::device_vector<double> &vy, const thrust::device_vector<double> &dts, thrust::device_vector<double> &tmp, thrust::device_vector<double> &x_buffer, thrust::device_vector<double> &y_buffer, thrust::device_vector<double> *_d_x, thrust::device_vector<double> *_d_y, thrust::device_vector<double> *_d_vx, thrust::device_vector<double> *_d_vy, thrust::device_vector<double> *_d_dts, thrust::device_vector<double> *_d_tmp, thrust::device_vector<double> *_d_x_buffer, thrust::device_vector<double> *_d_y_buffer)
// CHECK_PARTICLE_SIMULATION: for (i = 0; i < n_steps; ++i)
// CHECK_PARTICLE_SIMULATION: clad::custom_derivatives::thrust::reduce_pullback
// CHECK_PARTICLE_SIMULATION: for (; _t0; _t0--)
// CHECK_PARTICLE_SIMULATION: clad::custom_derivatives::thrust::copy_pullback
// CHECK_PARTICLE_SIMULATION: clad::custom_derivatives::thrust::transform_pullback

// RUN: ./ParticleSimulation.out | FileCheck -check-prefix CHECK_PARTICLE_SIMULATION_EXEC %s
// CHECK_PARTICLE_SIMULATION_EXEC: Running particle simulation demo.
// CHECK_PARTICLE_SIMULATION_EXEC: Gradients of final x-pos sum wrt initial vx: 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5

// RUN: %cladclang_cuda -I%S/../../include --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath %cudaldflags -o LinearRegression.out %S/../../demos/CUDA/LinearRegression.cu 2>&1 | FileCheck -check-prefix CHECK_LINEAR_REGRESSION %s
// CHECK_LINEAR_REGRESSION: void linear_regression_loss_grad(const thrust::device_vector<double> &x, const thrust::device_vector<double> &w, double y_true, thrust::device_vector<double> *_d_x, thrust::device_vector<double> *_d_w, double *_d_y_true) {
// CHECK_LINEAR_REGRESSION-NEXT:     double _d_y_pred = 0.;
// CHECK_LINEAR_REGRESSION-NEXT:     double y_pred = thrust::inner_product(std::begin(x), std::end(x), std::begin(w), 0.);
// CHECK_LINEAR_REGRESSION-NEXT:     double _d_error = 0.;
// CHECK_LINEAR_REGRESSION-NEXT:     double error = y_pred - y_true;
// CHECK_LINEAR_REGRESSION-NEXT:     {
// CHECK_LINEAR_REGRESSION-NEXT:         _d_error += 1 * error;
// CHECK_LINEAR_REGRESSION-NEXT:         _d_error += error * 1;
// CHECK_LINEAR_REGRESSION-NEXT:     }
// CHECK_LINEAR_REGRESSION-NEXT:     {
// CHECK_LINEAR_REGRESSION-NEXT:         _d_y_pred += _d_error;
// CHECK_LINEAR_REGRESSION-NEXT:         *_d_y_true += -_d_error;
// CHECK_LINEAR_REGRESSION-NEXT:     }
// CHECK_LINEAR_REGRESSION-NEXT:     {
// CHECK_LINEAR_REGRESSION-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r0 = std::begin((*_d_x));
// CHECK_LINEAR_REGRESSION-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r1 = std::end((*_d_x));
// CHECK_LINEAR_REGRESSION-NEXT:         thrust::detail::vector_base<double, thrust::device_allocator<double> >::const_iterator _r2 = std::begin((*_d_w));
// CHECK_LINEAR_REGRESSION-NEXT:         double _r3 = 0.;
// CHECK_LINEAR_REGRESSION-NEXT:         clad::custom_derivatives::thrust::inner_product_pullback(std::begin(x), std::end(x), std::begin(w), 0., _d_y_pred, &_r0, &_r1, &_r2, &_r3);
// CHECK_LINEAR_REGRESSION-NEXT:     }
// CHECK_LINEAR_REGRESSION-NEXT: }

// RUN: ./LinearRegression.out | FileCheck -check-prefix CHECK_LINEAR_REGRESSION_EXEC %s
// CHECK_LINEAR_REGRESSION_EXEC: Running linear regression demo.
// CHECK_LINEAR_REGRESSION_EXEC: Gradients of loss wrt weights (w): -9 -18 -27 -36 -45 -54 -63 -72 -81 -90
