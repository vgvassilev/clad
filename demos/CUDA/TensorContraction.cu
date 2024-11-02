//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to calculate a tensor contraction and its gradient.
//
// author:  Christina Koutsou <christinakoutsou22-at-gmail.com>
//----------------------------------------------------------------------------//

// RUN: /path/to/clang -std=c++17 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang   \
//          /path/to/libclad.so -I/path/to/clad/include -I/path/to/cuda/include            \
//          TensorContraction.cu -o TensorContraction                                      \
//          --cuda-path=path/to/cuda  --cuda-gpu-arch=%cuda_arch                           \
//          -L/path/to/cuda/lib64 -lcudart_static -ldl -lrt -pthread -lm -lstdc++
// RUN: ./TensorContraction

#include "clad/Differentiator/Differentiator.h"

typedef unsigned long long int size_type;

__device__ void computeStartStep(size_type& A_start, size_type& A_step,
                                 size_type& B_start, size_type& B_step,
                                 const int idx, const size_type A_dim[3],
                                 const size_type B_dim[3],
                                 const int contractDimA,
                                 const int contractDimB) {
  size_type A_a, A_b, A_c, B_d, B_e, B_f;

  switch (contractDimA) {
  case 0:
    A_b = idx / (A_dim[2] * B_dim[(contractDimB + 1) % 3] *
                 B_dim[(contractDimB + 2) % 3]);
    A_c = (idx /
           (B_dim[(contractDimB + 1) % 3] * B_dim[(contractDimB + 2) % 3])) %
          A_dim[2];
    A_start = 0 + A_b * A_dim[2] + A_c;
    A_step = A_dim[1] * A_dim[2];
    break;
  case 1:
    A_a = idx / (A_dim[2] * B_dim[(contractDimB + 1) % 3] *
                 B_dim[(contractDimB + 2) % 3]);
    A_c = (idx /
           (B_dim[(contractDimB + 1) % 3] * B_dim[(contractDimB + 2) % 3])) %
          A_dim[2];
    A_start = A_a * A_dim[1] * A_dim[2] + 0 + A_c;
    A_step = A_dim[2];
    break;
  case 2:
    A_a = idx / (A_dim[1] * B_dim[(contractDimB + 1) % 3] *
                 B_dim[(contractDimB + 2) % 3]);
    A_b = (idx /
           (B_dim[(contractDimB + 1) % 3] * B_dim[(contractDimB + 2) % 3])) %
          A_dim[1];
    A_start = A_a * A_dim[1] * A_dim[2] + A_b * A_dim[2];
    A_step = 1;
    break;
  }

  switch (contractDimB) {
  case 0:
    B_e = (idx / B_dim[2]) % B_dim[1];
    B_f = idx % B_dim[2];
    B_start = 0 + B_e * B_dim[2] + B_f;
    B_step = B_dim[1] * B_dim[2];
    break;
  case 1:
    B_d = (idx / B_dim[2]) % B_dim[0];
    B_f = idx % B_dim[2];
    B_start = B_d * B_dim[2] * B_dim[1] + 0 + B_f;
    B_step = B_dim[2];
    break;
  case 2:
    B_d = (idx / B_dim[1]) % B_dim[0];
    B_e = idx % B_dim[1];
    B_start = B_d * B_dim[2] * B_dim[1] + B_e * B_dim[2];
    B_step = 1;
    break;
  }
}

__global__ void tensorContraction3D(float* C, const float* A, const float* B,
                                    const size_type* A_dim,
                                    const size_type* B_dim,
                                    const int contractDimA,
                                    const int contractDimB) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Each thread computes one element of the output tensor
  int totalElements =
      A_dim[(contractDimA + 1) % 3] * A_dim[(contractDimA + 2) % 3] *
      B_dim[(contractDimB + 1) % 3] * B_dim[(contractDimB + 2) % 3];
  if (idx < totalElements) {
    size_type A_start, B_start, A_step, B_step;
    size_type A_a, A_b, A_c, B_d, B_e, B_f;

    computeStartStep(A_start, A_step, B_start, B_step, idx, A_dim, B_dim,
                     contractDimA, contractDimB);

    float sum = 0.0f;
    for (int i = 0; i < A_dim[contractDimA];
         i++) // A_dim[contractDimA] == B_dim[contractDimB]
      sum += A[A_start + (i * A_step)] * B[B_start + (i * B_step)];

    C[idx] = sum;
  }
}

void launchTensorContraction3D(float* C, const float* A, const float* B,
                               const size_type D1, const size_type D2,
                               const size_type D3, const size_type D4,
                               const size_type D5) {
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  const size_type A_size = D1 * D2 * D3 * sizeof(float);
  const size_type B_size = D3 * D4 * D5 * sizeof(float);
  const size_type C_size = D1 * D2 * D4 * D5 * sizeof(float);

  // Allocate device memory and copy data from host to device
  cudaMalloc(&d_A, A_size);
  cudaMalloc(&d_B, B_size);
  cudaMalloc(&d_C, C_size);
  cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, B_size, cudaMemcpyHostToDevice);

  const size_type A_dim[3] = {D1, D2, D3};
  const size_type B_dim[3] = {D3, D4, D5};

  size_type *d_A_dim = nullptr, *d_B_dim = nullptr;
  cudaMalloc(&d_A_dim, 3 * sizeof(size_type));
  cudaMalloc(&d_B_dim, 3 * sizeof(size_type));
  cudaMemcpy(d_A_dim, A_dim, 3 * sizeof(size_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_dim, B_dim, 3 * sizeof(size_type), cudaMemcpyHostToDevice);

  // Launch the kernel
  tensorContraction3D<<<1, 256>>>(d_C, d_A, d_B, d_A_dim, d_B_dim,
                                  /*contractDimA=*/2, /*contractDimB=*/0);

  // Copy the result from device to host
  cudaMemcpy(C, d_C, C_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_A_dim);
  cudaFree(d_B_dim);
}

int main() {
  const size_type D1 = 2, D2 = 3, D3 = 4, D4 = 3, D5 = 2;

  float A[D1][D2][D3] = {
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
      {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}};

  float B[D3][D4][D5] = {{{1, 2}, {3, 4}, {5, 6}},
                         {{7, 8}, {9, 10}, {11, 12}},
                         {{13, 14}, {15, 16}, {17, 18}},
                         {{19, 20}, {21, 22}, {23, 24}}};

  float C[D1][D2][D4][D5] = {0}; // Result tensor

  // Compute the gradient
  auto tensor_grad = clad::gradient(launchTensorContraction3D, "C, A, B");

  // Initialize the gradient inputs
  float gradC[D1][D2][D4][D5] = {{{{1, 1}, {1, 1}, {1, 1}},
                                  {{1, 1}, {1, 1}, {1, 1}},
                                  {{1, 1}, {1, 1}, {1, 1}}},
                                 {{{1, 1}, {1, 1}, {1, 1}},
                                  {{1, 1}, {1, 1}, {1, 1}},
                                  {{1, 1}, {1, 1}, {1, 1}}}};
  float gradA[D1][D2][D3] = {0};
  float gradB[D3][D4][D5] = {0};

  // Execute tensor contraction and its gradient
  tensor_grad.execute(&C[0][0][0][0], &A[0][0][0], &B[0][0][0], D1, D2, D3, D4,
                      D5, &gradC[0][0][0][0], &gradA[0][0][0], &gradB[0][0][0]);

  // Print the result
  std::cout << "Result C:\n";
  for (int i = 0; i < D1; ++i) {
    for (int j = 0; j < D2; ++j) {
      for (int k = 0; k < D4; ++k) {
        for (int l = 0; l < D5; ++l)
          std::cout << C[i][j][k][l] << " ";
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "Result C_grad w.r.t. A:\n";
  for (int i = 0; i < D1; ++i) {
    for (int j = 0; j < D2; ++j) {
      for (int k = 0; k < D3; ++k)
        std::cout << gradA[i][j][k] << " ";
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "Result C_grad w.r.t. B:\n";
  for (int i = 0; i < D3; ++i) {
    for (int j = 0; j < D4; ++j) {
      for (int k = 0; k < D5; ++k)
        std::cout << gradB[i][j][k] << " ";
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  return 0;
}

// CHECK-EXEC: Result C:
// CHECK-NEXT: 130 140
// CHECK-NEXT: 150 160
// CHECK-NEXT: 170 180
// CHECK-NEXT:
// CHECK-NEXT: 290 316
// CHECK-NEXT: 342 368
// CHECK-NEXT: 394 420
// CHECK-NEXT:
// CHECK-NEXT: 450 492
// CHECK-NEXT: 534 576
// CHECK-NEXT: 618 660
// CHECK-NEXT:
// CHECK-NEXT:
// CHECK-NEXT: 610 668
// CHECK-NEXT: 726 784
// CHECK-NEXT: 842 900
// CHECK-NEXT:
// CHECK-NEXT: 770 844
// CHECK-NEXT: 918 992
// CHECK-NEXT: 1066 1140
// CHECK-NEXT:
// CHECK-NEXT: 930 1020
// CHECK-NEXT: 1110 1200
// CHECK-NEXT: 1290 1380

// CHECK-EXEC: Result C_grad w.r.t. A:
// CHECK-NEXT: 21 57 93 129
// CHECK-NEXT: 21 57 93 129
// CHECK-NEXT: 21 57 93 129
// CHECK-NEXT:
// CHECK-NEXT: 21 57 93 129
// CHECK-NEXT: 21 57 93 129
// CHECK-NEXT: 21 57 93 129
// CHECK-NEXT:
// CHECK-EXEC: Result C_grad w.r.t. B:
// CHECK-NEXT: 66 66
// CHECK-NEXT: 66 66
// CHECK-NEXT: 66 66
// CHECK-NEXT:
// CHECK-NEXT: 72 72
// CHECK-NEXT: 72 72
// CHECK-NEXT: 72 72
// CHECK-NEXT:
// CHECK-NEXT: 78 78
// CHECK-NEXT: 78 78
// CHECK-NEXT: 78 78
// CHECK-NEXT:
// CHECK-NEXT: 84 84
// CHECK-NEXT: 84 84
// CHECK-NEXT: 84 84