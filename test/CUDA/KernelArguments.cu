// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:   --cuda-gpu-arch=%cudaarch %cudaldflags -o%t -Xclang -verify %s
// RUN: %if cuda-runtime %{ %cudarun %t | %filecheck_exec %s %}
// REQUIRES: cuda-compile
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
  do {                                                                         \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s\n", cudaGetErrorString(status));                  \
      std::exit(2);                                                            \
    }                                                                          \
  } while (false)

struct First {
  float unused;
};
struct Second {
  float value;
};
struct Derived : First, Second {};

__global__ void arguments(float* out, int flag, float scale,
                          const Second* input, const int* optional) {
  out[0] = flag ? 11 : 7;
  out[1] = scale;
  out[2] = input->value;
  out[3] = optional ? *optional : -1;
}

struct Scalar {
  int unrelated = 1;
  operator float() && { return 3.5f; }
};

struct Payload {
  float value;
  int count;
};
__global__ void aggregate(float* out, Payload data) {
  *out = data.value * data.count;
}

__global__ void no_arguments() {}

float host_offset() { return 0.5f; }
float host_reference(const float& value, float (&offset)()) {
  return value + offset();
}

__global__ void scaled(float* out, float x, int flag) {
  *out = x * x * (flag ? 2 : 1);
}

int main() {
  float *out, *seed, *dx;
  Derived* input;
  int* optional;
  CUDA_CHECK(cudaMallocManaged(&out, 4 * sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&seed, sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&dx, sizeof(float)));
  CUDA_CHECK(cudaMallocManaged(&input, sizeof(Derived)));
  CUDA_CHECK(cudaMallocManaged(&optional, sizeof(int)));
  input->unused = 1;
  input->value = 4.25f;
  *optional = 9;

  // Reading an int directly from the bool's address would also copy padding.
  struct Flag {
    bool value;
    unsigned char padding[sizeof(int) - sizeof(bool)];
  } flag{false, {1}};

  int failures = 0;
  auto check = [&](const char* name, float first, float last) {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("%s: %g %g %g %g\n", name, out[0], out[1], out[2], out[3]);
    failures += out[0] != first || out[1] != 3.5f || out[2] != 4.25f ||
                out[3] != last;
  };

  using Fn = decltype(&arguments);
  clad::CladFunction<Fn> direct(arguments, "", nullptr, true);
  clad::CladFunction<Fn, clad::ExtractFunctorTraits_t<Fn>, true> padded(
      arguments, "", nullptr, true);

  direct.execute_kernel(dim3(1), dim3(1), out, 0, 3.5f,
                        static_cast<const Second*>(input), optional);
  check("exact", 7, 9); // CHECK-EXEC: exact: 7 3.5 4.25 9

  direct.execute_kernel(dim3(1), dim3(1), out, flag.value, 3.5, input, optional);
  check("converted", 7, 9); // CHECK-EXEC: converted: 7 3.5 4.25 9

  padded.execute_kernel(dim3(1), dim3(1), out, flag.value, 3.5, input);
  check("padded", 7, -1); // CHECK-EXEC: padded: 7 3.5 4.25 -1

  direct.execute_kernel(dim3(1), dim3(1), out, 0, Scalar{}, input, optional);
  check("user conversion", 7, 9); // CHECK-EXEC: user conversion: 7 3.5 4.25 9
  padded.execute_kernel(dim3(1), dim3(1), out, 0, Scalar{}, input);
  check("padded user conversion", 7, -1);
  // CHECK-EXEC: padded user conversion: 7 3.5 4.25 -1

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  direct.execute_kernel(dim3(1), dim3(1), std::size_t(64), stream, out,
                        flag.value, 3.5, input, optional);
  check("stream", 7, 9); // CHECK-EXEC: stream: 7 3.5 4.25 9
  padded.execute_kernel(dim3(1), dim3(1), std::size_t(64), stream, out,
                        flag.value, 3.5, input);
  check("padded stream", 7, -1); // CHECK-EXEC: padded stream: 7 3.5 4.25 -1
  CUDA_CHECK(cudaStreamDestroy(stream));

  clad::CladFunction<decltype(&aggregate)> by_value(aggregate, "", nullptr, true);
  by_value.execute_kernel(dim3(1), dim3(1), out, Payload{2.5f, 4});
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("aggregate: %g\n", *out); // CHECK-EXEC: aggregate: 10
  failures += *out != 10;
  clad::CladFunction<decltype(&no_arguments)> empty(no_arguments, "", nullptr,
                                                 true);
  empty.execute_kernel(dim3(1), dim3(1));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const float host_value = 3.5f;
  clad::CladFunction<decltype(&host_reference)> reference(host_reference, "",
                                                       nullptr, false);
  float reference_result = reference.execute(host_value, host_offset);
  std::printf("host reference: %g\n", reference_result);
  // CHECK-EXEC: host reference: 4
  failures += reference_result != 4.0f;

  *seed = 1;
  *dx = 0;
  auto gradient = clad::gradient(scaled, "out,x");
  gradient.execute_kernel(dim3(1), dim3(1), out, 2.0, flag.value, seed, dx);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("gradient: %g\n", *dx); // CHECK-EXEC: gradient: 4
  failures += *dx != 4;

  CUDA_CHECK(cudaFree(optional));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(dx));
  CUDA_CHECK(cudaFree(seed));
  CUDA_CHECK(cudaFree(out));
  std::printf("failures: %d\n", failures); // CHECK-EXEC: failures: 0
  return failures ? 3 : 0;
}
