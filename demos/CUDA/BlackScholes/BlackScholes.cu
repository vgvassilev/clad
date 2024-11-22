/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample evaluates fair call and put prices for a
 * given set of European options by Black-Scholes formula.
 * See supplied whitepaper for more explanations.
 */

#include <helper_functions.h>  // helper functions for string parsing
#include <helper_cuda.h>  // helper functions CUDA error checking and initialization

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float *h_CallResult, float *h_PutResult,
                                float *h_StockPrice, float *h_OptionStrike,
                                float *h_OptionYears, float Riskfree,
                                float Volatility, int optN);

////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on GPU
////////////////////////////////////////////////////////////////////////////////
#include "BlackScholes_kernel.cuh"

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int OPT_N = 4000000;
const int NUM_ITERATIONS = 512;

const int OPT_SZ = OPT_N * sizeof(float);
const float RISKFREE = 0.02f;
const float VOLATILITY = 0.30f;

#define DIV_UP(a, b) (((a) + (b)-1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

/*
 * DISCLAIMER: The following file has been slightly modified to ensure
 * compatibility with Clad and to serve as a Clad demo. Specifically, parts of
 * the original `main` function have been moved to a separate function to use
 * `clad::gradient` on. Furthermore, Clad cannot clone checkCudaErrors
 * successfully, so these calls have been omitted. The same applies to the
 * cudaDeviceSynchronize function. New helper functions are included in another
 * file and invoked here to verify the gradient's results. Since Clad cannot
 * handle timers at the moment, the time measurement is included in
 * `main` and doesn't time exclusively the original kernel execution, but the
 * whole `launch` function and its gradient are timed in this version.
 *
 * The original file is available in NVIDIA's cuda-samples repository on GitHub.
 *
 * Relevant documentation regarding the problem at hand can be found in NVIDIA's
 * cuda-samples repository. Using Clad, we compute some of the Greeks
 * (sensitivities) for the Black-Scholes model and verify them against
 * approximations of their theoretical values as denoted in Wikipedia
 * (https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model).
 *
 * To build and run the demo, use the following command: make run
 */

#include "clad/Differentiator/Differentiator.h"
#include <helper_grad_verify.h>

void launch(float* h_CallResultCPU, float* h_CallResultGPU,
            float* h_PutResultCPU, float* h_PutResultGPU, float* h_StockPrice,
            float* h_OptionStrike, float* h_OptionYears) {

  //'d_' prefix - GPU (device) memory space
  float
      // Results calculated by GPU
      *d_CallResult = nullptr,
      *d_PutResult = nullptr,
      // GPU instance of input data
      *d_StockPrice = nullptr, *d_OptionStrike = nullptr,
      *d_OptionYears = nullptr;

  printf("...allocating GPU memory for options.\n");
  cudaMalloc((void**)&d_CallResult, OPT_SZ);
  cudaMalloc((void**)&d_PutResult, OPT_SZ);
  cudaMalloc((void**)&d_StockPrice, OPT_SZ);
  cudaMalloc((void**)&d_OptionStrike, OPT_SZ);
  cudaMalloc((void**)&d_OptionYears, OPT_SZ);

  // Copy options data to GPU memory for further processing
  printf("...copying input data to GPU mem.\n");
  cudaMemcpy(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice);
  printf("Data init done.\n\n");

  printf("Executing Black-Scholes GPU kernel (%i iterations)...\n",
         NUM_ITERATIONS);
  int i;
  for (i = 0; i < NUM_ITERATIONS; i++) {
    BlackScholesGPU<<<DIV_UP((OPT_N / 2), 128), 128 /*480, 128*/>>>(
        (float2 *)d_CallResult, (float2 *)d_PutResult, (float2 *)d_StockPrice,
        (float2 *)d_OptionStrike, (float2 *)d_OptionYears, RISKFREE, VOLATILITY,
        OPT_N);
  }

  // Both call and put is calculated

  printf("\nReading back GPU results...\n");
  // Read back GPU results to compare them to CPU results
  cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_PutResultGPU, d_PutResult, OPT_SZ, cudaMemcpyDeviceToHost);

  printf("...releasing GPU memory.\n");
  cudaFree(d_OptionYears);
  cudaFree(d_OptionStrike);
  cudaFree(d_StockPrice);
  cudaFree(d_PutResult);
  cudaFree(d_CallResult);
}

int main(int argc, char **argv) {
  // Start logs
  printf("[%s] - Starting...\n", argv[0]);

  //'h_' prefix - CPU (host) memory space
  float
      // Results calculated by CPU for reference
      *h_CallResultCPU,
      *h_PutResultCPU,
      // CPU copy of GPU results
      *h_CallResultGPU, *h_PutResultGPU,
      // CPU instance of input data
      *h_StockPrice, *h_OptionStrike, *h_OptionYears;

  double delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;

  StopWatchInterface *hTimer = NULL;
  int i;

  findCudaDevice(argc, (const char **)argv);

  sdkCreateTimer(&hTimer);

  printf("Initializing data...\n");
  printf("...allocating CPU memory for options.\n");
  h_CallResultCPU = (float *)malloc(OPT_SZ);
  h_PutResultCPU = (float *)malloc(OPT_SZ);
  h_CallResultGPU = (float *)malloc(OPT_SZ);
  h_PutResultGPU = (float *)malloc(OPT_SZ);
  h_StockPrice = (float *)malloc(OPT_SZ);
  h_OptionStrike = (float *)malloc(OPT_SZ);
  h_OptionYears = (float *)malloc(OPT_SZ);

  printf("...generating input data in CPU mem.\n");
  srand(5347);

  // Generate options set
  for (i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }

  /*******************************************************************************/

  // Compute gradients
  auto callGrad = clad::gradient(
      launch, "h_CallResultGPU, h_StockPrice, h_OptionStrike, h_OptionYears");
  auto putGrad = clad::gradient(
      launch, "h_PutResultGPU, h_StockPrice, h_OptionStrike, h_OptionYears");

  // Declare and initialize the derivatives
  float* d_CallResultGPU = (float*)malloc(OPT_SZ);
  float* d_PutResultGPU = (float*)malloc(OPT_SZ);
  float* d_StockPrice = (float*)calloc(OPT_N, sizeof(float));
  float* d_OptionStrike = (float*)calloc(OPT_N, sizeof(float));
  float* d_OptionYears = (float*)calloc(OPT_N, sizeof(float));

  for (int i = 0; i < OPT_N; i++) {
    d_CallResultGPU[i] = 1.0f;
    d_PutResultGPU[i] = 1.0f;
  }

  /*******************************************************************************/

  checkCudaErrors(cudaDeviceSynchronize());
  sdkResetTimer(&hTimer);
  sdkStartTimer(&hTimer);
  // Compute the values and derivatives of the price of the call options
  callGrad.execute(h_CallResultCPU, h_CallResultGPU, h_PutResultCPU,
                   h_PutResultGPU, h_StockPrice, h_OptionStrike, h_OptionYears,
                   d_CallResultGPU, d_StockPrice, d_OptionStrike,
                   d_OptionYears);

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&hTimer);
  gpuTime = sdkGetTimerValue(&hTimer) / NUM_ITERATIONS;

  // Both call and put is calculated
  printf("Options count             : %i     \n", 2 * OPT_N);
  printf("BlackScholesGPU() time    : %f msec\n", gpuTime);
  printf("Effective memory bandwidth: %f GB/s\n",
         ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (gpuTime * 1E-3));
  printf("Gigaoptions per second    : %f     \n\n",
         ((double)(2 * OPT_N) * 1E-9) / (gpuTime * 1E-3));

  printf(
      "BlackScholes, Throughput = %.4f GOptions/s, Time = %.5f s, Size = %u "
      "options, NumDevsUsed = %u, Workgroup = %u\n",
      (((double)(2.0 * OPT_N) * 1.0E-9) / (gpuTime * 1.0E-3)), gpuTime * 1e-3,
      (2 * OPT_N), 1, 128);

  printf("Checking the results...\n");
  printf("...running CPU calculations.\n\n");
  // Calculate options values on CPU
  BlackScholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike,
                  h_OptionYears, RISKFREE, VOLATILITY, OPT_N);

  printf("Comparing the results...\n");
  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  sum_delta = 0;
  sum_ref = 0;
  max_delta = 0;

  for (i = 0; i < OPT_N; i++) {
    ref = h_CallResultCPU[i];
    delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

    if (delta > max_delta) {
      max_delta = delta;
    }

    sum_delta += delta;
    sum_ref += fabs(ref);
  }

  L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n", L1norm);
  printf("Max absolute error: %E\n\n", max_delta);

  // Verify delta
  computeL1norm<Call, Delta>(h_StockPrice, h_OptionStrike, h_OptionYears,
                             d_StockPrice);
  // Verify derivatives with respect to the Strike price
  computeL1norm<Call, dX>(h_StockPrice, h_OptionStrike, h_OptionYears,
                          d_OptionStrike);
  // Verify theta
  computeL1norm<Call, Theta>(h_StockPrice, h_OptionStrike, h_OptionYears,
                             d_OptionYears);
  /*******************************************************************************/
  // Re-initialize data for next gradient call
  for (int i = 0; i < OPT_N; i++)
  {
      h_CallResultCPU[i] = 0.0f;
      h_PutResultCPU[i] = -1.0f;
      d_CallResultGPU[i] = 1.0f;
      d_PutResultGPU[i] = 1.0f;
  }
  for (int i = 0; i < OPT_N; i++)
  {
      d_StockPrice[i] = 0.f;
      d_OptionStrike[i] = 0.f;
      d_OptionYears[i] = 0.f;
  }
  // Compute the values and derivatives of the price of the Put options
  putGrad.execute(h_CallResultCPU, h_CallResultGPU, h_PutResultCPU,
                  h_PutResultGPU, h_StockPrice, h_OptionStrike, h_OptionYears,
                  d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears);
  // Verify delta
  computeL1norm<Put, Delta>(h_StockPrice, h_OptionStrike, h_OptionYears,
                            d_StockPrice);
  // Verify derivatives with respect to the Strike price
  computeL1norm<Put, dX>(h_StockPrice, h_OptionStrike, h_OptionYears,
                         d_OptionStrike);
  // Verify theta
  computeL1norm<Put, Theta>(h_StockPrice, h_OptionStrike, h_OptionYears,
                            d_OptionYears);
  /*******************************************************************************/

  printf("Shutting down...\n");
  printf("...releasing CPU memory.\n");
  free(h_OptionYears);
  free(h_OptionStrike);
  free(h_StockPrice);
  free(h_PutResultGPU);
  free(h_CallResultGPU);
  free(h_PutResultCPU);
  free(h_CallResultCPU);
  free(d_OptionYears);
  free(d_OptionStrike);
  free(d_StockPrice);
  free(d_PutResultGPU);
  free(d_CallResultGPU);
  sdkDeleteTimer(&hTimer);
  printf("Shutdown done.\n");

  printf("\n[BlackScholes] - Test Summary\n");

  if (L1norm > 1e-6) {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");
  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}