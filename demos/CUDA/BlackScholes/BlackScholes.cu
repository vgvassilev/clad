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

/*
 * DISCLAIMER: The following file has been modified slightly to make it
 * compatible with Clad. The original file can be found at NVIDIA's cuda-samples
 * repository at GitHub.
 *
 * Relevant documentation regarding the problem at hand can be found at NVIDIA's
 * cuda-samples repository. With the use of Clad, we compute some of the Greeks
 * (sensitivities) for Black-Scholes and verify them using the
 * theoretical values as denoted in Wikipedia
 * (https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model).
 *
 * To build and run the demo, run the following command: make run
 */

#include "clad/Differentiator/Differentiator.h"

#include <helper_cuda.h> // helper functions CUDA error checking and initialization
#include <helper_functions.h> // helper functions for string parsing

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options on CPU
////////////////////////////////////////////////////////////////////////////////
extern "C" void BlackScholesCPU(float* h_CallResult, float* h_PutResult,
                                float* h_StockPrice, float* h_OptionStrike,
                                float* h_OptionYears, float Riskfree,
                                float Volatility, int optN);
extern "C" double CND(double d);

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

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

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

  cudaMalloc((void**)&d_CallResult, OPT_SZ);
  cudaMalloc((void**)&d_PutResult, OPT_SZ);
  cudaMalloc((void**)&d_StockPrice, OPT_SZ);
  cudaMalloc((void**)&d_OptionStrike, OPT_SZ);
  cudaMalloc((void**)&d_OptionYears, OPT_SZ);

  // Copy options data to GPU memory for further processing
  cudaMemcpy(d_StockPrice, h_StockPrice, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_OptionStrike, h_OptionStrike, OPT_SZ, cudaMemcpyHostToDevice);
  cudaMemcpy(d_OptionYears, h_OptionYears, OPT_SZ, cudaMemcpyHostToDevice);

  BlackScholesGPU<<<DIV_UP((OPT_N / 2), 128), 128 /*480, 128*/>>>(
      (float2*)d_CallResult, (float2*)d_PutResult, (float2*)d_StockPrice,
      (float2*)d_OptionStrike, (float2*)d_OptionYears, RISKFREE, VOLATILITY,
      OPT_N);

  // Both call and put is calculated

  // Read back GPU results to compare them to CPU results
  cudaMemcpy(h_CallResultGPU, d_CallResult, OPT_SZ, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_PutResultGPU, d_PutResult, OPT_SZ, cudaMemcpyDeviceToHost);

  // Calculate options values on CPU
  BlackScholesCPU(h_CallResultCPU, h_PutResultCPU, h_StockPrice, h_OptionStrike,
                  h_OptionYears, RISKFREE, VOLATILITY, OPT_N);

  cudaFree(d_OptionYears);
  cudaFree(d_OptionStrike);
  cudaFree(d_StockPrice);
  cudaFree(d_PutResult);
  cudaFree(d_CallResult);
}

double d1(double S, double X, double T) {
  return (log(S / X) + (RISKFREE + 0.5 * VOLATILITY * VOLATILITY) * T) /
         (VOLATILITY * sqrt(T));
}

double N_prime(double d) {
  const double RSQRT2PI =
      0.39894228040143267793994605993438; // 1 / sqrt(2 * PI)
  return RSQRT2PI * exp(-0.5 * d * d);
}

enum Greek { Delta, dX, Theta };

double computeL1norm_Call(float* S, float* X, float* T, float* d, Greek greek) {
  double delta, ref, sum_delta, sum_ref;
  sum_delta = 0;
  sum_ref = 0;
  switch (greek) {
  case Delta:
    for (int i = 0; i < OPT_N; i++) {
      double d1_val = d1(S[i], X[i], T[i]);
      ref = CND(d1_val);
      delta = fabs(d[i] - ref);
      sum_delta += delta;
      sum_ref += fabs(ref);
    }
    break;
  case dX:
    for (int i = 0; i < OPT_N; i++) {
      double T_val = T[i];
      double d1_val = d1(S[i], X[i], T_val);
      double d2_val = d1_val - VOLATILITY * sqrt(T_val);
      double expRT = exp(-RISKFREE * T_val);
      ref = -expRT * CND(d2_val);
      delta = fabs(d[i] - ref);
      sum_delta += delta;
      sum_ref += fabs(ref);
    }
    break;
  case Theta:
    for (int i = 0; i < OPT_N; i++) {
      double S_val = S[i], X_val = X[i], T_val = T[i];
      double d1_val = d1(S_val, X_val, T_val);
      double d2_val = d1_val - VOLATILITY * sqrt(T_val);
      double expRT = exp(-RISKFREE * T_val);
      ref =
          (S_val * N_prime(d1_val) * VOLATILITY) / (2 * sqrt(T_val)) +
          RISKFREE * X_val * expRT *
              CND(d2_val); // theta is with respect to t, so -theta is the
                           // approximation of the derivative with respect to T
      delta = fabs(d[i] - ref);
      sum_delta += delta;
      sum_ref += fabs(ref);
    }
  }

  return sum_delta / sum_ref;
}

double computeL1norm_Put(float* S, float* X, float* T, float* d, Greek greek) {
  double delta, ref, sum_delta, sum_ref;
  sum_delta = 0;
  sum_ref = 0;
  switch (greek) {
  case Delta:
    for (int i = 0; i < OPT_N; i++) {
      double d1_val = d1(S[i], X[i], T[i]);
      ref = CND(d1_val) - 1.0;
      delta = fabs(d[i] - ref);
      sum_delta += delta;
      sum_ref += fabs(ref);
    }
    break;
  case dX:
    for (int i = 0; i < OPT_N; i++) {
      double T_val = T[i];
      double d1_val = d1(S[i], X[i], T_val);
      double d2_val = d1_val - VOLATILITY * sqrt(T_val);
      double expRT = exp(-RISKFREE * T_val);
      ref = expRT * CND(-d2_val);
      delta = fabs(d[i] - ref);
      sum_delta += delta;
      sum_ref += fabs(ref);
    }
    break;
  case Theta:
    for (int i = 0; i < OPT_N; i++) {
      double S_val = S[i], X_val = X[i], T_val = T[i];
      double d1_val = d1(S_val, X_val, T_val);
      double d2_val = d1_val - VOLATILITY * sqrt(T_val);
      double expRT = exp(-RISKFREE * T_val);
      ref = (S_val * N_prime(d1_val) * VOLATILITY) / (2 * sqrt(T_val)) -
            RISKFREE * X_val * expRT * CND(-d2_val);
      delta = fabs(d[i] - ref);
      sum_delta += delta;
      sum_ref += fabs(ref);
    }
  }

  return sum_delta / sum_ref;
}

int main(int argc, char** argv) {
  float* h_CallResultCPU = (float*)malloc(OPT_SZ);
  float* h_PutResultCPU = (float*)malloc(OPT_SZ);
  float* h_CallResultGPU = (float*)malloc(OPT_SZ);
  float* h_PutResultGPU = (float*)malloc(OPT_SZ);
  float* h_StockPrice = (float*)malloc(OPT_SZ);
  float* h_OptionStrike = (float*)malloc(OPT_SZ);
  float* h_OptionYears = (float*)malloc(OPT_SZ);

  srand(5347);

  // Generate options set
  for (int i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }

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

  // Launch the kernel and the gradient

  // Compute the derivatives of the price of the call options
  callGrad.execute(h_CallResultCPU, h_CallResultGPU, h_PutResultCPU,
                   h_PutResultGPU, h_StockPrice, h_OptionStrike, h_OptionYears,
                   d_CallResultGPU, d_StockPrice, d_OptionStrike,
                   d_OptionYears);

  // Calculate max absolute difference and L1 distance
  // between CPU and GPU results
  double delta, ref, sum_delta, sum_ref, L1norm;
  sum_delta = 0;
  sum_ref = 0;

  for (int i = 0; i < OPT_N; i++) {
    ref = h_CallResultCPU[i];
    delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);
    sum_delta += delta;
    sum_ref += fabs(ref);
  }

  L1norm = sum_delta / sum_ref;
  printf("L1norm = %E\n", L1norm);
  if (L1norm > 1e-6) {
    printf("Original test failed\n");
    return EXIT_FAILURE;
  }

  // Verify delta
  L1norm = computeL1norm_Call(h_StockPrice, h_OptionStrike, h_OptionYears,
                              d_StockPrice, Delta);
  printf("L1norm of delta for Call option = %E\n", L1norm);
  if (L1norm > 1e-5) {
    printf("Gradient test failed: the difference between the computed and the "
           "approximated theoretical delta for Call option is larger than "
           "expected\n");
    return EXIT_FAILURE;
  }

  // Verify derivatives with respect to the Strike price
  L1norm = computeL1norm_Call(h_StockPrice, h_OptionStrike, h_OptionYears,
                              d_OptionStrike, dX);
  printf("L1norm of derivative of Call w.r.t. the strike price = %E\n", L1norm);
  if (L1norm > 1e-5) {
    printf(
        "Gradient test failed: the difference between the computed and the "
        "approximated theoretical derivative of Call w.r.t. the strike price "
        "is larger than expected\n");
    return EXIT_FAILURE;
  }

  // Verify theta
  L1norm = computeL1norm_Call(h_StockPrice, h_OptionStrike, h_OptionYears,
                              d_OptionYears, Theta);
  printf("L1norm of theta for Call option = %E\n", L1norm);
  if (L1norm > 1e-5) {
    printf("Gradient test failed: the difference between the computed and the "
           "approximated theoretical theta for Call option is larger than "
           "expected\n");
    return EXIT_FAILURE;
  }

  // Compute the derivatives of the price of the Put options
  for (int i = 0; i < OPT_N; i++) {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    d_CallResultGPU[i] = 1.0f;
    d_PutResultGPU[i] = 1.0f;
  }

  for (int i = 0; i < OPT_N; i++) {
    d_StockPrice[i] = 0.f;
    d_OptionStrike[i] = 0.f;
    d_OptionYears[i] = 0.f;
  }

  putGrad.execute(h_CallResultCPU, h_CallResultGPU, h_PutResultCPU,
                  h_PutResultGPU, h_StockPrice, h_OptionStrike, h_OptionYears,
                  d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears);

  // Verify delta
  L1norm = computeL1norm_Put(h_StockPrice, h_OptionStrike, h_OptionYears,
                             d_StockPrice, Delta);
  printf("L1norm of delta for Put option = %E\n", L1norm);
  if (L1norm > 1e-5) {
    printf("Gradient test failed: the difference between the computed and "
           "the approximated theoretical delta for Put option is larger than "
           "expected\n");
    return EXIT_FAILURE;
  }

  // Verify derivatives with respect to the Strike price
  L1norm = computeL1norm_Put(h_StockPrice, h_OptionStrike, h_OptionYears,
                             d_OptionStrike, dX);
  printf("L1norm of derivative of Put w.r.t. the strike price = %E\n", L1norm);
  if (L1norm > 1e-6) {
    printf("Gradient test failed: the difference between the computed and the "
           "approximated theoretcial derivative of "
           "Put w.r.t. the strike price is larger than expected\n");
    return EXIT_FAILURE;
  }

  // Verify theta
  L1norm = computeL1norm_Put(h_StockPrice, h_OptionStrike, h_OptionYears,
                             d_OptionYears, Theta);
  printf("L1norm of theta for Put option = %E\n", L1norm);
  if (L1norm > 1e-5) {
    printf("Gradient test failed: the difference between the computed and the "
           "approximated theoretical theta for Put option is larger than "
           "expected\n");
    return EXIT_FAILURE;
  }

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

  return EXIT_SUCCESS;
}
