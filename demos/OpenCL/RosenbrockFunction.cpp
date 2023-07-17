//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to calculate the Rosenbrock function.
//
// author:  Martin Vasilev <mrtn.vassilev-at-gmail.com>
//----------------------------------------------------------------------------//

// To run the demo please type:
// path/to/clang++  -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so  -I../include/ -framework opencl -std=c++11 RosenbrockFunction.cpp
//
// A typical invocation would be:
// ../../../../../obj/Debug+Asserts/bin/clang++  -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../../obj/Debug+Asserts/lib/libclad.dylib     \
// -I../../include/ -framework opencl -std=c++11 RosenbrockFunction.cpp

// Necessary for clad to work include
//#include "clad/Differentiator/Differentiator.h"

#include "stdafx.h"

#include <stdlib.h>   // For _MAX_PATH definition
#include <stdio.h>
#include <string.h>

#include "experimental_offload.h"

#define MAX_DATA_SIZE 1024*1024*48

#define OFFLOAD

// Rosenbrock function declaration
float rosenbrock_func(float x, float y) {
  return (x - 1.0f) * (x - 1.0f) + 100.0f * (y - x * x) * (y - x * x);
}

float rosenbrockX_execute(float x, float y) {
  return 2.F * (x - 1.F) - 400.F * x * (y - x * x);
}
static float rosenbrockY_execute(float x, float y) {
  return 200.F * (y - x * x);
}


float rosenbrock(float x[], int size) {
  //auto rosenbrockX = clad::differentiate(rosenbrock_func, 0);
  //auto rosenbrockY = clad::differentiate(rosenbrock_func, 1);
  float sum = 0;
  for (int i = 0; i < size - 1; i++) {
    //float one = rosenbrockX.execute(x[i], x[i + 1]);
    //float two = rosenbrockY.execute(x[i], x[i + 1]);
    float one = rosenbrockX_execute(x[i], x[i + 1]);
    float two = rosenbrockY_execute(x[i], x[i + 1]);
    sum += one + two;
    //printf("%f,", one + two);
  }
  return sum;
}

/*
float rosenbrock_offloaded(float x[], int size) {
return clad::experimantal_offload([=] {
auto rosenbrockX = clad::differentiate(rosenbrock_func, 0);
auto rosenbrockY = clad::differentiate(rosenbrock_func, 1);
float sum = 0;
for (int i = 0; i < size-1; i++) {
float one = rosenbrockX.execute(x[i], x[i + 1]);
float two = rosenbrockY.execute(x[i], x[i + 1]);
sum += one + two;
}
return sum;
});
}
*/

float rosenbrock_offloaded(float x[], int size) {
  return experimental_offloaded(x, size);
}

int main(int argc, char* argv[]) {
#ifdef OFFLOAD
  init_experimental_offload();
#endif

  float *Xarray = (float *)malloc(MAX_DATA_SIZE*sizeof(float));

  // Generate test data
  for (int j = 0; j<MAX_DATA_SIZE; j++) {
    Xarray[j] = (float)rand() / RAND_MAX;
  }

  float a = 0;
  for (int i = 0; i<100; i++) {
#ifndef OFFLOAD
    float result = rosenbrock(Xarray, MAX_DATA_SIZE);
#else
    float result = rosenbrock_offloaded(Xarray, MAX_DATA_SIZE);
#endif
    a += result;
  }
  printf("The result is %f\n", a);

#ifdef OFFLOAD
  done_experimental_offload();
#endif

  return 0;
}
