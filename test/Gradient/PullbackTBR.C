// RUN: %cladclang %s -I%S/../../include -oPullbackTBR.out 2>&1 | %filecheck %s
// RUN: ./PullbackTBR.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oPullbackTBR_notbr.out
// RUN: ./PullbackTBR_notbr.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// Helper that modifies memory parameter in place; triggers TBR analysis on call traversal.
void scale_in_place(double* p) {
  *p = *p * 2.0;
}

// CHECK: inline void scale_in_place_pullback(double *p, double *_d_p) {
// CHECK-NEXT:     *p = *p * 2.;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_p;
// CHECK-NEXT:         *_d_p = 0.;
// CHECK-NEXT:         *_d_p += _r_d0 * 2.;
// CHECK-NEXT:     }
// CHECK-NEXT: }

double top_level_tbr(double x) {
  double val = x;
  scale_in_place(&val);
  return val;
}

// CHECK: void top_level_tbr_pullback(double x, double _d_y, double *_d_x) {
// CHECK-DAG: clad::record_range
// CHECK-DAG: scale_in_place_pullback
// CHECK:     *_d_x += _d_val;
// CHECK: }

// Top-level function with non-const memory parameter directly differentiated with clad::pullback.
// This exercises the null-parent guard in DiffPlanner.cpp:1914:
// For this top-level request:
//   request.EnableTBRAnalysis == true
//   request.Mode == DiffMode::pullback
//   utils::hasMemoryTypeParams(request.Function) == true
// while Saved.get() is null because this is the root request without a parent.
// Without the null-parent guard `if (Saved.get())`,
// Saved.get()->addFunctionModifiedParams(FD, ...) would dereference a null pointer and crash.
void top_level_mem_tbr(double* arr, double factor) {
  scale_in_place(arr);
  *arr = *arr * factor;
}

// CHECK: void top_level_mem_tbr_pullback(double *arr, double factor, double *_d_arr, double *_d_factor) {
// CHECK-DAG: clad::record_range
// CHECK-DAG: scale_in_place(arr);
// CHECK-DAG: double [[T:_t[0-9]+]] = *arr;
// CHECK-DAG: *arr = [[T]];
// CHECK-DAG: clad::peek_range
// CHECK-DAG: scale_in_place_pullback(arr, _d_arr);
// CHECK-DAG: clad::drop_range
// CHECK: }

int main() {
  auto pb = clad::pullback(top_level_tbr);
  double dx = 0.0;
  pb.execute(3.0, 1.0, &dx);
  std::printf("TBR pullback: dx=%.2f\n", dx);
  // CHECK-EXEC: TBR pullback: dx=2.00

  auto pb_mem = clad::pullback(top_level_mem_tbr);
  double arr_val = 3.0;
  double d_arr = 1.0, d_factor = 0.0;
  pb_mem.execute(&arr_val, 4.0, &d_arr, &d_factor);
  std::printf("Top-level memory TBR pullback: d_arr=%.2f, d_factor=%.2f\n", d_arr, d_factor);
  // CHECK-EXEC: Top-level memory TBR pullback: d_arr=8.00, d_factor=6.00
  return 0;
}
