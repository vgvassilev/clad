// RUN: %cladclang %s -I%S/../../include -oRecordRangeExtent.out 2>&1 | %filecheck %s
// RUN: ./RecordRangeExtent.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oRecordRangeExtent.out
// RUN: ./RecordRangeExtent.out | %filecheck_exec %s

// A recorded range's extent appears at four places -- the record, the two
// replays around the pullback, and the drop. Spelling the call's argument out
// at each of them would evaluate it four times, and an argument whose value
// changed in between would leave the record and its replays disagreeing about
// how many elements there are. It is read once instead.

#include "clad/Differentiator/Differentiator.h"

#include <cstdio>

int g_calls = 0;
int howMany() {
  ++g_calls;
  return 2;
}

// Nonlinear in its own output, so its pullback needs the pre-call values and
// the caller has to capture them.
void squarer(int n, double* out) {
  for (int i = 0; i < n; i++)
    out[i] = out[i] * out[i];
}

double f(const double* x) {
  double b[4] = {x[0], x[1], 0, 0};
  squarer(howMany(), b);
  return b[0] + b[1];
}

// CHECK: void f_grad(const double *x, double *_d_x) {
// CHECK: {{(unsigned long|std::size_t)}} _recn0 = howMany();
// CHECK: clad::record_range(_rec0, b, _recn0);
// CHECK: clad::peek_range(_rec0, b, _recn0);
// CHECK: clad::peek_range(_rec0, b, _recn0);
// CHECK-NEXT: clad::drop_range(_rec0, _recn0);

int main() {
  auto g = clad::gradient(f, "x");
  double x[4] = {1, 2, 3, 4};
  double dx[4] = {0, 0, 0, 0};
  g_calls = 0;
  g.execute(x, dx);
  // The extent is read once here. clad separately repeats the call's own
  // arguments in the pullback call, which is why this is not 1.
  printf("extent reads: %d\n", g_calls);
  // CHECK-EXEC: extent reads: 3
  printf("dx={%.2f, %.2f}\n", dx[0], dx[1]);
  // CHECK-EXEC: dx={2.00, 4.00}
  return 0;
}
