// RUN: %cladclang %s -I%S/../../include -oTBRCalls.out 2>&1 | %filecheck %s
// RUN: ./TBRCalls.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s \
// RUN:   -I%S/../../include -oTBRCalls.out
// RUN: ./TBRCalls.out | %filecheck_exec %s

// Verifies that TBR analysis records caller state mutated through a call even
// when the argument does not resolve to a storage location -- the GradBench
// GMM shape `f(..., &v[0])`, where the address-of wraps a user-defined
// operator[] call. The dependency fallback used to come back empty for such
// arguments, so no pre-call state was stored and the pullback replayed from
// post-call values.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

void fillfrom(int d, const double* x, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = x[i];
}

// Reads and overwrites out in place; its pullback needs the primal values.
void squarer(int d, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = out[i] * out[i];
}

struct Vec2 {
  double vals[2];
  double& operator[](int i) { return vals[i]; }
  const double& operator[](int i) const { return vals[i]; }
  double& at(int i) { return vals[i]; }
  const double& at(int i) const { return vals[i]; }
  double* data() { return vals; }
  const double* data() const { return vals; }
};

struct W {
  double* ptr;
};

// The mutated buffer is reached through &buf[0] over a user-defined
// operator[].
double t1(const double* x) {
  Vec2 buf;
  fillfrom(2, x, &buf[0]);
  squarer(2, &buf[0]);
  return buf[0] + buf[1];
}

// The mutated buffer is reached through a data() accessor.
double t2(const double* x) {
  Vec2 buf;
  fillfrom(2, x, buf.data());
  squarer(2, buf.data());
  return buf[0] + buf[1];
}

// The mutated buffer is reached through &buf[0] on a raw array.
double t3(const double* x) {
  double buf[2];
  fillfrom(2, x, &buf[0]);
  squarer(2, &buf[0]);
  return buf[0] + buf[1];
}

// The mutated buffer is reached through a struct-carried pointer.
double t4(const double* x) {
  double buf[2];
  W w{buf};
  fillfrom(2, x, buf);
  squarer(2, w.ptr);
  return buf[0] + buf[1];
}

// A call-shaped assignment lvalue overwritten after a nonlinear read: the
// element must be stored and restored so the read's adjoint sees its primal.
double t5(const double* x) {
  Vec2 buf;
  buf.at(0) = x[0];
  buf.at(1) = x[1];
  double t = buf.at(0) * buf.at(0) + buf.at(1) * buf.at(1);
  buf.at(0) = 3 * x[1];
  return t + buf.at(0);
}

double t6(const double* x) {
  Vec2 buf;
  buf[0] = x[0];
  buf[1] = x[1];
  double t = buf[0] * buf[0] + buf[1] * buf[1];
  buf[0] = 3 * x[1];
  return t + buf[0];
}

// An in-loop overwrite through operator[] results: each reverse iteration
// must see its own element values, restored by re-evaluating the lvalue.
double t8(const double* x) {
  Vec2 v;
  double total = 0;
  for (int k = 1; k <= 2; k++) {
    for (int j = 0; j < 2; j++)
      v[j] = k * x[j];
    total += v[0] * v[0] + v[1] * v[1];
  }
  return total;
}

double sqnorm(int d, const double* v) {
  double t = 0;
  for (int i = 0; i < d; i++)
    t += v[i] * v[i];
  return t;
}

void diffem(int d, const double* x, const double* y, double* out) {
  for (int i = 0; i < d; i++)
    out[i] = x[i] - y[i];
}

// The GradBench GMM loop: buf is overwritten through &buf[0] on every
// iteration, and each iteration's values feed that iteration's nonlinear
// read, so every iteration needs its own restore in the reverse sweep.
double t9(const double* x, const double* m) {
  Vec2 buf;
  double total = 0;
  for (int k = 0; k < 2; k++) {
    diffem(2, x, m + 2 * k, &buf[0]);
    total += sqnorm(2, &buf[0]);
  }
  return total;
}

// A value live across two mutating calls: squarer overwrites what the first
// sqnorm's pullback still needs after squarer's own pullback replays.
double t10(const double* x) {
  double buf[2];
  fillfrom(2, x, buf);
  double t1 = sqnorm(2, buf);
  squarer(2, buf);
  double t2 = sqnorm(2, buf);
  return t1 + t2;
}

// Nothing is read nonlinearly, so no pre-call state may be stored.
double t7(const double* x) {
  double buf[2];
  fillfrom(2, x, &buf[0]);
  return buf[0] + buf[1];
}

// CHECK: void t1_grad(const double *x, double *_d_x) {
// CHECK: squarer_reverse_forw(2, &{{.*}}, 0, &{{.*}}, _tracker0);
// CHECK: _tracker0.restore();
// CHECK: squarer_pullback(2, &{{.*}});

// CHECK: void t7_grad(const double *x, double *_d_x) {
// CHECK-NOT: _tracker
// CHECK-NOT: clad::push
// CHECK: fillfrom_pullback(2, x, &buf[0]

// In a loop each iteration gets its own tracker snapshot through a tape.
// CHECK: void t9_grad(const double *x, const double *m, double *_d_x, double *_d_m) {
// CHECK: clad::tape<clad::restore_tracker> _tracker0 = {};
// CHECK: clad::push(_tracker0, clad::restore_tracker());
// CHECK: clad::back(_tracker0).restore();
// CHECK: diffem_pullback(2, x, m + 2 * k, &
// CHECK-NEXT: clad::back(_tracker0).restore();
// CHECK-NEXT: clad::pop(_tracker0);

int main() {
  double x[2] = {1, 2};
  double dx[2];
#define TEST(fn)                                                               \
  {                                                                            \
    dx[0] = dx[1] = 0;                                                         \
    auto grad = clad::gradient(fn, "x");                                       \
    grad.execute(x, dx);                                                       \
    printf(#fn ": dx={%.2f, %.2f}\n", dx[0], dx[1]);                           \
  }
  // f = x0^2 + x1^2 => df/dx_i = 2 x_i
  TEST(t1); // CHECK-EXEC: t1: dx={2.00, 4.00}
  TEST(t2); // CHECK-EXEC: t2: dx={2.00, 4.00}
  TEST(t3); // CHECK-EXEC: t3: dx={2.00, 4.00}
  TEST(t4); // CHECK-EXEC: t4: dx={2.00, 4.00}
  // f = x0^2 + x1^2 + 3 x1 => df/dx0 = 2 x0, df/dx1 = 2 x1 + 3
  TEST(t5); // CHECK-EXEC: t5: dx={2.00, 7.00}
  TEST(t6); // CHECK-EXEC: t6: dx={2.00, 7.00}
  // f = sum_k k^2 |x|^2 = 5 |x|^2 => df/dx_i = 10 x_i
  TEST(t8); // CHECK-EXEC: t8: dx={10.00, 20.00}
  // f = x0 + x1
  TEST(t7); // CHECK-EXEC: t7: dx={1.00, 1.00}
  {
    double m[4] = {0.5, 1, 3, -1};
    double dm[4] = {0, 0, 0, 0};
    dx[0] = dx[1] = 0;
    auto grad = clad::gradient(t9, "x, m");
    grad.execute(x, m, dx, dm);
    // f = sum_k sum_i (x_i - m_ki)^2
    printf("t9: dx={%.2f, %.2f} dm={%.2f, %.2f, %.2f, %.2f}\n", dx[0], dx[1],
           dm[0], dm[1], dm[2], dm[3]);
    // CHECK-EXEC: t9: dx={-3.00, 8.00} dm={-1.00, -2.00, 4.00, -6.00}
  }
  // f = |x|^2 + |x^2|^2 => df/dx_i = 2 x_i + 4 x_i^3
  TEST(t10); // CHECK-EXEC: t10: dx={6.00, 36.00}
}
