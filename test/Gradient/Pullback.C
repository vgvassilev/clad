// RUN: %cladclang %s -I%S/../../include -oPullback.out 2>&1 | %filecheck %s
// RUN: ./Pullback.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oPullback.out
// RUN: ./Pullback.out | %filecheck_exec %s
// RUN: %cladclang -std=c++14 %s -I%S/../../include -oPullback14.out
// RUN: ./Pullback14.out | %filecheck_exec %s
// RUN: %cladclang -std=c++20 %s -I%S/../../include -oPullback20.out
// RUN: ./Pullback20.out | %filecheck_exec %s

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// 1. Scalar function with double parameters
double f_scalar(double x, double y) {
  return 3.0 * x * x + 4.0 * y;
}

// CHECK: void f_scalar_pullback(double x, double y, double _d_y0, double *_d_x, double *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 3. * _d_y0 * x;
// CHECK-NEXT:         *_d_x += 3. * x * _d_y0;
// CHECK-NEXT:         *_d_y += 4. * _d_y0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f_scalar_pullback_0(double x, double y, double _d_y0, double *_d_x) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 3. * _d_y0 * x;
// CHECK-NEXT:         *_d_x += 3. * x * _d_y0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// CHECK: void f_scalar_pullback_1(double x, double y, double _d_y0, double *_d_y) {
// CHECK-NEXT:     *_d_y += 4. * _d_y0;
// CHECK-NEXT: }

// 2. Float function
float f_float(float x, float y) {
  return 2.0f * x + 5.0f * y;
}

// CHECK: void f_float_pullback(float x, float y, float _d_y0, float *_d_x, float *_d_y) {
// CHECK-NEXT:     {
// CHECK-NEXT:         *_d_x += 2.F * _d_y0;
// CHECK-NEXT:         *_d_y += 5.F * _d_y0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// 3. Const reference return (requires output seed)
const double& f_const_ref(const double& x) {
  return x;
}

// CHECK: void f_const_ref_pullback(const double &x, double _d_y, double *_d_x) {
// CHECK-NEXT:     *_d_x += _d_y;
// CHECK-NEXT: }

// 4. Void return (NO seed)
void f_void(double x, double* out) {
  *out = 3.0 * x;
}

// CHECK: void f_void_pullback(double x, double *out, double *_d_x, double *_d_out) {
// CHECK-NEXT:     *out = 3. * x;
// CHECK-NEXT:     {
// CHECK-NEXT:         double _r_d0 = *_d_out;
// CHECK-NEXT:         *_d_out = 0.;
// CHECK-NEXT:         *_d_x += 3. * _r_d0;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// 5. Pointer return (NO seed)
double* f_ptr(double* p) {
  return p;
}

// CHECK: void f_ptr_pullback(double *p, double *_d_p) {
// CHECK-NEXT: }

// 5b. Mutating pointer return (NO seed)
double* f_ptr_scale(double* p) {
  *p = *p * 3.0;
  return p;
}

// CHECK: void f_ptr_scale_pullback(double *p, double *_d_p) {
// CHECK:     double _r_d0 = *_d_p;
// CHECK:     *_d_p = 0.;
// CHECK:     *_d_p += _r_d0 * 3.;
// CHECK: }

// 6. Non-const reference return (NO seed)
double& f_ref(double& r) {
  return r;
}

// CHECK: void f_ref_pullback(double &r, double *_d_r) {
// CHECK-NEXT: }

// 6b. Mutating non-const reference return (NO seed)
double& f_ref_scale(double& r) {
  r = r * 4.0;
  return r;
}

// CHECK: void f_ref_scale_pullback(double &r, double *_d_r) {
// CHECK:     double _r_d0 = *_d_r;
// CHECK:     *_d_r = 0.;
// CHECK:     *_d_r += _r_d0 * 4.;
// CHECK: }

// 7. Functor
struct SquareFunctor {
  double factor;
  double operator()(double x) const {
    return factor * x * x;
  }
};

// CHECK: void operator_call_pullback(double x, double _d_y, SquareFunctor *_d_this, double *_d_x) const {
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_this->factor += _d_y * x * x;
// CHECK-NEXT:         *_d_x += this->factor * _d_y * x;
// CHECK-NEXT:         *_d_x += this->factor * x * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// 8. Member function
struct Multiplier {
  double factor;
  double multiply(double x) const {
    return factor * x;
  }
  double compute(double x, double y) const {
    return factor * x * y;
  }
};

// CHECK: void multiply_pullback(double x, double _d_y, Multiplier *_d_this, double *_d_x) const {
// CHECK-NEXT:     {
// CHECK-NEXT:         _d_this->factor += _d_y * x;
// CHECK-NEXT:         *_d_x += this->factor * _d_y;
// CHECK-NEXT:     }
// CHECK-NEXT: }

// 9. Free noexcept function
double f_noexcept(double x) noexcept {
  return x * x * 3.0;
}

// CHECK: void f_noexcept_pullback(double x, double _d_y, double *_d_x) noexcept {
// CHECK:     *_d_x += _d_y * 3. * x;
// CHECK:     *_d_x += x * _d_y * 3.;
// CHECK: }

// 10. Mixed heterogeneous parameters
double f_mixed(double x, float y) {
  return x * y;
}

// 11. Multi-parameter function for non-contiguous and reordered partial selection
double f_3args(double x, double y, double z) {
  return 2.0 * x * y + 3.0 * z;
}

#if __cplusplus >= 201402L
// 12. constexpr function for automatically scheduled immediate evaluation
constexpr double f_constexpr(double x, double y) {
  return 3.0 * x * x + 4.0 * y;
}

#if !defined(__clang_major__) || __clang_major__ > 16
constexpr double eval_pb_imm_full() {
  auto pb = clad::pullback(f_constexpr);
  double dx = 0.0, dy = 0.0;
  pb.execute(2.0, 3.0, 1.0, &dx, &dy);
  return dx + dy;
}
constexpr double v_imm_full = eval_pb_imm_full();
static_assert(v_imm_full == 16.0, "constexpr full pullback static_assert mismatch");

constexpr double eval_pb_imm_partial_x() {
  auto pb_x = clad::pullback(f_constexpr, "x");
  double dx = 0.0;
  pb_x.execute(2.0, 3.0, 1.0, &dx);
  return dx;
}
constexpr double v_imm_px = eval_pb_imm_partial_x();
static_assert(v_imm_px == 12.0, "constexpr partial x pullback static_assert mismatch");

constexpr double eval_pb_imm_partial_y() {
  auto pb_y = clad::pullback(f_constexpr, "y");
  double dy = 0.0;
  pb_y.execute(2.0, 3.0, 1.0, nullptr, &dy);
  return dy;
}
constexpr double v_imm_py = eval_pb_imm_partial_y();
static_assert(v_imm_py == 4.0, "constexpr partial y pullback static_assert mismatch");

#if __cplusplus >= 202002L
consteval double eval_consteval_pb_full() {
  return eval_pb_imm_full();
}
static_assert(eval_consteval_pb_full() == 16.0, "consteval full pullback static_assert mismatch");
#endif
#endif
#endif

// 13. Volatile and Const-Volatile member functions
struct VolatileTarget {
  double factor;
  double method(double x) volatile {
    return factor * x;
  }
  double methodCV(double x) const volatile {
    return factor * x;
  }
  double compute(double x, double y) volatile {
    return factor * x * y;
  }
};

// 14. 3-arg heterogeneous function for later parameter selection
double f_mixed3(int a, double b, float c) {
  return a * b + c;
}

// 15. No-seed functions with partial selection of later parameter
void f_void_mut(double* a, double* b) {
  *a += 2.0;
  *b = *b * 3.0;
}

double* f_ptr_multi(double* a, double* b) {
  *a += 2.0;
  *b += 3.0;
  return b;
}

double& f_ref_multi(double& a, double& b) {
  a += 2.0;
  b += 3.0;
  return b;
}

// 16. Regular named binary functor
struct BinaryFunctor {
  double factor;
  double operator()(double x, double y) const {
    return factor * x * y;
  }
};

// 17. Recursive function
double f_recursive(double x, int n) {
  if (n <= 0)
    return 1.0;
  return x * f_recursive(x, n - 1);
}

// 18. Early-return function
double f_early(double x, double y) {
  if (x > 5.0)
    return x * 2.0;
  return x * y;
}

// 19. A custom pullback whose seed name collides with the primal's public
// wrapper name. The custom declaration itself is valid because its primal is
// named `primal`; the generated public wrapper takes the original primal name
// `x` and must deterministically rename the custom seed from `x` to `_d_y`.
double f_wrapper_name_collision(double x) { return x * x; }

namespace clad { namespace custom_derivatives {
void f_wrapper_name_collision_pullback(double primal, double x,
                                       double* d_primal) {
  *d_primal += 2.0 * primal * x;
}
} } // namespace clad::custom_derivatives

// CHECK: void f_wrapper_name_collision_pullback(double x, double _d_y, double *_d_x) {
// CHECK-NEXT:     clad::custom_derivatives::f_wrapper_name_collision_pullback(x, _d_y, _d_x);
// CHECK-NEXT: }

// 20. By-value, reference and struct-reference intermediate parameter reuse
double f_interm_val(double t, double x) {
  t = x * x;
  return t * 3.0;
}

struct ScratchBuf {
  double val;
};

double f_interm_ref(double& t, double x) {
  t = x * x;
  return t * 3.0;
}

double f_interm_struct(ScratchBuf& b, double x) {
  b.val = x * x;
  return b.val * 3.0;
}

// 21. Member function for nullable _d_this test
struct MemberOffset {
  double factor;
  double scale(double x, double y) const {
    return factor * x * y;
  }
};

int main() {
  // Test 1: Unit seed (1.0)
  auto pb1 = clad::pullback(f_scalar);
  auto fn_ptr = pb1.getFunctionPtr();
  static_assert(std::is_same<decltype(fn_ptr), void (*)(double, double, double, double*, double*)>::value,
                "pb1 getFunctionPtr type mismatch");
  double dx1 = 0.0, dy1 = 0.0;
  pb1.execute(2.0, 3.0, 1.0, &dx1, &dy1);
  printf("Unit seed: dx=%.2f, dy=%.2f\n", dx1, dy1);
  // CHECK-EXEC: Unit seed: dx=12.00, dy=4.00

  // Test 2: Scaled seed (3.5)
  double dx2 = 0.0, dy2 = 0.0;
  pb1.execute(2.0, 3.0, 3.5, &dx2, &dy2);
  printf("Scaled seed: dx=%.2f, dy=%.2f\n", dx2, dy2);
  // CHECK-EXEC: Scaled seed: dx=42.00, dy=14.00

  // Test 3: Zero seed (0.0)
  double dx3 = 0.0, dy3 = 0.0;
  pb1.execute(2.0, 3.0, 0.0, &dx3, &dy3);
  printf("Zero seed: dx=%.2f, dy=%.2f\n", dx3, dy3);
  // CHECK-EXEC: Zero seed: dx=0.00, dy=0.00

  // Test 4: Negative seed (-2.0)
  double dx4 = 0.0, dy4 = 0.0;
  pb1.execute(2.0, 3.0, -2.0, &dx4, &dy4);
  printf("Negative seed: dx=%.2f, dy=%.2f\n", dx4, dy4);
  // CHECK-EXEC: Negative seed: dx=-24.00, dy=-8.00

  // Test 5: Accumulation into nonzero initial adjoints
  double dx_acc = 100.0, dy_acc = 200.0;
  pb1.execute(2.0, 3.0, 1.0, &dx_acc, &dy_acc);
  printf("Nonzero initial: dx=%.2f, dy=%.2f\n", dx_acc, dy_acc);
  // CHECK-EXEC: Nonzero initial: dx=112.00, dy=204.00

  // Successive accumulation
  pb1.execute(2.0, 3.0, 1.0, &dx_acc, &dy_acc);
  printf("Successive: dx=%.2f, dy=%.2f\n", dx_acc, dy_acc);
  // CHECK-EXEC: Successive: dx=124.00, dy=208.00

  // Test 6: Partial argument selection
  auto pb_x = clad::pullback(f_scalar, "x");
  double dx_only = 0.0;
  pb_x.execute(2.0, 3.0, 1.0, &dx_only);
  printf("Arg x only: dx=%.2f\n", dx_only);
  // CHECK-EXEC: Arg x only: dx=12.00

  auto pb_y = clad::pullback(f_scalar, "y");
  double dy_only = 0.0;
  pb_y.execute(2.0, 3.0, 1.0, nullptr, &dy_only);
  printf("Arg y only: dy=%.2f\n", dy_only);
  // CHECK-EXEC: Arg y only: dy=4.00

  // Test 7: Float seed
  auto pb_flt = clad::pullback(f_float);
  float f_dx = 0.0f, f_dy = 0.0f;
  pb_flt.execute(1.0f, 1.0f, 2.0f, &f_dx, &f_dy);
  printf("Float pullback: dx=%.2f, dy=%.2f\n", f_dx, f_dy);
  // CHECK-EXEC: Float pullback: dx=4.00, dy=10.00

  // Test 8: const ref return (has seed)
  auto pb_cr = clad::pullback(f_const_ref);
  double in_val = 5.0, d_in = 0.0;
  pb_cr.execute(in_val, 3.0, &d_in);
  printf("Const ref: d_in=%.2f\n", d_in);
  // CHECK-EXEC: Const ref: d_in=3.00

  // Test 9: void return (no seed)
  auto pb_void = clad::pullback(f_void);
  double x_val = 4.0, out_val = 0.0, d_x_val = 0.0, d_out_val = 2.0;
  pb_void.execute(x_val, &out_val, &d_x_val, &d_out_val);
  printf("Void return: d_x=%.2f\n", d_x_val);
  // CHECK-EXEC: Void return: d_x=6.00

  // Test 10: pointer return (no seed)
  // In Clad's reverse-mode differentiation model (CladUtils.cpp:GetDerivativeType),
  // pointer and non-const reference return types represent aliases to caller/operand
  // memory. Reverse-mode cotangents flow through the referenced memory's adjoint buffer
  // (_d_p / _d_r), so no separate scalar cotangent return seed exists. Identity pullbacks
  // perform no arithmetic operations that modify the adjoint buffer, leaving pre-seeded
  // values intact.
  auto pb_ptr = clad::pullback(f_ptr);
  double p_val = 7.0, d_p_val = 5.0;
  pb_ptr.execute(&p_val, &d_p_val);
  printf("Pointer identity return: pre-seeded alias adjoint remains %.2f\n", d_p_val);
  // CHECK-EXEC: Pointer identity return: pre-seeded alias adjoint remains 5.00

  // Test 10b: Mutating pointer return (no seed, cotangent flows through pre-seeded memory)
  auto pb_ptr_scale = clad::pullback(f_ptr_scale);
  double p_scale_val = 7.0, d_p_scale = 2.0;
  pb_ptr_scale.execute(&p_scale_val, &d_p_scale);
  printf("Pointer mutating return: d_p=%.2f\n", d_p_scale);
  // CHECK-EXEC: Pointer mutating return: d_p=6.00

  // Test 11: non-const ref return (no seed)
  auto pb_ref = clad::pullback(f_ref);
  double r_val = 8.0, d_r_val = 6.0;
  pb_ref.execute(r_val, &d_r_val);
  printf("Reference identity return: pre-seeded alias adjoint remains %.2f\n", d_r_val);
  // CHECK-EXEC: Reference identity return: pre-seeded alias adjoint remains 6.00

  // Test 11b: Mutating non-const reference return (no seed, cotangent flows through pre-seeded memory)
  auto pb_ref_scale = clad::pullback(f_ref_scale);
  double r_scale_val = 8.0, d_r_scale = 2.0;
  pb_ref_scale.execute(r_scale_val, &d_r_scale);
  printf("Reference mutating return: d_r=%.2f\n", d_r_scale);
  // CHECK-EXEC: Reference mutating return: d_r=8.00

  // Test 12: Functor
  SquareFunctor sq{4.0};
  SquareFunctor d_sq{0.0};
  auto pb_fn = clad::pullback(sq);
  double d_x_fn = 0.0;
  pb_fn.execute(3.0, 1.0, &d_sq, &d_x_fn);
  printf("Functor: d_x=%.2f, d_sq.factor=%.2f\n", d_x_fn, d_sq.factor);
  // CHECK-EXEC: Functor: d_x=24.00, d_sq.factor=9.00

  // Test 13: Member function
  Multiplier m{5.0};
  auto pb_mem = clad::pullback(&Multiplier::multiply);
  Multiplier d_this{0.0};
  double d_x_mem = 0.0;
  pb_mem.execute(m, 2.0, 1.0, &d_this, &d_x_mem);
  printf("Member fn: d_x=%.2f, d_this.factor=%.2f\n", d_x_mem, d_this.factor);
  // CHECK-EXEC: Member fn: d_x=5.00, d_this.factor=2.00

  // Test 13b: Free noexcept function
  auto pb_noex = clad::pullback(f_noexcept);
  double dx_noex = 0.0;
  pb_noex.execute(2.0, 1.0, &dx_noex);
  printf("Free noexcept: dx=%.2f\n", dx_noex);
  // CHECK-EXEC: Free noexcept: dx=12.00

#if __cplusplus >= 201402L
  // Test 14: The same constexpr source used at run time
  auto pb_imm = clad::pullback(f_constexpr);
  double dx_imm = 0.0, dy_imm = 0.0;
  pb_imm.execute(2.0, 3.0, 1.0, &dx_imm, &dy_imm);
  printf("Immediate mode: dx=%.2f, dy=%.2f\n", dx_imm, dy_imm);
  // CHECK-EXEC: Immediate mode: dx=12.00, dy=4.00
#endif


  // Test 15: Canonical deduplication between full-parameter syntax
  // Both clad::pullback(f_scalar) and clad::pullback(f_scalar, "x, y")
  // evaluate DVI.size() == Function->getNumParams() and return f_scalar_pullback,
  // correctly sharing a single generated declaration.
  auto pb_all_named = clad::pullback(f_scalar, "x, y");
  double dx_dedup = 0.0, dy_dedup = 0.0;
  pb_all_named.execute(2.0, 3.0, 1.0, &dx_dedup, &dy_dedup);
  printf("Deduplicated full pullback: dx=%.2f, dy=%.2f\n", dx_dedup, dy_dedup);
  // CHECK-EXEC: Deduplicated full pullback: dx=12.00, dy=4.00

  // Test 16: Heterogeneous parameters partial selection
  auto pb_mixed_y = clad::pullback(f_mixed, "y");
  float dy_mixed = 0.0f;
  pb_mixed_y.execute(2.0, 3.0f, 1.0, nullptr, &dy_mixed);
  printf("Mixed partial y: dy=%.2f\n", dy_mixed);
  // CHECK-EXEC: Mixed partial y: dy=2.00

  // Test 17: Multi-parameter non-contiguous selection ("x, z")
  auto pb_xz = clad::pullback(f_3args, "x, z");
  double dx_3 = 0.0, dz_3 = 0.0;
  pb_xz.execute(2.0, 3.0, 4.0, 1.0, &dx_3, nullptr, &dz_3);
  printf("Non-contiguous xz: dx=%.2f, dz=%.2f\n", dx_3, dz_3);
  // CHECK-EXEC: Non-contiguous xz: dx=6.00, dz=3.00

  // Test 18: Multi-parameter reordered selection ("z, x")
  auto pb_zx = clad::pullback(f_3args, "z, x");
  double dx_zx = 0.0, dz_zx = 0.0;
  pb_zx.execute(2.0, 3.0, 4.0, 1.0, &dx_zx, nullptr, &dz_zx);
  printf("Reordered zx: dx=%.2f, dz=%.2f\n", dx_zx, dz_zx);
  // CHECK-EXEC: Reordered zx: dx=6.00, dz=3.00

  // Test 19: Member function multi-param partial selection
  Multiplier m2{4.0};
  Multiplier d_m2{0.0};
  double dx_comp = 0.0;
  auto pb_mem_comp = clad::pullback(&Multiplier::compute, "x");
  pb_mem_comp.execute(m2, 2.0, 3.0, 1.0, &d_m2, &dx_comp, nullptr);
  printf("Member partial x: dx=%.2f, d_this.factor=%.2f\n", dx_comp, d_m2.factor);
  // CHECK-EXEC: Member partial x: dx=12.00, d_this.factor=6.00

  // Test 20: Volatile member full pullback
  VolatileTarget vt{3.0};
  volatile VolatileTarget d_vt{0.0};
  double dx_v = 0.0;
  auto pb_vol = clad::pullback(&VolatileTarget::method);
  auto vol_fn_ptr = pb_vol.getFunctionPtr();
  static_assert(std::is_same<decltype(vol_fn_ptr), void (VolatileTarget::*)(double, double, volatile VolatileTarget*, double*) volatile>::value,
                "pb_vol getFunctionPtr type mismatch");
  pb_vol.execute(vt, 4.0, 1.0, &d_vt, &dx_v);
  printf("Volatile member: dx=%.2f, d_this.factor=%.2f\n", dx_v, d_vt.factor);
  // CHECK-EXEC: Volatile member: dx=3.00, d_this.factor=4.00

  // Test 21: Const-volatile member full pullback
  const volatile VolatileTarget cvt{3.0};
  volatile VolatileTarget d_cvt{0.0};
  double dx_cv = 0.0;
  auto pb_cvol = clad::pullback(&VolatileTarget::methodCV);
  auto cvol_fn_ptr = pb_cvol.getFunctionPtr();
  static_assert(std::is_same<decltype(cvol_fn_ptr), void (VolatileTarget::*)(double, double, volatile VolatileTarget*, double*) const volatile>::value,
                "pb_cvol getFunctionPtr type mismatch");
  pb_cvol.execute(cvt, 4.0, 1.0, &d_cvt, &dx_cv);
  printf("Const volatile member: dx=%.2f, d_this.factor=%.2f\n", dx_cv, d_cvt.factor);
  // CHECK-EXEC: Const volatile member: dx=3.00, d_this.factor=4.00

  // Test 22: Const member later parameter partial selection
  Multiplier m_lat{4.0};
  Multiplier d_m_lat{0.0};
  double dy_lat = 0.0;
  auto pb_mem_lat = clad::pullback(&Multiplier::compute, "y");
  pb_mem_lat.execute(m_lat, 2.0, 3.0, 1.0, &d_m_lat, nullptr, &dy_lat);
  printf("Member partial y: dy=%.2f, d_this.factor=%.2f\n", dy_lat, d_m_lat.factor);
  // CHECK-EXEC: Member partial y: dy=8.00, d_this.factor=6.00

  // Test 23: Volatile member later parameter partial selection
  VolatileTarget vt_lat{5.0};
  volatile VolatileTarget d_vt_lat{0.0};
  double dy_v_lat = 0.0;
  auto pb_vol_lat = clad::pullback(&VolatileTarget::compute, "y");
  pb_vol_lat.execute(vt_lat, 2.0, 3.0, 1.0, &d_vt_lat, nullptr, &dy_v_lat);
  printf("Volatile member partial y: dy=%.2f, d_this.factor=%.2f\n", dy_v_lat, d_vt_lat.factor);
  // CHECK-EXEC: Volatile member partial y: dy=10.00, d_this.factor=6.00

  // Test 24: Heterogeneous 3-param last-parameter selection
  auto pb_m3 = clad::pullback(f_mixed3, "c");
  float dc_m3 = 0.0f;
  pb_m3.execute(3, 4.0, 5.0f, 1.0, nullptr, nullptr, &dc_m3);
  printf("Mixed 3 last param: dc=%.2f\n", dc_m3);
  // CHECK-EXEC: Mixed 3 last param: dc=1.00

  // Test 25: No-seed void multi-param partial selection of later parameter
  auto pb_void_b = clad::pullback(f_void_mut, "b");
  double va = 1.0, vb = 2.0, d_vb = 2.0;
  pb_void_b.execute(&va, &vb, nullptr, &d_vb);
  printf("Void partial b: db=%.2f\n", d_vb);
  // CHECK-EXEC: Void partial b: db=6.00

  // Test 26: No-seed pointer multi-param partial selection of later parameter
  auto pb_ptr_b = clad::pullback(f_ptr_multi, "b");
  double pa = 1.0, pb_val = 2.0, d_pb_val = 5.0;
  pb_ptr_b.execute(&pa, &pb_val, nullptr, &d_pb_val);
  printf("Pointer partial b: d_b=%.2f\n", d_pb_val);
  // CHECK-EXEC: Pointer partial b: d_b=5.00

  // Test 27: No-seed reference multi-param partial selection of later parameter
  auto pb_ref_b = clad::pullback(f_ref_multi, "b");
  double ra = 1.0, rb = 2.0, d_rb = 6.0;
  pb_ref_b.execute(ra, rb, nullptr, &d_rb);
  printf("Reference partial b: d_b=%.2f\n", d_rb);
  // CHECK-EXEC: Reference partial b: d_b=6.00

  // Test 28: Named binary functor partial selection of later parameter
  BinaryFunctor bf{3.0};
  BinaryFunctor d_bf{0.0};
  double dy_bf = 0.0;
  auto pb_bf_y = clad::pullback(bf, "y");
  pb_bf_y.execute(2.0, 4.0, 1.0, &d_bf, nullptr, &dy_bf);
  printf("Binary functor partial y: dy=%.2f, d_this.factor=%.2f\n", dy_bf, d_bf.factor);
  // CHECK-EXEC: Binary functor partial y: dy=6.00, d_this.factor=8.00

  // Test 29: Recursive function pullback
  auto pb_rec = clad::pullback(f_recursive);
  double dx_rec = 0.0;
  int dn_rec = 0;
  pb_rec.execute(3.0, 3, 1.0, &dx_rec, &dn_rec);
  printf("Recursive pullback: dx=%.2f\n", dx_rec);
  // CHECK-EXEC: Recursive pullback: dx=27.00

  // Test 30: Early return function pullback
  auto pb_early = clad::pullback(f_early);
  double dx_e1 = 0.0, dy_e1 = 0.0;
  pb_early.execute(6.0, 3.0, 1.0, &dx_e1, &dy_e1);
  printf("Early return true branch: dx=%.2f, dy=%.2f\n", dx_e1, dy_e1);
  // CHECK-EXEC: Early return true branch: dx=2.00, dy=0.00
  double dx_e2 = 0.0, dy_e2 = 0.0;
  pb_early.execute(4.0, 3.0, 1.0, &dx_e2, &dy_e2);
  printf("Early return false branch: dx=%.2f, dy=%.2f\n", dx_e2, dy_e2);
  // CHECK-EXEC: Early return false branch: dx=3.00, dy=4.00

  auto pb_collision = clad::pullback(f_wrapper_name_collision);
  double dx_collision = 0.0;
  pb_collision.execute(3.0, 2.0, &dx_collision);
  printf("Wrapper name collision: dx=%.2f\n", dx_collision);
  // CHECK-EXEC: Wrapper name collision: dx=12.00

  // Test 31: By-value intermediate parameter reuse in partial pullback with non-unit seed & accumulation
  auto pb_ival = clad::pullback(f_interm_val, "x");
  double dx_ival = 10.0;
  pb_ival.execute(0.0, 2.0, 2.5, nullptr, &dx_ival);
  auto grad_ival = clad::gradient(f_interm_val, "x");
  double g_dx_ival = 0.0;
  grad_ival.execute(0.0, 2.0, &g_dx_ival);
  printf("By-value intermediate: dx=%.2f, grad=%.2f, agrees=%d\n",
         dx_ival, g_dx_ival, (dx_ival == 10.0 + 2.5 * g_dx_ival));
  // CHECK-EXEC: By-value intermediate: dx=40.00, grad=12.00, agrees=1

  // Test 32: Reference and struct-reference intermediate parameter reuse in partial pullback
  auto pb_iref = clad::pullback(f_interm_ref, "x");
  double dummy_t = 0.0, dx_iref = 5.0;
  pb_iref.execute(dummy_t, 2.0, 3.0, nullptr, &dx_iref);
  auto pb_istruct = clad::pullback(f_interm_struct, "x");
  ScratchBuf sbuf{0.0};
  double dx_istruct = 7.0;
  pb_istruct.execute(sbuf, 2.0, 3.0, nullptr, &dx_istruct);
  printf("Ref intermediate: dx=%.2f, Struct intermediate: dx=%.2f\n",
         dx_iref, dx_istruct);
  // CHECK-EXEC: Ref intermediate: dx=41.00, Struct intermediate: dx=43.00

  // Test 33: Member function partial pullback with nullable d_this, non-unit seed and accumulation
  MemberOffset mo{3.0};
  auto pb_mo_y = clad::pullback(&MemberOffset::scale, "y");
  double dy_mo = 5.0;
  pb_mo_y.execute(mo, 4.0, 2.0, 2.0, nullptr, nullptr, &dy_mo);
  printf("Member nullable d_this: dy=%.2f\n", dy_mo);
  // CHECK-EXEC: Member nullable d_this: dy=29.00

  return 0;
}
