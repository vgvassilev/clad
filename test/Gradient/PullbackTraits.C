// RUN: %cladclang -fsyntax-only %s -I%S/../../include 2>&1
// RUN: %cladclang %s -I%S/../../include -oPullbackTraits.out
// RUN: ./PullbackTraits.out | %filecheck_exec %s

#include "clad/Differentiator/FunctionTraits.h"
#include <cstdio>
#include <type_traits>

struct TargetClass {
  double method_const(double) const;
  double method_noex(double) const noexcept;
  double method_lref(double) &;
  double method_rref(double) &&;
};

// 1. Scalar return
using T_scalar = clad::PullbackDerivedFnTraits_t<double (*)(double, float)>;
using E_scalar = void (*)(double, float, double, double*, float*);
static_assert(std::is_same<T_scalar, E_scalar>::value,
              "Scalar return pullback trait mismatch");

// 2. Const reference return (requires output seed)
using T_const_ref =
    clad::PullbackDerivedFnTraits_t<const double& (*)(const double&, int)>;
using E_const_ref = void (*)(const double&, int, double, double*, int*);
static_assert(std::is_same<T_const_ref, E_const_ref>::value,
              "Const reference return pullback trait mismatch");

// 3. Non-const reference return (no output seed)
using T_non_const_ref =
    clad::PullbackDerivedFnTraits_t<double& (*)(double&, int)>;
using E_non_const_ref = void (*)(double&, int, double*, int*);
static_assert(std::is_same<T_non_const_ref, E_non_const_ref>::value,
              "Non-const reference return pullback trait mismatch");

// 4. Pointer return (no output seed)
using T_ptr = clad::PullbackDerivedFnTraits_t<double* (*)(double*, int)>;
using E_ptr = void (*)(double*, int, double*, int*);
static_assert(std::is_same<T_ptr, E_ptr>::value,
              "Pointer return pullback trait mismatch");

// 5. Void return (no output seed)
using T_void = clad::PullbackDerivedFnTraits_t<void (*)(double, double*)>;
using E_void = void (*)(double, double*, double*, double*);
static_assert(std::is_same<T_void, E_void>::value,
              "Void return pullback trait mismatch");

// 6. Const member function
using T_member_const =
    clad::PullbackDerivedFnTraits_t<decltype(&TargetClass::method_const)>;
using E_member_const =
    void (TargetClass::*)(double, double, TargetClass*, double*) const;
static_assert(std::is_same<T_member_const, E_member_const>::value,
              "Const member function pullback trait mismatch");

// 7. Noexcept member function
using T_member_noex =
    clad::PullbackDerivedFnTraits_t<decltype(&TargetClass::method_noex)>;
using E_member_noex =
    void (TargetClass::*)(double, double, TargetClass*, double*) const noexcept;
static_assert(std::is_same<T_member_noex, E_member_noex>::value,
              "Noexcept member function pullback trait mismatch");

// 8. Ref-qualified member functions (& and &&)
using T_member_lref =
    clad::PullbackDerivedFnTraits_t<decltype(&TargetClass::method_lref)>;
using E_member_lref = void (TargetClass::*)(double, double, TargetClass*, double*) &;
static_assert(std::is_same<T_member_lref, E_member_lref>::value,
              "Lvalue-ref-qualified member function pullback trait mismatch");

using T_member_rref =
    clad::PullbackDerivedFnTraits_t<decltype(&TargetClass::method_rref)>;
using E_member_rref = void (TargetClass::*)(double, double, TargetClass*, double*) &&;
static_assert(std::is_same<T_member_rref, E_member_rref>::value,
              "Rvalue-ref-qualified member function pullback trait mismatch");

#if __cpp_noexcept_function_type > 0
// 9. Free noexcept function
double free_noexcept(double) noexcept;
using T_free_noex = clad::PullbackDerivedFnTraits_t<decltype(&free_noexcept)>;
using E_free_noex = void (*)(double, double, double*) noexcept;
static_assert(std::is_same<T_free_noex, E_free_noex>::value,
              "Free noexcept function pullback trait mismatch");
#endif

// 10. Volatile parameter parity
using T_vol = clad::PullbackAdjointParamType_t<volatile double>;
using E_vol = volatile double*;
static_assert(std::is_same<T_vol, E_vol>::value,
              "Volatile parameter pullback adjoint trait mismatch");

using T_const_vol = clad::PullbackAdjointParamType_t<const volatile double>;
using E_const_vol = volatile double*;
static_assert(std::is_same<T_const_vol, E_const_vol>::value,
              "Const volatile parameter pullback adjoint trait mismatch");

// 11. Volatile and Const-Volatile member functions
struct VolatileTarget {
  double method(double) volatile;
  double methodCV(double) const volatile;
};

using T_member_vol =
    clad::PullbackDerivedFnTraits_t<decltype(&VolatileTarget::method)>;
using E_member_vol =
    void (VolatileTarget::*)(double, double, volatile VolatileTarget*, double*) volatile;
static_assert(std::is_same<T_member_vol, E_member_vol>::value,
              "Volatile member function pullback trait mismatch");

using T_member_const_vol =
    clad::PullbackDerivedFnTraits_t<decltype(&VolatileTarget::methodCV)>;
using E_member_const_vol =
    void (VolatileTarget::*)(double, double, volatile VolatileTarget*, double*) const volatile;
static_assert(std::is_same<T_member_const_vol, E_member_const_vol>::value,
              "Const volatile member function pullback trait mismatch");

// 12. Non-default calling conventions (e.g. ms_abi on x86_64) fail closed:
// PullbackDerivedFnTraits does not define ::type for non-default calling conventions.
template <typename...> using void_t_helper = void;
template <typename F, typename = void>
struct has_pullback_derived_fn_traits : std::false_type {};

template <typename F>
struct has_pullback_derived_fn_traits<
    F, void_t_helper<typename clad::PullbackDerivedFnTraits<F>::type>>
    : std::true_type {};

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
void __attribute__((ms_abi)) fn_ms_abi_test(double);
void fn_std_test(double);

static_assert(has_pullback_derived_fn_traits<decltype(&fn_std_test)>::value,
              "Standard calling convention must have PullbackDerivedFnTraits");
static_assert(!has_pullback_derived_fn_traits<decltype(&fn_ms_abi_test)>::value,
              "Non-default calling convention must fail-closed without PullbackDerivedFnTraits");
#endif

int main() {
  std::printf("Pullback traits verified successfully\n");
  // CHECK-EXEC: Pullback traits verified successfully
  return 0;
}
