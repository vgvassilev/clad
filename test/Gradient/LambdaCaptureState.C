// RUN: %cladclang %s -I%S/../../include -o %t
// RUN: %t | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -disable-tbr -Xclang -plugin-arg-clad -Xclang -disable-va -o %t.no-analysis
// RUN: %t.no-analysis | %filecheck_exec %s
// RUN: %cladclang %s -I%S/../../include -Xclang -plugin-arg-clad -Xclang -enable-va -o %t.va
// RUN: %t.va | %filecheck_exec %s
// UNSUPPORTED: clang-11, clang-12, clang-13, clang-14, clang-15, clang-16

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

double read_before_mutation(double x) {
  auto inner = [&x] { return x * x; };
  double first = inner();
  x *= 3;
  return first + inner();
}

double read_reference_initializer(double x) {
  auto inner = [&value = x] { return value * value; };
  double first = inner();
  x *= 3;
  return first + inner();
}

double closure_alias(double x) {
  auto inner = [x] { return x * x; };
  auto& alias = inner;
  const auto& another = alias;
  return another();
}

double mutable_alias(double x) {
  auto inner = [x]() mutable { x *= 2; return x * x; };
  auto& alias = inner;
  double first = alias();
  return first + inner();
}

double explicit_call_operator(double x) {
  auto outer = [x] {
    auto inner = [x] { return x * x; };
    return inner.operator()();
  };
  return outer.operator()();
}

struct CopyValue {
  double value;
  CopyValue() : value(0) {}
  CopyValue(double x) : value(x) {}
  CopyValue(const CopyValue& other) : value(2 * other.value) {}
};

double mutable_record(double x) {
  auto inner = [v = CopyValue(x)]() mutable {
    v.value *= 2;
    return v.value * v.value;
  };
  double first = inner();
  return first + inner();
}

double pointer_capture(double x) {
  auto inner = [p = &x] { *p *= 2; return *p * *p; };
  double first = inner();
  return first + inner();
}

double reference_return(double x) {
  auto inner = [&x]() -> double& { return x; };
  return inner() * inner();
}

double reference_return_assignment(double x) {
  auto inner = [&x]() -> double& { return x; };
  inner() *= 2;
  return x * x;
}

double conditional_mutation(double x) {
  auto inner = [&x] {
    if (x > 0)
      x *= 2;
    else
      x *= 3;
    return x * x;
  };
  double first = inner();
  return first + inner();
}

double local_storage(double x) {
  auto inner = [&x] {
    double local = x;
    local *= 2;
    x *= 2;
    return local * x;
  };
  double first = inner();
  return first + inner();
}

double nested_mutation(double x) {
  auto outer = [&x] {
    auto inner = [&x] { x *= 2; return x * x; };
    double first = inner();
    return first + inner();
  };
  double first = outer();
  return first + outer();
}

double reference_to_closure(double x) {
  auto inner = [&x] { x *= 2; return x * x; };
  auto outer = [&inner] { return inner(); };
  double first = outer();
  return first + outer();
}

double nested_local_storage(double x) {
  auto outer = [&x] {
    double y = x;
    auto inner = [&y] { y *= 2; return y * y; };
    x *= 2;
    return inner();
  };
  double first = outer();
  return first + outer();
}

double nested_snapshot(double x) {
  auto outer = [&x] {
    auto inner = [y = x]() mutable { y *= 2; return y * y; };
    double first = inner();
    x *= 2;
    return first + inner();
  };
  double first = outer();
  return first + outer();
}

double nested_record_snapshot(double x) {
  auto outer = [&x] {
    auto inner = [v = CopyValue(x)]() mutable {
      v.value *= 2;
      return v.value * v.value;
    };
    x *= 2;
    return inner();
  };
  double first = outer();
  return first + outer();
}

double reference_return_loop(double x) {
  auto inner = [&x]() -> double& { return x; };
  double result = 0;
  for (int i = 0; i < 3; ++i) {
    inner() *= 2;
    result += inner() * inner();
  }
  return result;
}

int main() {
  INIT_GRADIENT(read_before_mutation);
  INIT_GRADIENT(read_reference_initializer);
  INIT_GRADIENT(closure_alias);
  INIT_GRADIENT(mutable_alias);
  INIT_GRADIENT(explicit_call_operator);
  INIT_GRADIENT(mutable_record);
  INIT_GRADIENT(pointer_capture);
  INIT_GRADIENT(reference_return);
  INIT_GRADIENT(reference_return_assignment);
  INIT_GRADIENT(conditional_mutation);
  INIT_GRADIENT(local_storage);
  INIT_GRADIENT(nested_mutation);
  INIT_GRADIENT(reference_to_closure);
  INIT_GRADIENT(nested_local_storage);
  INIT_GRADIENT(nested_snapshot);
  INIT_GRADIENT(nested_record_snapshot);
  INIT_GRADIENT(reference_return_loop);

  double dx = 0;
  TEST_GRADIENT(read_before_mutation, 1, 2, &dx); // CHECK-EXEC: {40.00}
  TEST_GRADIENT(read_reference_initializer, 1, -2, &dx); // CHECK-EXEC-NEXT: {-40.00}
  TEST_GRADIENT(closure_alias, 1, 2, &dx); // CHECK-EXEC-NEXT: {4.00}
  TEST_GRADIENT(mutable_alias, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(explicit_call_operator, 1, -2, &dx); // CHECK-EXEC-NEXT: {-4.00}
  TEST_GRADIENT(mutable_record, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(mutable_record, 1, -2, &dx); // CHECK-EXEC-NEXT: {-80.00}
  TEST_GRADIENT(pointer_capture, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(reference_return, 1, 2, &dx); // CHECK-EXEC-NEXT: {4.00}
  TEST_GRADIENT(reference_return_assignment, 1, 2, &dx); // CHECK-EXEC-NEXT: {16.00}
  TEST_GRADIENT(conditional_mutation, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(conditional_mutation, 1, -2, &dx); // CHECK-EXEC-NEXT: {-360.00}
  TEST_GRADIENT(local_storage, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(nested_mutation, 1, 2, &dx); // CHECK-EXEC-NEXT: {1360.00}
  TEST_GRADIENT(reference_to_closure, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(nested_local_storage, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(nested_snapshot, 1, 2, &dx); // CHECK-EXEC-NEXT: {400.00}
  TEST_GRADIENT(nested_record_snapshot, 1, 2, &dx); // CHECK-EXEC-NEXT: {80.00}
  TEST_GRADIENT(reference_return_loop, 1, 2, &dx); // CHECK-EXEC-NEXT: {336.00}
}
