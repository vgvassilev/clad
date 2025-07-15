// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oNonDifferentiable.out 2>&1 | %filecheck %s
// RUN: ./NonDifferentiable.out | %filecheck_exec %s

#define non_differentiable __attribute__((annotate("another_attribute"), annotate("non_differentiable")))

#include "clad/Differentiator/Differentiator.h"

typedef struct {
    int i;
} non_differentiable Input;

void result(int *out, Input in) { *out = in.i; }

void fn(int *out, Input in) { result(out, in); }

class SimpleFunctions1 {
public:
  SimpleFunctions1() noexcept : x(0), y(0), x_pointer(&x), y_pointer(&y) {}
  SimpleFunctions1(double p_x, double p_y) noexcept : x(p_x), y(p_y), x_pointer(&x), y_pointer(&y) {}
  double x;
  non_differentiable double y;
  double* x_pointer;
  non_differentiable double* y_pointer;
  double mem_fn_1(double i, double j) { return (x + y) * i + i * j * j; }
  non_differentiable double mem_fn_2(double i, double j) { return i * j; }
  double mem_fn_3(double i, double j) { return mem_fn_1(i, j) + i * j; }
  double mem_fn_4(double i, double j) { return mem_fn_2(i, j) + i * j; }
  double mem_fn_5(double i, double j) { return mem_fn_2(i, j) * mem_fn_1(i, j) * i; }
  SimpleFunctions1 operator+(const SimpleFunctions1& other) const {
    return SimpleFunctions1(x + other.x, y + other.y);
  }
};

namespace clad {
  template <> void zero_init(SimpleFunctions1& f) {
    f.x = 0;
    f.y = 0;
    f.x_pointer = &f.x;
    f.y_pointer = &f.y;
  }
}

double fn_s1_mem_fn(double i, double j) {
  SimpleFunctions1 obj(2, 3);
  return obj.mem_fn_1(i, j) + i * j;
}

double fn_s1_field(double i, double j) {
  SimpleFunctions1 obj(2, 3);
  return obj.x * obj.y + i * j;
}

double fn_s1_field_pointer(double i, double j) {
  SimpleFunctions1 obj(2, 3);
  return (*obj.x_pointer) * (*obj.y_pointer) + i * j;
}

double fn_s1_operator(double i, double j) {
  SimpleFunctions1 obj1(2, 3);
  SimpleFunctions1 obj2(3, 5);
  return (obj1 + obj2).mem_fn_1(i, j);
}

class non_differentiable SimpleFunctions2 {
public:
  SimpleFunctions2() noexcept : x(0), y(0) {}
  SimpleFunctions2(double p_x, double p_y) noexcept : x(p_x), y(p_y) {}
  double x;
  double y;
  double mem_fn(double i, double j) { return (x + y) * i + i * j * j; }
  SimpleFunctions2 operator+(const SimpleFunctions2& other) const {
    return SimpleFunctions2(x + other.x, y + other.y);
  }
};

double fn_s2_mem_fn(double i, double j) {
  SimpleFunctions2 obj(2, 3);
  return obj.mem_fn(i, j) + i * j;
}

double fn_s2_field(double i, double j) {
  SimpleFunctions2 *obj0, obj(2, 3);
  return obj.x * obj.y + i * j;
}

double fn_s2_operator(double i, double j) {
  SimpleFunctions2 obj1(2, 3);
  SimpleFunctions2 obj2(3, 5);
  return (obj1 + obj2).mem_fn(i, j);
}

double fn_non_diff_var(double i, double j) {
  non_differentiable double k = i * i * j;
  return k;
}

non_differentiable
double fn_non_diff(double i, double j) {
  return i * j;
}

double fn_non_diff_call(double i, double j) {
  return fn_non_diff(i, j) + i * j;
}

double fn_non_diff_param(double i, SimpleFunctions2& S) {
  return S.mem_fn(i, i) + i;
}

double fn_non_diff_param_call(double i, double j) {
  SimpleFunctions2 obj1(j, j);
  return fn_non_diff_param(i, obj1);
}

#define INIT_EXPR(classname)                                                   \
  classname expr_1(2, 3);                                                      \
  classname expr_2(3, 5);

#define TEST_CLASS(classname, name, i, j)                                      \
  auto d_##name = clad::gradient(&classname::name);                            \
  double result_##name[2] = {};						       \
  d_##name.execute(expr_1, i, j, &result_##name[0], &result_##name[1]);	       \
  printf("%.2f %.2f\n\n", result_##name[0], result_##name[1]);

#define TEST_FUNC(name, i, j)                                                  \
  auto d_##name = clad::gradient(&name);	                               \
  double result_##name[2] = {};						       \
  d_##name.execute(i, j, &result_##name[0], &result_##name[1]);		       \
  printf("%.2f %.2f\n\n", result_##name[0], result_##name[1]);

int main() {
  // FIXME: The parts of this test that are commented out are currently not working, due to bugs
  // not related to the implementation of the non-differentiable attribute.
  INIT_EXPR(SimpleFunctions1);

  /*TEST_CLASS(SimpleFunctions1, mem_fn_1, 3, 5)*/

  /*TEST_CLASS(SimpleFunctions1, mem_fn_3, 3, 5)*/
  
  /*TEST_CLASS(SimpleFunctions1, mem_fn_4, 3, 5)*/
  
  /*TEST_CLASS(SimpleFunctions1, mem_fn_5, 3, 5)*/

  TEST_FUNC(fn_s1_mem_fn, 3, 5) // CHECK-EXEC: 35.00 33.00

  TEST_FUNC(fn_s1_field, 3, 5) // CHECK-EXEC: 5.00 3.00

  TEST_FUNC(fn_s1_field_pointer, 3, 5) // CHECK-EXEC: 5.00 3.00

  /*TEST_FUNC(fn_s1_operator, 3, 5)*/

  TEST_FUNC(fn_s2_mem_fn, 3, 5) // CHECK-EXEC: 5.00 3.00

  /*TEST_FUNC(fn_s2_field, 3, 5)*/

  /*TEST_FUNC(fn_s2_operator, 3, 5)*/

  TEST_FUNC(fn_non_diff_var, 3, 5) // CHECK-EXEC: 0.00 0.00

  TEST_FUNC(fn_non_diff_call, 3, 5) // CHECK-EXEC: 5.00 3.00

  TEST_FUNC(fn_non_diff_param_call, 3, 5) // CHECK-EXEC: 1.00 0.00
    // CHECK: void mem_fn_1_pullback(double i, double j, double _d_y, SimpleFunctions1 *_d_this, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     {
    // CHECK-NEXT:         _d_this->x += _d_y * i;
    // CHECK-NEXT:         *_d_i += (this->x + this->y) * _d_y;
    // CHECK-NEXT:         *_d_i += _d_y * j * j;
    // CHECK-NEXT:         *_d_j += i * _d_y * j;
    // CHECK-NEXT:         *_d_j += i * j * _d_y;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }

    // CHECK: void fn_s1_mem_fn_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     SimpleFunctions1 obj(2, 3);
    // CHECK-NEXT:     SimpleFunctions1 _d_obj(obj);
    // CHECK-NEXT:     clad::zero_init(_d_obj);
    // CHECK-NEXT:     SimpleFunctions1 _t0 = obj;
    // CHECK-NEXT:     {
    // CHECK-NEXT:         double _r0 = 0.;
    // CHECK-NEXT:         double _r1 = 0.;
    // CHECK-NEXT:         obj = _t0;
    // CHECK-NEXT:         obj.mem_fn_1_pullback(i, j, 1, &_d_obj, &_r0, &_r1);
    // CHECK-NEXT:         *_d_i += _r0;
    // CHECK-NEXT:         *_d_j += _r1;
    // CHECK-NEXT:         *_d_i += 1 * j;
    // CHECK-NEXT:         *_d_j += i * 1;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }
    
    // CHECK: void fn_s1_field_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     SimpleFunctions1 obj(2, 3);
    // CHECK-NEXT:     SimpleFunctions1 _d_obj(obj);
    // CHECK-NEXT:     clad::zero_init(_d_obj);
    // CHECK-NEXT:     {
    // CHECK-NEXT:         _d_obj.x += 1 * obj.y;
    // CHECK-NEXT:         *_d_i += 1 * j;
    // CHECK-NEXT:         *_d_j += i * 1;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }
    
    // CHECK: void fn_s1_field_pointer_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     SimpleFunctions1 obj(2, 3);
    // CHECK-NEXT:     SimpleFunctions1 _d_obj(obj);
    // CHECK-NEXT:     clad::zero_init(_d_obj);
    // CHECK-NEXT:     {
    // CHECK-NEXT:         *_d_obj.x_pointer += 1 * *obj.y_pointer;
    // CHECK-NEXT:         *_d_i += 1 * j;
    // CHECK-NEXT:         *_d_j += i * 1;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }

    // CHECK: void fn_s2_mem_fn_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     SimpleFunctions2 obj(2, 3);
    // CHECK-NEXT:     {
    // CHECK-NEXT:         *_d_i += 1 * j;
    // CHECK-NEXT:         *_d_j += i * 1;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }

    // CHECK: void fn_non_diff_var_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     double k = i * i * j;
    // CHECK-NEXT: }
  
    // CHECK: void fn_non_diff_call_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:     {
    // CHECK-NEXT:         *_d_i += 1 * j;
    // CHECK-NEXT:         *_d_j += i * 1;
    // CHECK-NEXT:     }
    // CHECK-NEXT: }

    // CHECK:     void fn_non_diff_param_pullback(double i, SimpleFunctions2 &S, double _d_y, double *_d_i) {
    // CHECK-NEXT:         *_d_i += _d_y;
    // CHECK-NEXT:     }

    // CHECK:     void fn_non_diff_param_call_grad(double i, double j, double *_d_i, double *_d_j) {
    // CHECK-NEXT:         SimpleFunctions2 obj1(j, j);
    // CHECK-NEXT:         SimpleFunctions2 _t0 = obj1;
    // CHECK-NEXT:         {
    // CHECK-NEXT:             double _r0 = 0.;
    // CHECK-NEXT:             obj1 = _t0;
    // CHECK-NEXT:             fn_non_diff_param_pullback(i, obj1, 1, &_r0);
    // CHECK-NEXT:             *_d_i += _r0;
    // CHECK-NEXT:         }
    // CHECK-NEXT:     }

  auto grad = clad::gradient(fn, "out");
  // CHECK: void result_pullback(int *out, Input in, int *_d_out) {
  // CHECK-NEXT: int _t0 = *out;
  // CHECK-NEXT: *out = in.i;
  // CHECK-NEXT: {
  // CHECK-NEXT:    *out = _t0;
  // CHECK-NEXT:    int _r_d0 = *_d_out;
  // CHECK-NEXT:    *_d_out = 0;
  // CHECK-NEXT: }
  // CHECK-NEXT:}

  // CHECK: void fn_grad_0(int *out, Input in, int *_d_out) {
  // CHECK-NEXT:    result(out, in);
  // CHECK-NEXT:    result_pullback(out, in, _d_out);
  // CHECK-NEXT:}
}
