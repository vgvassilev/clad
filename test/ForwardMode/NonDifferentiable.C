// RUN: %cladclang %s -I%S/../../include -oNonDifferentiable.out 2>&1 | %filecheck %s
// RUN: ./NonDifferentiable.out | %filecheck_exec %s

#define non_differentiable __attribute__((annotate("another_attribute"), annotate("non_differentiable")))

#include "clad/Differentiator/Differentiator.h"

#include <new>

extern "C" int printf(const char* fmt, ...);

class SimpleFunctions1 {
public:
  SimpleFunctions1() noexcept : x(0), y(0) {}
  SimpleFunctions1(double p_x, double p_y) noexcept : x(p_x), y(p_y) {}
  double x;
  non_differentiable double y;
  double mem_fn_1(double i, double j) { return (x + y) * i + i * j * j; }
  non_differentiable double mem_fn_2(double i, double j) { return i * j; }
  double mem_fn_3(double i, double j) { return mem_fn_1(i, j) + i * j; }
  double mem_fn_4(double i, double j) { return mem_fn_2(i, j) + i * j; }
  double mem_fn_5(double i, double j) { return mem_fn_2(i, j) * mem_fn_1(i, j) * i; }
  SimpleFunctions1 operator+(const SimpleFunctions1& other) const {
    return SimpleFunctions1(x + other.x, y + other.y);
  }
};

double fn_s1_mem_fn(double i, double j) {
  SimpleFunctions1 obj(2, 3);
  return obj.mem_fn_1(i, j) + i * j;
}

double fn_s1_field(double i, double j) {
  SimpleFunctions1 obj(2, 3);
  return obj.x * obj.y + i * j;
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

non_differentiable
double fn_non_diff(double i, double j) {
  return i * j;
}

double fn_non_diff_call(double i, double j) {
  return fn_non_diff(i, j) + i * j;
}

non_differentiable void release(int* p) { delete p; }

// The arguments of a non-differentiable call are cloned as written.
double fn_non_diff_new(double i, double j) {
  release(new (std::nothrow) int(1));
  return i * j;
}

double assignedArguments(double u, double v) {
  double r = 0, s = 0;
  fn_non_diff(r = u * v, s = u + v);
  return r + s;
}

non_differentiable double opaqueArgument(double x) { return x; }

double conditionalArguments(double u, bool enabled) {
  double r = u;
  opaqueArgument(enabled && (r = u * 3));
  return r;
}

double compoundArguments(double u) {
  double r = u;
  opaqueArgument(r *= u);
  return r;
}

double incrementedArguments(double u) {
  double values[] = {u, u * u};
  double* p = values;
  int i = 0;
  opaqueArgument(i++);
  opaqueArgument(*p++);
  return *p + i * u;
}

double cubedArgument(double u) {
  double r = 0;
  opaqueArgument(r = u * u * u);
  return r;
}

double integerArgument(double u) {
  int i = 0;
  opaqueArgument(i = u);
  return i * u;
}

double writesInHelper(double u) {
  double r = 0;
  opaqueArgument(r = u * u);
  return r;
}

double callsHelper(double u) { return writesInHelper(u) * 2; }

double elvisArgument(double u, double z) {
  double r = 0;
  opaqueArgument(z ?: (r = u * u));
  return r;
}

#define INIT_EXPR(classname)                                                   \
  classname expr_1(2, 3);                                                      \
  classname expr_2(3, 5);

#define TEST_CLASS(classname, name, i, j)                                      \
  auto d_##name = clad::differentiate(&classname::name, "i");                  \
  printf("%.2f\n", d_##name.execute(expr_1, i, j));                            \
  printf("%.2f\n", d_##name.execute(expr_2, i, j));                            \
  printf("\n");

#define TEST_FUNC(name, i, j)                                                  \
  auto d_##name = clad::differentiate(&name, "i");                             \
  printf("%.2f\n", d_##name.execute(i, j));                                    \
  printf("\n");

int main() {
  INIT_EXPR(SimpleFunctions1);

  TEST_CLASS(SimpleFunctions1, mem_fn_1, 3, 5) // CHECK-EXEC: 30.00
                                               // CHECK-EXEC: 33.00

  TEST_CLASS(SimpleFunctions1, mem_fn_3, 3, 5) // CHECK-EXEC: 35.00
                                               // CHECK-EXEC: 38.00

  TEST_CLASS(SimpleFunctions1, mem_fn_4, 3, 5) // CHECK-EXEC: 5.00
                                               // CHECK-EXEC: 5.00

  TEST_CLASS(SimpleFunctions1, mem_fn_5, 3, 5) // CHECK-EXEC: 2700.00
                                                // CHECK-EXEC: 2970.00

  TEST_FUNC(fn_s1_mem_fn, 3, 5) // CHECK-EXEC: 35.00

  TEST_FUNC(fn_s1_field, 3, 5) // CHECK-EXEC: 5.00

  TEST_FUNC(fn_s1_operator, 3, 5) // CHECK-EXEC: 38.00

  TEST_FUNC(fn_s2_mem_fn, 3, 5) // CHECK-EXEC: 5.00

  TEST_FUNC(fn_s2_field, 3, 5) // CHECK-EXEC: 5.00

  TEST_FUNC(fn_s2_operator, 3, 5) // CHECK-EXEC: 0.00

  TEST_FUNC(fn_non_diff_call, 3, 5) // CHECK-EXEC: 5.00

  TEST_FUNC(fn_non_diff_new, 3, 5) // CHECK-EXEC: 5.00

  auto assigned = clad::differentiate(assignedArguments, "u");
  printf("%.2f\n", assigned.execute(2, 3)); // CHECK-EXEC: 4.00
  auto conditional = clad::differentiate(conditionalArguments, "u");
  printf("%.2f\n", conditional.execute(2, false)); // CHECK-EXEC: 1.00
  printf("%.2f\n", conditional.execute(2, true)); // CHECK-EXEC: 3.00
  auto compound = clad::differentiate(compoundArguments, "u");
  printf("%.2f\n", compound.execute(2)); // CHECK-EXEC: 4.00
  auto incremented = clad::differentiate(incrementedArguments, "u");
  printf("%.2f\n", incremented.execute(2)); // CHECK-EXEC: 5.00
  auto assignedVector =
      clad::differentiate<clad::opts::vector_mode>(assignedArguments);
  double du = 0, dv = 0;
  assignedVector.execute(2, 3, &du, &dv);
  printf("%.2f %.2f\n", du, dv); // CHECK-EXEC: 4.00 3.00
  auto conditionalVector =
      clad::differentiate<clad::opts::vector_mode>(conditionalArguments, "u");
  du = 0;
  conditionalVector.execute(2, false, &du);
  printf("%.2f\n", du); // CHECK-EXEC: 1.00
  du = 0;
  conditionalVector.execute(2, true, &du);
  printf("%.2f\n", du); // CHECK-EXEC: 3.00
  auto cubed = clad::differentiate(cubedArgument, "u");
  printf("%.2f\n", cubed.execute(2)); // CHECK-EXEC: 12.00
  auto integer = clad::differentiate(integerArgument, "u");
  printf("%.2f\n", integer.execute(2.5)); // CHECK-EXEC: 2.00
  auto helper = clad::differentiate(callsHelper, "u");
  printf("%.2f\n", helper.execute(3)); // CHECK-EXEC: 12.00
  auto elvis = clad::differentiate(elvisArgument, "u");
  printf("%.2f\n", elvis.execute(3, 0)); // CHECK-EXEC: 6.00

  // CHECK: double mem_fn_1_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions1 *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return _d_this->x * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<double, double> mem_fn_1_pushforward(double i, double j, SimpleFunctions1 *_d_this, double _d_i, double _d_j) {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return {_t0 * i + _t1 * j, _d_this->x * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j};
  // CHECK-NEXT: }

  // CHECK: double mem_fn_3_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions1 *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = this->mem_fn_1_pushforward(i, j, _d_this, _d_i, _d_j);
  // CHECK-NEXT:     return _t0.pushforward + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double mem_fn_4_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions1 *_d_this = &_d_this_obj;
  // CHECK-NEXT:     return _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double mem_fn_5_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions1 *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = this->mem_fn_1_pushforward(i, j, _d_this, _d_i, _d_j);
  // CHECK-NEXT:     double _t1 = this->mem_fn_2(i, j);
  // CHECK-NEXT:     double _t2 = _t1 * _t0.value;
  // CHECK-NEXT:     return (_t1 * _t0.pushforward) * i + _t2 * _d_i;
  // CHECK-NEXT: }

  // CHECK: double fn_s1_mem_fn_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_obj{0, 0};
  // CHECK-NEXT:     SimpleFunctions1 obj{2, 3};
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = obj.mem_fn_1_pushforward(i, j, &_d_obj, _d_i, _d_j);
  // CHECK-NEXT:     return _t0.pushforward + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_s1_field_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_obj{0, 0};
  // CHECK-NEXT:     SimpleFunctions1 obj{2, 3};
  // CHECK-NEXT:     double &_t0 = obj.y;
  // CHECK-NEXT:     return _d_obj.x * _t0 + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<SimpleFunctions1, SimpleFunctions1> operator_plus_pushforward(const SimpleFunctions1 &other, const SimpleFunctions1 *_d_this, const SimpleFunctions1 &_d_other) const {
  // CHECK-NEXT:     return {SimpleFunctions1(this->x + other.x, this->y + other.y), SimpleFunctions1(_d_this->x + _d_other.x, 0.)};
  // CHECK-NEXT: }

  // CHECK: double fn_s1_operator_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_obj1{0, 0};
  // CHECK-NEXT:     SimpleFunctions1 obj1{2, 3};
  // CHECK-NEXT:     SimpleFunctions1 _d_obj2{0, 0};
  // CHECK-NEXT:     SimpleFunctions1 obj2{3, 5};
  // CHECK-NEXT:     clad::ValueAndPushforward<SimpleFunctions1, SimpleFunctions1> _t0 = obj1.operator_plus_pushforward(obj2, &_d_obj1, _d_obj2);
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = _t0.value.mem_fn_1_pushforward(i, j, &_t0.pushforward, _d_i, _d_j);
  // CHECK-NEXT:     return _t1.pushforward;
  // CHECK-NEXT: }

  // CHECK: double fn_s2_mem_fn_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions2 obj(2, 3);
  // CHECK-NEXT:     return _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_s2_field_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions2 *obj0, obj(2, 3);
  // CHECK-NEXT:     return _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_s2_operator_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions2 obj1(2, 3);
  // CHECK-NEXT:     SimpleFunctions2 obj2(3, 5);
  // CHECK-NEXT:     return 0;
  // CHECK-NEXT: }

  // CHECK: double fn_non_diff_call_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     return _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_non_diff_new_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     release(new (std::nothrow) int(1));
  // CHECK-NEXT:     return _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double assignedArguments_darg0(double u, double v) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     double _d_v = 0;
  // CHECK-NEXT:     double _d_r = 0, _d_s = 0;
  // CHECK-NEXT:     double r = 0, s = 0;
  // CHECK-NEXT:     fn_non_diff(((_d_r = _d_u * v + u * _d_v) , (r = u * v)), ((_d_s = _d_u + _d_v) , (s = u + v)));
  // CHECK-NEXT:     return _d_r + _d_s;
  // CHECK-NEXT: }

  // CHECK: double conditionalArguments_darg0(double u, bool enabled) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     bool _d_enabled = 0;
  // CHECK-NEXT:     double _d_r = _d_u;
  // CHECK-NEXT:     double r = u;
  // CHECK-NEXT:     opaqueArgument(enabled && (((_d_r = _d_u * 3) , (r = u * 3))));
  // CHECK-NEXT:     return _d_r;
  // CHECK-NEXT: }

  // CHECK: double compoundArguments_darg0(double u) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     double _d_r = _d_u;
  // CHECK-NEXT:     double r = u;
  // CHECK-NEXT:     opaqueArgument(((_d_r = _d_r * u + r * _d_u) , (r *= u)));
  // CHECK-NEXT:     return _d_r;
  // CHECK-NEXT: }

  // CHECK: double incrementedArguments_darg0(double u) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     double _d_values[2] = {_d_u, _d_u * u + u * _d_u};
  // CHECK-NEXT:     double values[2] = {u, u * u};
  // CHECK-NEXT:     double *_d_p = _d_values;
  // CHECK-NEXT:     double *p = values;
  // CHECK-NEXT:     int _d_i = 0;
  // CHECK-NEXT:     int i = 0;
  // CHECK-NEXT:     opaqueArgument(i++);
  // CHECK-NEXT:     opaqueArgument(*(_d_p++ , p++));
  // CHECK-NEXT:     return *_d_p + _d_i * u + i * _d_u;
  // CHECK-NEXT: }

  // CHECK: void assignedArguments_dvec(double u, double v, double *_d_u, double *_d_v) {
  // CHECK-NEXT:     unsigned long indepVarCount = 2UL;
  // CHECK-NEXT:     clad::array<double> _d_vector_u = clad::one_hot_vector(indepVarCount, 0UL);
  // CHECK-NEXT:     clad::array<double> _d_vector_v = clad::one_hot_vector(indepVarCount, 1UL);
  // CHECK-NEXT:     clad::array<double> _d_vector_r(clad::zero_vector(indepVarCount)), _d_vector_s(clad::zero_vector(indepVarCount));
  // CHECK-NEXT:     double r = 0, s = 0;
  // CHECK-NEXT:     fn_non_diff(((_d_vector_r = _d_vector_u * v + u * _d_vector_v) , (r = u * v)), ((_d_vector_s = _d_vector_u + _d_vector_v) , (s = u + v)));
  // CHECK-NEXT:     {
  // CHECK-NEXT:         clad::array<double> _d_vector_return(_d_vector_r + _d_vector_s);
  // CHECK-NEXT:         *_d_u = _d_vector_return[0UL];
  // CHECK-NEXT:         *_d_v = _d_vector_return[1UL];
  // CHECK-NEXT:         return;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  // CHECK: void conditionalArguments_dvec_0(double u, bool enabled, double *_d_u) {
  // CHECK-NEXT:     unsigned long indepVarCount = 1UL;
  // CHECK-NEXT:     clad::array<double> _d_vector_u = clad::one_hot_vector(indepVarCount, 0UL);
  // CHECK-NEXT:     clad::array<bool> _d_vector_enabled = clad::zero_vector(indepVarCount);
  // CHECK-NEXT:     clad::array<double> _d_vector_r(_d_vector_u);
  // CHECK-NEXT:     double r = u;
  // CHECK-NEXT:     opaqueArgument(enabled && (((_d_vector_r = _d_vector_u * 3 + u * (clad::zero_vector(indepVarCount))) , (r = u * 3))));
  // CHECK-NEXT:     {
  // CHECK-NEXT:         clad::array<double> _d_vector_return(_d_vector_r);
  // CHECK-NEXT:         *_d_u = _d_vector_return[0UL];
  // CHECK-NEXT:         return;
  // CHECK-NEXT:     }
  // CHECK-NEXT: }

  // CHECK: double cubedArgument_darg0(double u) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     double _d_r = 0;
  // CHECK-NEXT:     double r = 0;
  // CHECK-NEXT:     opaqueArgument(((_d_r = (_d_u * u + u * _d_u) * u + (u * u) * _d_u) , (r = u * u * u)));
  // CHECK-NEXT:     return _d_r;
  // CHECK-NEXT: }

  // CHECK: double integerArgument_darg0(double u) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     int _d_i = 0;
  // CHECK-NEXT:     int i = 0;
  // CHECK-NEXT:     opaqueArgument(i = u);
  // CHECK-NEXT:     return _d_i * u + i * _d_u;
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<double, double> writesInHelper_pushforward(double u, double _d_u) {
  // CHECK-NEXT:     double _d_r = 0;
  // CHECK-NEXT:     double r = 0;
  // CHECK-NEXT:     opaqueArgument(((_d_r = _d_u * u + u * _d_u) , (r = u * u)));
  // CHECK-NEXT:     return {r, _d_r};
  // CHECK-NEXT: }

  // CHECK: double callsHelper_darg0(double u) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = writesInHelper_pushforward(u, _d_u);
  // CHECK-NEXT:     return _t0.pushforward * 2;
  // CHECK-NEXT: }

  // CHECK: double elvisArgument_darg0(double u, double z) {
  // CHECK-NEXT:     double _d_u = 1;
  // CHECK-NEXT:     double _d_z = 0;
  // CHECK-NEXT:     double _d_r = 0;
  // CHECK-NEXT:     double r = 0;
  // CHECK-NEXT:     opaqueArgument(z ?: (((_d_r = _d_u * u + u * _d_u) , (r = u * u))));
  // CHECK-NEXT:     return _d_r;
  // CHECK-NEXT: }
}
