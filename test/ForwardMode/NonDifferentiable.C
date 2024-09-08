// RUN: %cladclang %s -I%S/../../include -oNonDifferentiable.out 2>&1 | %filecheck %s
// RUN: ./NonDifferentiable.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#define non_differentiable __attribute__((annotate("another_attribute"), annotate("non_differentiable")))

#include "clad/Differentiator/Differentiator.h"

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

  // CHECK: double mem_fn_1_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions1 *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return (_d_this->x + 0.) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j;
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<double, double> mem_fn_1_pushforward(double i, double j, SimpleFunctions1 *_d_this, double _d_i, double _d_j);

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
  // CHECK-NEXT:     return 0 + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double mem_fn_5_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_this_obj;
  // CHECK-NEXT:     SimpleFunctions1 *_d_this = &_d_this_obj;
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = this->mem_fn_1_pushforward(i, j, _d_this, _d_i, _d_j);
  // CHECK-NEXT:     double _t1 = this->mem_fn_2(i, j);
  // CHECK-NEXT:     double &_t2 = _t0.value;
  // CHECK-NEXT:     double _t3 = _t1 * _t2;
  // CHECK-NEXT:     return (0 * _t2 + _t1 * _t0.pushforward) * i + _t3 * _d_i;
  // CHECK-NEXT: }

  // CHECK: double fn_s1_mem_fn_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_obj({0, 0});
  // CHECK-NEXT:     SimpleFunctions1 obj({2, 3});
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = obj.mem_fn_1_pushforward(i, j, &_d_obj, _d_i, _d_j);
  // CHECK-NEXT:     return _t0.pushforward + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_s1_field_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_obj({0, 0});
  // CHECK-NEXT:     SimpleFunctions1 obj({2, 3});
  // CHECK-NEXT:     double &_t0 = obj.x;
  // CHECK-NEXT:     double &_t1 = obj.y;
  // CHECK-NEXT:     return _d_obj.x * _t1 + _t0 * 0. + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<SimpleFunctions1, SimpleFunctions1> operator_plus_pushforward(const SimpleFunctions1 &other, const SimpleFunctions1 *_d_this, const SimpleFunctions1 &_d_other) const;

  // CHECK: double fn_s1_operator_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions1 _d_obj1({0, 0});
  // CHECK-NEXT:     SimpleFunctions1 obj1({2, 3});
  // CHECK-NEXT:     SimpleFunctions1 _d_obj2({0, 0});
  // CHECK-NEXT:     SimpleFunctions1 obj2({3, 5});
  // CHECK-NEXT:     clad::ValueAndPushforward<SimpleFunctions1, SimpleFunctions1> _t0 = obj1.operator_plus_pushforward(obj2, &_d_obj1, _d_obj2);
  // CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t1 = _t0.value.mem_fn_1_pushforward(i, j, &_t0.pushforward, _d_i, _d_j);
  // CHECK-NEXT:     return _t1.pushforward;
  // CHECK-NEXT: }

  // CHECK: double fn_s2_mem_fn_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions2 obj(2, 3);
  // CHECK-NEXT:     return 0 + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_s2_field_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions2 *obj0, obj(2, 3);
  // CHECK-NEXT:     double &_t0 = obj.x;
  // CHECK-NEXT:     double &_t1 = obj.y;
  // CHECK-NEXT:     return 0. * _t1 + _t0 * 0. + _d_i * j + i * _d_j;
  // CHECK-NEXT: }

  // CHECK: double fn_s2_operator_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     SimpleFunctions2 obj1(2, 3);
  // CHECK-NEXT:     SimpleFunctions2 obj2(3, 5);
  // CHECK-NEXT:     return 0;
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<double, double> mem_fn_1_pushforward(double i, double j, SimpleFunctions1 *_d_this, double _d_i, double _d_j) {
  // CHECK-NEXT:     double _t0 = (this->x + this->y);
  // CHECK-NEXT:     double _t1 = i * j;
  // CHECK-NEXT:     return {_t0 * i + _t1 * j, (_d_this->x + 0.) * i + _t0 * _d_i + (_d_i * j + i * _d_j) * j + _t1 * _d_j};
  // CHECK-NEXT: }

  // CHECK: clad::ValueAndPushforward<SimpleFunctions1, SimpleFunctions1> operator_plus_pushforward(const SimpleFunctions1 &other, const SimpleFunctions1 *_d_this, const SimpleFunctions1 &_d_other) const {
  // CHECK-NEXT:     return {SimpleFunctions1(this->x + other.x, this->y + other.y), SimpleFunctions1(_d_this->x + _d_other.x, 0. + 0.)};
  // CHECK-NEXT: }

}