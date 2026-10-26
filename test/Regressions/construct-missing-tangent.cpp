// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// A construction whose argument has no tangent crashed Sema. Reduced from
// ImpactX, which names a profiling region with std::string(T::type).

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

// CHECK: warning: 'v' is accessed through an expression without a derivative; its derivative is treated as 0
// CHECK: warning: the derivative of this construction is built on the same pointer; what its constructor and destructor do through it happens twice
// CHECK: warning: the derivative of this construction is built on the same pointer; what its constructor and destructor do through it happens twice

struct Drift { static constexpr const char* type = "Drift"; };

// The shape of std::string(const char*): a defaulted allocator argument.
struct Alloc {};
struct Name {
  int n = 0;
  Name() = default;
  Name(const char* s, const Alloc& = Alloc()) : n(s[0] == 'D' ? 5 : 0) {}
};
double length(const Name& s) { return s.n; }

// Several arguments, one of them defaulted.
template <class T> double named(double x) {
  return length(Name(T::type)) * x * x;
}

// CHECK: double named_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     clad::ValueAndPushforward<double, double> _t0 = length_pushforward(Name({Drift::type, Alloc()}), Name(Name()));

// A temporary, of a class that reads its `const char*` like std::string.
struct Label {
  double n = 0;
  Label() = default;
  Label(const char* s, int k) : n(s[0] == 'D' ? k : 0) {}
};

template <class T> double labeled(double x) {
  auto l = Label(T::type, 2);
  return l.n * x * x;
}

// CHECK: double labeled_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     Label _d_l = Label();
// CHECK-NEXT:     Label l = Label(Drift::type, 2);

// An object argument without a tangent.
struct V { double a; };
struct Holder { V v; };
Holder holder{{5}};
Holder* hp = &holder;
struct W {
  double val;
  W(const V& r) : val(r.a) {}
};

double object(double x) {
  W w(hp->v);
  return w.val * x;
}

// CHECK: double object_darg0(double x) {
// CHECK-NEXT:     double _d_x = 1;
// CHECK-NEXT:     V _d_arg = {0.};
// CHECK-NEXT:     W _d_w(_d_arg);
// CHECK-NEXT:     W w(hp->v);

// A pointer to const without a derivative next to an argument with one: the
// derivative object is built on the same pointer, and clad says so.
int scale = 2;
const int* sp = &scale;
struct Scaled {
  double v;
  Scaled(double x, const int* s) : v(x * *s) {}
};

double scaled(double x) {
  Scaled c(x, sp);
  return c.v * x;
}

// A type that points to itself is judged once.
struct Node {
  const Node* next;
  int id;
};
Node node{nullptr, 1};
const Node* np = &node;
struct Linked {
  double v;
  Linked(double x, const Node* n) : v(x * n->id) {}
};

double linked(double x) {
  Linked l(x, np);
  return l.v * x;
}

int main() {
  auto d1 = clad::differentiate(named<Drift>, "x");
  printf("%.2f\n", d1.execute(3)); // CHECK-EXEC: 30.00
  auto d2 = clad::differentiate(labeled<Drift>, "x");
  printf("%.2f\n", d2.execute(3)); // CHECK-EXEC: 12.00
  auto d4 = clad::differentiate(object, "x");
  printf("%.2f\n", d4.execute(3)); // CHECK-EXEC: 5.00
  auto d5 = clad::differentiate(scaled, "x");
  printf("%.2f\n", d5.execute(3)); // CHECK-EXEC: 12.00
  auto d6 = clad::differentiate(linked, "x");
  printf("%.2f\n", d6.execute(3)); // CHECK-EXEC: 6.00
}
