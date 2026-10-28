// RUN: %cladclang -fsyntax-only -ferror-limit=0 -std=c++17 \
// RUN:   -Xclang -verify=expected,refused -I%S/../../include %s
// RUN: %cladclang -DACCEPTED -std=c++17 -I%S/../../include %s -o %t \
// RUN:   -Xclang -verify 2>&1 | %filecheck %s
// RUN: %t | %filecheck_exec %s

// Reverse mode points the adjoint of a global pointer at the adjoint of an
// object only where the differentiated code assigns the pointer. Until then
// the adjoint is null: what the pointer points to when the gradient starts has
// no adjoint. A derivative that needs it is refused, instead of a gradient that
// dereferences null. The second run builds and executes only the gradients
// that are accepted.

#include "clad/Differentiator/Differentiator.h"
#include <cstdio>

struct P { double x; };
P obj, other; // expected-warning {{gradient uses a global variable 'obj'; rerunning the gradient requires 'obj' to be reset}} \
              // expected-warning {{gradient uses a global variable 'other'; rerunning the gradient requires 'other' to be reset}}
P* gp = &obj; // expected-warning {{gradient uses a global variable 'gp'; rerunning the gradient requires 'gp' to be reset}} \
              // refused-note 9 {{'gp' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double gd = 0; // expected-warning {{gradient uses a global variable 'gd'; rerunning the gradient requires 'gd' to be reset}}
struct H { double* p; double v; };
H gh{&gd, 1}; // expected-warning {{gradient uses a global variable 'gh'; rerunning the gradient requires 'gh' to be reset}} \
              // refused-note {{'gh.p' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double* gv = &gd; // expected-warning {{gradient uses a global variable 'gv'; rerunning the gradient requires 'gv' to be reset}} \
                  // refused-note 2 {{'gv' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
void add_scaled(double* out, double x) {
  *out += *gv * x; // refused-error {{cannot differentiate this access through 'gv': the derivative of what it points to is unknown here}}
}

#ifndef ACCEPTED
//===--- Refused ----------------------------------------------------------===//

// The reproducer from the issue: a write, then a read, through the pointer.
double write_read(double a) {
  gp->x = a * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
  return gp->x * a;
}

// A read alone needs the adjoint too: its derivative is added to it.
double read_only(double a) {
  return gp->x * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}

// The access comes before the assignment.
double assigned_after(double a) {
  double r = gp->x * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
  gp = &obj;
  return r;
}

// Every kind of access, and an integer, which clad differentiates too.
double* gdp = &gd; // refused-warning {{gradient uses a global variable 'gdp'; rerunning the gradient requires 'gdp' to be reset}} \
                   // refused-note 5 {{'gdp' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double deref(double a) {
  *gdp = a * a; // refused-error {{cannot differentiate this access through 'gdp': the derivative of what it points to is unknown here}}
  return *gdp * a;
}
double subscript(double a) {
  gdp[0] = a; // refused-error {{cannot differentiate this access through 'gdp': the derivative of what it points to is unknown here}}
  return gdp[0] * a;
}
int gi = 0;
int* gip = &gi; // refused-warning {{gradient uses a global variable 'gip'; rerunning the gradient requires 'gip' to be reset}} \
                // refused-note {{'gip' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double integer(double a) {
  int n = *gip; // refused-error {{cannot differentiate this access through 'gip': the derivative of what it points to is unknown here}}
  return n * a;
}
double* gdp_target = &gd;
double** gpp = &gdp_target; // refused-warning {{gradient uses a global variable 'gpp'; rerunning the gradient requires 'gpp' to be reset}} \
                            // refused-note {{'gpp' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double pointer_to_pointer(double a) {
  **gpp = a; // refused-error {{cannot differentiate this access through 'gpp': the derivative of what it points to is unknown here}}
  return **gpp * a;
}

// The value is followed into a local copy.
double local_copy(double a) {
  double* p = gdp; // refused-note {{'p' is copied from 'gdp' here}}
  return *p * a; // refused-error {{cannot differentiate this access through 'gdp': the derivative of what it points to is unknown here}}
}

// A pointer member of a global object, and an element of a global array.
double member(double a) {
  return *gh.p * a; // refused-error {{cannot differentiate this access through 'gh.p': the derivative of what it points to is unknown here}}
}
double* gpa[2] = {&gd, &gd}; // refused-warning {{gradient uses a global variable 'gpa'; rerunning the gradient requires 'gpa' to be reset}} \
                             // refused-note {{'gpa[1]' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double element(double a) {
  gpa[0] = &gd;
  return *gpa[0] * *gpa[1] * a; // refused-error {{cannot differentiate this access through 'gpa[1]': the derivative of what it points to is unknown here}}
}
// An element whose index is not a constant may be any of them.
double any_element(double a, int i) {
  gpa[0] = &gd;
  gpa[1] = &gd;
  gpa[i] = &gd;
  gpa[i] = gdp;
  return *gpa[0] * a; // refused-error {{cannot differentiate this access through 'gdp': the derivative of what it points to is unknown here}}
}
// A part of a local object.
double part(double a) {
  H h{&gd, 1};
  h.p = gdp; // refused-note {{'h.p' is copied from 'gdp' here}}
  H copy = h;
  return *copy.p * a; // refused-error {{cannot differentiate this access through 'gdp': the derivative of what it points to is unknown here}}
}

// What the pointer evaluates to through arithmetic, operators and calls.
double arithmetic(double a) {
  P* p = gp + 0; // refused-note {{'p' is copied from 'gp' here}}
  P* q = 0 + gp;
  return (p->x + q->x + (+gp)->x) * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}
double incremented(double a) {
  P* p = gp++; // refused-note {{'p' is copied from 'gp' here}}
  return p->x * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}
double operators(double a) {
  P* p = ((void)0, gp); // refused-note {{'p' is copied from 'gp' here}}
  P* local = &obj;
  P* q = a > 0 ? local : gp;
  return (p->x + q->x) * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}
double through_address(double a) {
  P* p = *&gp; // refused-note {{'p' is copied from 'gp' here}} \
               // refused-note {{the address of 'gp' is taken here, and clad does not follow its derivative through it}}
  return p->x * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}
P* current() { return gp; }
double returned(double a) {
  return current()->x * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}
struct In { double x; };
struct Base { In in; };
struct Derived : Base {};
Derived gder;
Derived* gpder = &gder; // refused-warning {{gradient uses a global variable 'gpder'; rerunning the gradient requires 'gpder' to be reset}} \
                        // refused-note {{'gpder' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double base_member(double a) {
  double* q = &(*gpder).in.x; // refused-error {{cannot differentiate this access through 'gpder': the derivative of what it points to is unknown here}}
  return *q * a;
}

// A default member initializer, a method, an operator and a constructor.
struct Q { P* p = gp; };
double default_member(double a) {
  Q q{}; // refused-note {{'q.p' is copied from 'gp' here}}
  return q.p->x * a; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
}
struct W {
  P* p;
  double operator()(double a) {
    return p->x * a; // refused-error {{cannot differentiate this access through 'gw.p': the derivative of what it points to is unknown here}}
  }
  void retarget() { p = &other; }
};
W gw{&obj}; // refused-warning {{gradient uses a global variable 'gw'; rerunning the gradient requires 'gw' to be reset}} \
            // refused-note {{'gw.p' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double member_operator(double a) {
  return gw(a); // refused-note {{in the derivative of 'operator()', called here}}
}
double method_reassigns(double a) {
  gw.p = &obj;
  gw.retarget(); // refused-note {{'gw' may be reassigned by this call to 'retarget', which does not update its derivative}}
  return gw.p->x * a; // refused-error {{cannot differentiate this access through 'gw': the derivative of what it points to is unknown here}}
}
struct Holder {
  P* p;
  double v;
  Holder(P* q) : p(q) {
    v = p->x; // refused-error {{cannot differentiate this access through 'gp': the derivative of what it points to is unknown here}}
  }
};
double constructor(double a) {
  Holder h(gp); // refused-note {{'q' is copied from 'gp' here}} \
                // refused-note {{in the derivative of 'Holder', called here}}
  return h.v * a;
}
struct S { static P* sp; };
P* S::sp = &obj; // refused-note {{'sp' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
S gs; // refused-warning {{gradient uses a global variable 'gs'; rerunning the gradient requires 'gs' to be reset}}
double static_member(double a) {
  P* p = gs.sp; // refused-note {{'p' is copied from 'sp' here}}
  return p->x * a; // refused-error {{cannot differentiate this access through 'sp': the derivative of what it points to is unknown here}}
}

// An assignment counts only where it is certain to have run.
P* gc = &obj; // refused-warning {{gradient uses a global variable 'gc'; rerunning the gradient requires 'gc' to be reset}} \
              // refused-note 3 {{'gc' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double one_branch(double a) {
  if (a > 0)
    gc = &obj;
  return gc->x * a; // refused-error {{cannot differentiate this access through 'gc': the derivative of what it points to is unknown here}}
}
double end_of_iteration(double a) {
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    r += gc->x * a; // refused-error {{cannot differentiate this access through 'gc': the derivative of what it points to is unknown here}}
    gc = &obj;
  }
  return r;
}
double switch_without_default(double a) {
  switch ((int)a) {
  case 2:
    gc = &obj;
    break;
  }
  return gc->x * a; // refused-error {{cannot differentiate this access through 'gc': the derivative of what it points to is unknown here}}
}

// A call that may reassign the pointer does so without its adjoint.
P* gq = &obj; // refused-warning {{gradient uses a global variable 'gq'; rerunning the gradient requires 'gq' to be reset}}
void point_at_other() { gq = &other; }
double reassigned_by_callee(double a) {
  gq = &obj;
  point_at_other(); // refused-note {{'gq' may be reassigned by this call to 'point_at_other', which does not update its derivative}}
  return gq->x * a; // refused-error {{cannot differentiate this access through 'gq': the derivative of what it points to is unknown here}}
}
double reassigned_in_loop(double a) {
  gq = &obj;
  double r = 0;
  for (int i = 0; i < 2; ++i) {
    r += gq->x * a; // refused-error {{cannot differentiate this access through 'gq': the derivative of what it points to is unknown here}}
    point_at_other(); // refused-note {{'gq' may be reassigned by this call to 'point_at_other', which does not update its derivative}}
  }
  return r;
}
void bind(P*& r) { r = &other; }
double passed_by_reference(double a) {
  gq = &obj;
  bind(gq); // refused-note {{'gq' is passed by reference to 'bind' here, which may reassign it without updating its derivative}}
  return gq->x * a; // refused-error {{cannot differentiate this access through 'gq': the derivative of what it points to is unknown here}}
}
void bind_address(P** r) { *r = &other; }
double passed_by_address(double a) {
  gq = &obj;
  bind_address(&gq); // refused-note {{'gq' is passed by reference to 'bind_address' here, which may reassign it without updating its derivative}}
  return gq->x * a; // refused-error {{cannot differentiate this access through 'gq': the derivative of what it points to is unknown here}}
}
double address_kept(double a) {
  gq = &obj;
  P** q = &gq; // refused-note {{the address of 'gq' is taken here, and clad does not follow its derivative through it}}
  *q = &other;
  return gq->x * a; // refused-error {{cannot differentiate this access through 'gq': the derivative of what it points to is unknown here}}
}

// The pullback of a callee reads the adjoint the caller left.
double callee_reads(double a) {
  double r = 0;
  add_scaled(&r, a); // refused-note {{in the derivative of 'add_scaled', called here}}
  return r;
}
double square_into(double* p, double x) {
  *p = x * x; // refused-error {{cannot differentiate this access through 'gv': the derivative of what it points to is unknown here}}
  return *p * x;
}
double passed_to_callee(double a) {
  return square_into(gv, a); // refused-note {{'p' is copied from 'gv' here}} \
                             // refused-note {{in the derivative of 'square_into', called here}}
}

// A derivative clad cannot see into is handed the adjoint as it is.
double* gw2 = &gd; // refused-warning {{gradient uses a global variable 'gw2'; rerunning the gradient requires 'gw2' to be reset}} \
                   // refused-note {{'gw2' has no derivative to point to until the differentiated code assigns it; assign it before the access, or pass what it points to as a parameter}}
double scale_by(double* p, double x);
namespace clad {
namespace custom_derivatives {
void scale_by_pullback(double* p, double x, double d_y, double* d_p,
                       double* d_x) {
  *d_p += d_y * x;
  *d_x += d_y * *p;
}
} // namespace custom_derivatives
} // namespace clad
double custom(double a) {
  return scale_by(gw2, a); // refused-error {{cannot differentiate passing 'gw2' to 'scale_by': the derivative of what it points to is unknown here}}
}

// A global reference cannot be given an adjoint at all.
P& gr = obj; // refused-note {{pass what 'gr' refers to as a parameter instead}}
double reference(double a) {
  gr.x = a; // refused-error {{cannot differentiate this use of the global reference 'gr': clad cannot give a reference a derivative}}
  return gr.x * a;
}

void refused() {
  clad::gradient(write_read);
  clad::gradient(read_only);
  clad::gradient(assigned_after);
  clad::gradient(deref);
  clad::gradient(subscript);
  clad::gradient(integer);
  clad::gradient(pointer_to_pointer);
  clad::gradient(local_copy);
  clad::gradient(member);
  clad::gradient(element);
  clad::gradient(any_element);
  clad::gradient(part);
  clad::gradient(arithmetic);
  clad::gradient(incremented);
  clad::gradient(operators);
  clad::gradient(through_address);
  clad::gradient(returned);
  clad::gradient(base_member);
  clad::gradient(default_member);
  clad::gradient(member_operator);
  clad::gradient(method_reassigns);
  clad::gradient(constructor);
  clad::gradient(static_member);
  clad::gradient(one_branch);
  clad::gradient(end_of_iteration);
  clad::gradient(switch_without_default);
  clad::gradient(reassigned_by_callee);
  clad::gradient(reassigned_in_loop);
  clad::gradient(passed_by_reference);
  clad::gradient(passed_by_address);
  clad::gradient(address_kept);
  clad::gradient(callee_reads);
  clad::gradient(passed_to_callee);
  clad::gradient(custom);
  clad::gradient(reference);
}
#endif // ACCEPTED

//===--- Accepted ---------------------------------------------------------===//

// The adjoint is known once the differentiated code assigns the pointer, on
// every path to the access. f = a^3, f' = 3a^2.
double assigned(double a) {
  gp = &obj;
  gp->x = a * a;
  return gp->x * a;
}

// CHECK: void assigned_grad(double a, double *_d_a) {
// CHECK:     _d_gp = &_d_obj;
// CHECK-NEXT:     gp = &obj;

double both_branches(double a) {
  if (a > 0)
    gp = &obj;
  else
    gp = &other;
  gp->x = a * a;
  return gp->x * a;
}

double every_case(double a) {
  switch ((int)a) {
  case 1:
    gp = &obj;
    break;
  default:
    gp = &other;
  }
  gp->x = a * a;
  return gp->x * a;
}

double early_return(double a) {
  if (a < 0)
    return a;
  gp = &obj;
  gp->x = a * a;
  return gp->x * a;
}

double in_each_iteration(double a) {
  double r = 0;
  for (int i = 0; i < 1; ++i) {
    gp = &obj;
    gp->x = a * a;
    r += gp->x * a;
  }
  return r;
}

double left_by_break(double a) {
  while (true) {
    gp = &obj;
    break;
  }
  gp->x = a * a;
  return gp->x * a;
}

// What follows a return is never run.
double unreachable(double a) {
  gp = &obj;
  gp->x = a * a;
  return gp->x * a;
  gp = &other;
  return gp->x;
}

// The member of a global object.
double member_assigned(double a) {
  gh.p = &gd;
  *gh.p = a * a;
  return *gh.p * a;
}

// A callee reads the adjoint the caller gave the pointer. f = 3a, f' = 3.
double callee_reads_assigned(double a) {
  gv = &gd;
  gd = 3;
  double r = 0;
  add_scaled(&r, a); // expected-warning {{'add_scaled' reads the global 'gv' through a pointer or reference; clad does not differentiate through it}}
  return r;
}

// A pointer tested, compared or copied without being accessed through needs
// no adjoint. f = a^2, f' = 2a.
double tested(double a) { return gp ? a * a : a; }
double compared(double a) { return gp == &obj ? a * a : a; }
double copied(double a) {
  P* p = gp;
  (void)p;
  return a * a;
}
// A condition is not differentiated. f = a^2 while obj.x > 0.
double in_condition(double a) { return gp->x > 0 ? a * a : a; }

// A const pointer has no adjoint: what it points to is read as a constant.
const double table[] = {2, 4};
const double* const ptable = table;
double constant(double a) { return ptable[1] * a; }

#define TEST(F, want)                                                          \
  {                                                                            \
    double d = 0;                                                              \
    clad::gradient(F).execute(a, &d);                                          \
    printf("%s %.2f %.2f\n", #F, d, want);                                     \
  }

int main() {
  double a = 2;
  TEST(assigned, 3 * a * a);              // CHECK-EXEC: assigned 12.00 12.00
  TEST(both_branches, 3 * a * a);         // CHECK-EXEC: both_branches 12.00 12.00
  TEST(every_case, 3 * a * a);            // CHECK-EXEC: every_case 12.00 12.00
  TEST(early_return, 3 * a * a);          // CHECK-EXEC: early_return 12.00 12.00
  TEST(in_each_iteration, 3 * a * a);     // CHECK-EXEC: in_each_iteration 12.00 12.00
  TEST(left_by_break, 3 * a * a);         // CHECK-EXEC: left_by_break 12.00 12.00
  TEST(unreachable, 3 * a * a);           // CHECK-EXEC: unreachable 12.00 12.00
  TEST(member_assigned, 3 * a * a);       // CHECK-EXEC: member_assigned 12.00 12.00
  TEST(callee_reads_assigned, 3.0);       // CHECK-EXEC: callee_reads_assigned 3.00 3.00
  TEST(tested, 2 * a);                    // CHECK-EXEC: tested 4.00 4.00
  TEST(compared, 2 * a);                  // CHECK-EXEC: compared 4.00 4.00
  TEST(copied, 2 * a);                    // CHECK-EXEC: copied 4.00 4.00
  obj.x = 1;
  TEST(in_condition, 2 * a);              // CHECK-EXEC: in_condition 4.00 4.00
  TEST(constant, 4.0);                    // CHECK-EXEC: constant 4.00 4.00
}
