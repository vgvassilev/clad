// RUN: %cladnumdiffclang %s -I%S/../../include -oUserDefinedPointers.out -Xclang -verify 2>&1
// RUN: ./UserDefinedPointers.out | %filecheck_exec %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct myStruct
{
  double data;
  bool effect;
  myStruct(double x, bool eff) { 
    data = x;
    effect = eff;
  }
  ~myStruct() = default;
};

myStruct operator+(myStruct a, myStruct b){
  myStruct out(0, false);
  out.data = a.data + b.data;
  out.effect = a.effect || b.effect;
  return out;
}

myStruct updateIndexParamValue(myStruct arg,
                               std::size_t idx, std::size_t currIdx,
                               int multiplier, numerical_diff::precision &h_val,
                               std::size_t n = 0, std::size_t i = 0) {
  if (idx == currIdx) {
    h_val = (h_val == 0) ? numerical_diff::get_h(arg.data) : h_val;
    if (arg.effect)
      return myStruct(arg.data + h_val * multiplier, arg.effect);
  }
  return arg;
}

myStruct* updateIndexParamValue(myStruct* arg,
                               std::size_t idx, std::size_t currIdx,
                               int multiplier, numerical_diff::precision &h_val,
                               std::size_t n = 0, std::size_t i = 0) {
  myStruct* temp = numerical_diff::getBufferManager()
                       .make_buffer_space<myStruct>(1, true, arg->data,
                                                    arg->effect);
  if (idx == currIdx) {
    h_val = (h_val == 0) ? numerical_diff::get_h(arg->data) : h_val;
    if (arg->effect)
      temp->data = arg->data + h_val * multiplier;
  }
  return temp;
}

double func3(myStruct* a, myStruct b) { 
    a->data = a->data * b.data;
    a->data = a->data + a->data;
    return a->data;
}

int main(){ // expected-no-diagnostics
  myStruct a(3, true), b(5, true);
  double userDefined_res =
      numerical_diff::forward_central_difference(func3, &a, 0, 1, 0, false, &a, b);
  printf("Result is = %f\n", userDefined_res); // CHECK-EXEC: Result is = 10.000000 

}
