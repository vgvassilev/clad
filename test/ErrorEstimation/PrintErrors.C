// RUN: %cladclang %s -x c++ -lstdc++ -I%S/../../include -oPrintErrors.out 2>&1 | FileCheck %s
// RUN: ./PrintErrors.out | FileCheck -check-prefix=CHECK-EXEC %s

//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

#include <fstream> // Necessary for printing data to file.
#include <iostream> 

float func1(float x, float y) {
  x = x - y - y * y;
  float t = x * y * y;
  return t;
}

//CHECK: void func1_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error, {{.+}}ostream &_error_stream) {
//CHECK-NEXT:     double _delta_x = 0;
//CHECK-NEXT:     float _EERepl_x0 = x;
//CHECK-NEXT:     float _t0;
//CHECK-NEXT:     float _t1;
//CHECK-NEXT:     float _EERepl_x1;
//CHECK-NEXT:     float _t2;
//CHECK-NEXT:     float _t3;
//CHECK-NEXT:     float _t4;
//CHECK-NEXT:     float _t5;
//CHECK-NEXT:     float _d_t = 0;
//CHECK-NEXT:     double _delta_t = 0;
//CHECK-NEXT:     float _EERepl_t0;
//CHECK-NEXT:     _t1 = y;
//CHECK-NEXT:     _t0 = y;
//CHECK-NEXT:     x = x - y - _t1 * _t0;
//CHECK-NEXT:     _EERepl_x1 = x;
//CHECK-NEXT:     _t4 = x;
//CHECK-NEXT:     _t3 = y;
//CHECK-NEXT:     _t5 = _t4 * _t3;
//CHECK-NEXT:     _t2 = y;
//CHECK-NEXT:     float t = _t5 * _t2;
//CHECK-NEXT:     _EERepl_t0 = t;
//CHECK-NEXT:     float func1_return = t;
//CHECK-NEXT:     goto _label0;
//CHECK-NEXT:   _label0:
//CHECK-NEXT:     _d_t += 1;
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r2 = _d_t * _t2;
//CHECK-NEXT:         float _r3 = _r2 * _t3;
//CHECK-NEXT:         * _d_x += _r3;
//CHECK-NEXT:         float _r4 = _t4 * _r2;
//CHECK-NEXT:         * _d_y += _r4;
//CHECK-NEXT:         float _r5 = _t5 * _d_t;
//CHECK-NEXT:         * _d_y += _r5;
//CHECK-NEXT:         _delta_t += std::abs(_d_t * _EERepl_t0 * {{.+}});
//CHECK-NEXT:         _error_stream << "t" << " : " << std::abs(_d_t * _EERepl_t0 * {{.+}}) << "\n";
//CHECK-NEXT:     }
//CHECK-NEXT:     {
//CHECK-NEXT:         float _r_d0 = * _d_x;
//CHECK-NEXT:         * _d_x += _r_d0;
//CHECK-NEXT:         * _d_y += -_r_d0;
//CHECK-NEXT:         float _r0 = -_r_d0 * _t0;
//CHECK-NEXT:         * _d_y += _r0;
//CHECK-NEXT:         float _r1 = _t1 * -_r_d0;
//CHECK-NEXT:         * _d_y += _r1;
//CHECK-NEXT:         _delta_x += std::abs(_r_d0 * _EERepl_x1 * {{.+}});
//CHECK-NEXT:         _error_stream << "x" << " : " << std::abs(_r_d0 * _EERepl_x1 * {{.+}}) << "\n";
//CHECK-NEXT:         * _d_x -= _r_d0;
//CHECK-NEXT:         * _d_x;
//CHECK-NEXT:     }
//CHECK-NEXT:     _delta_x += std::abs(* _d_x * _EERepl_x0 * {{.+}});
//CHECK-NEXT:     _error_stream << "x" << " : " << std::abs(* _d_x * _EERepl_x0 * {{.+}}) << "\n";
//CHECK-NEXT:     double _delta_y = 0;
//CHECK-NEXT:     _delta_y += std::abs(* _d_y * y * {{.+}});
//CHECK-NEXT:     _error_stream << "y" << " : " << std::abs(* _d_y * y * {{.+}}) << "\n";
//CHECK-NEXT:     _final_error += _delta_{{x|y|t}} + _delta_{{x|y|t}} + _delta_{{x|y|t}};
//CHECK-NEXT:     _error_stream << "\nFinal error contribution by {{x|y|t}} = " << _delta_{{x|y|t}} << "\n";
//CHECK-NEXT:     _error_stream << "\nFinal error contribution by {{x|y|t}} = " << _delta_{{x|y|t}} << "\n";
//CHECK-NEXT:     _error_stream << "\nFinal error contribution by {{x|y|t}} = " << _delta_{{x|y|t}} << "\n";
//CHECK-NEXT: }

int main(){
  auto err_func = clad::estimate_error<true>(func1);
  float res[2] = {0};
  double x = 0.7779, y = 0.999999, err = 0;
  err_func.execute(x, y, &res[0], &res[1], err, std::cout);
  //CHECK-EXEC: t : {{.+}}
  //CHECK-EXEC: x : {{.+}}
  //CHECK-EXEC: x : {{.+}}
  //CHECK-EXEC: y : {{.+}}
  //CHECK-EXEC: Final error contribution by {{x|y|t}} = {{.+}}
  //CHECK-EXEC: Final error contribution by {{x|y|t}} = {{.+}}
  //CHECK-EXEC: Final error contribution by {{x|y|t}} = {{.+}}
}
