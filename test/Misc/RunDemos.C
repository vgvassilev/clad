// RUN: %cladclang %S/../../demos/BasicUsage.cpp -I%S/../../include 2>&1
// RUN: %cladclang %S/../../demos/ControlFlow.cpp -I%S/../../include 2>&1
// RUN: %cladclang %S/../../demos/DebuggingClad.cpp -I%S/../../include 2>&1
// RUN: %cladclang %S/../../demos/RosenbrockFunction.cpp -I%S/../../include 2>&1
// RUN: %cladclang %S/../../demos/ComputerGraphics/smallpt/SmallPT.cpp -I%S/../../include 2>&1


//-----------------------------------------------------------------------------/
//  Demo: Gradient.cpp
//-----------------------------------------------------------------------------/

// RUN: %cladclang %S/../../demos/Gradient.cpp -I%S/../../include -oGradient.out 2>&1 | FileCheck -check-prefix CHECK_GRADIENT %s
// CHECK_GRADIENT-NOT:{{.*error|warning|note:.*}}
// CHECK_GRADIENT:float sphere_implicit_func_darg0(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT: float _d_x = 1;
// CHECK_GRADIENT: float _d_y = 0;
// CHECK_GRADIENT: float _d_z = 0;
// CHECK_GRADIENT: float _d_xc = 0;
// CHECK_GRADIENT: float _d_yc = 0;
// CHECK_GRADIENT: float _d_zc = 0;
// CHECK_GRADIENT: float _d_r = 0;
// CHECK_GRADIENT: float _t0 = (x - xc);
// CHECK_GRADIENT: float _t1 = (x - xc);
// CHECK_GRADIENT: float _t2 = (y - yc);
// CHECK_GRADIENT: float _t3 = (y - yc);
// CHECK_GRADIENT: float _t4 = (z - zc);
// CHECK_GRADIENT: float _t5 = (z - zc);
// CHECK_GRADIENT: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_darg1(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT: float _d_x = 0;
// CHECK_GRADIENT: float _d_y = 1;
// CHECK_GRADIENT: float _d_z = 0;
// CHECK_GRADIENT: float _d_xc = 0;
// CHECK_GRADIENT: float _d_yc = 0;
// CHECK_GRADIENT: float _d_zc = 0;
// CHECK_GRADIENT: float _d_r = 0;
// CHECK_GRADIENT: float _t0 = (x - xc);
// CHECK_GRADIENT: float _t1 = (x - xc);
// CHECK_GRADIENT: float _t2 = (y - yc);
// CHECK_GRADIENT: float _t3 = (y - yc);
// CHECK_GRADIENT: float _t4 = (z - zc);
// CHECK_GRADIENT: float _t5 = (z - zc);
// CHECK_GRADIENT: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK_GRADIENT:}

// CHECK_GRADIENT:float sphere_implicit_func_darg2(float x, float y, float z, float xc, float yc, float zc, float r) {
// CHECK_GRADIENT: float _d_x = 0;
// CHECK_GRADIENT: float _d_y = 0;
// CHECK_GRADIENT: float _d_z = 1;
// CHECK_GRADIENT: float _d_xc = 0;
// CHECK_GRADIENT: float _d_yc = 0;
// CHECK_GRADIENT: float _d_zc = 0;
// CHECK_GRADIENT: float _d_r = 0;
// CHECK_GRADIENT: float _t0 = (x - xc);
// CHECK_GRADIENT: float _t1 = (x - xc);
// CHECK_GRADIENT: float _t2 = (y - yc);
// CHECK_GRADIENT: float _t3 = (y - yc);
// CHECK_GRADIENT: float _t4 = (z - zc);
// CHECK_GRADIENT: float _t5 = (z - zc);
// CHECK_GRADIENT: return (_d_x - _d_xc) * _t1 + _t0 * (_d_x - _d_xc) + (_d_y - _d_yc) * _t3 + _t2 * (_d_y - _d_yc) + (_d_z - _d_zc) * _t5 + _t4 * (_d_z - _d_zc) - (_d_r * r + r * _d_r);
// CHECK_GRADIENT:}

// RUN: ./Gradient.out | FileCheck -check-prefix CHECK_GRADIENT_EXEC %s
// CHECK_GRADIENT_EXEC: Result is N=(10.000000,0.000000,0.000000)

//-----------------------------------------------------------------------------/
// Demo: Rosenbrock Function
//-----------------------------------------------------------------------------/
// RUN: %cladclang %S/../../demos/RosenbrockFunction.cpp -I%S/../../include -oRosenbrockFunction.out 2>&1 | FileCheck -check-prefix CHECK_ROSENBROCK %s
// CHECK_ROSENBROCK-NOT:{{.*error|warning|note:.*}}
// CHECK_ROSENBROCK:double rosenbrock_func_darg0(double x, double y) {
// CHECK_ROSENBROCK: double _d_x = 1;
// CHECK_ROSENBROCK: double _d_y = 0;
// CHECK_ROSENBROCK: double _t0 = (x - 1);
// CHECK_ROSENBROCK: double _t1 = (x - 1);
// CHECK_ROSENBROCK: double _t2 = (y - x * x);
// CHECK_ROSENBROCK: double _t3 = 100 * _t2;
// CHECK_ROSENBROCK: double _t4 = (y - x * x);
// CHECK_ROSENBROCK: return (_d_x - 0) * _t1 + _t0 * (_d_x - 0) + (0 * _t2 + 100 * (_d_y - (_d_x * x + x * _d_x))) * _t4 + _t3 * (_d_y - (_d_x * x + x * _d_x));
// CHECK_ROSENBROCK:}
// CHECK_ROSENBROCK:double rosenbrock_func_darg1(double x, double y) {
// CHECK_ROSENBROCK: double _d_x = 0;
// CHECK_ROSENBROCK: double _d_y = 1;
// CHECK_ROSENBROCK: double _t0 = (x - 1);
// CHECK_ROSENBROCK: double _t1 = (x - 1);
// CHECK_ROSENBROCK: double _t2 = (y - x * x);
// CHECK_ROSENBROCK: double _t3 = 100 * _t2;
// CHECK_ROSENBROCK: double _t4 = (y - x * x);
// CHECK_ROSENBROCK: return (_d_x - 0) * _t1 + _t0 * (_d_x - 0) + (0 * _t2 + 100 * (_d_y - (_d_x * x + x * _d_x))) * _t4 + _t3 * (_d_y - (_d_x * x + x * _d_x));
// CHECK_ROSENBROCK:}
// RUN: ./RosenbrockFunction.out | FileCheck -check-prefix CHECK_ROSENBROCK_EXEC %s
// CHECK_ROSENBROCK_EXEC: The result is -899.000000.

//-----------------------------------------------------------------------------/
// Demo: ODE Solver Sensitivity
//-----------------------------------------------------------------------------/
// RUN: %cladclang %S/../../demos/ODESolverSensitivity.cpp -I%S/../../include -oODESolverSensitivity.out

//-----------------------------------------------------------------------------/
// Demo: Error Estimation Float Sum
//-----------------------------------------------------------------------------/

// RUN: %cladclang %S/../../demos/ErrorEstimation/FloatSum.cpp -I%S/../../include 2>&1  | FileCheck -check-prefix CHECK_FLOAT_SUM %s
//CHECK_FLOAT_SUM-NOT: {{.*error|warning|note:.*}}

//CHECK_FLOAT_SUM: void vanillaSum_grad(float x, unsigned int n, clad::array_ref<float> _d_x, clad::array_ref<unsigned int> _d_n, double &_final_error) {
//CHECK_FLOAT_SUM:     float _d_sum = 0;
//CHECK_FLOAT_SUM:     double _delta_sum = 0;
//CHECK_FLOAT_SUM:     float _EERepl_sum0;
//CHECK_FLOAT_SUM:     unsigned long _t0;
//CHECK_FLOAT_SUM:     unsigned int _d_i = 0;
//CHECK_FLOAT_SUM:     clad::tape<float> _EERepl_sum1 = {};
//CHECK_FLOAT_SUM:     float sum = 0.;
//CHECK_FLOAT_SUM:     _EERepl_sum0 = sum;
//CHECK_FLOAT_SUM:     _t0 = 0;
//CHECK_FLOAT_SUM:     for (unsigned int i = 0; i < n; i++) {
//CHECK_FLOAT_SUM:         _t0++;
//CHECK_FLOAT_SUM:         sum = sum + x;
//CHECK_FLOAT_SUM:         clad::push(_EERepl_sum1, sum);
//CHECK_FLOAT_SUM:     }
//CHECK_FLOAT_SUM:     goto _label0;
//CHECK_FLOAT_SUM:   _label0:
//CHECK_FLOAT_SUM:     _d_sum += 1;
//CHECK_FLOAT_SUM:     for (; _t0; _t0--) {
//CHECK_FLOAT_SUM:         {
//CHECK_FLOAT_SUM:             float _r_d0 = _d_sum;
//CHECK_FLOAT_SUM:             _d_sum += _r_d0;
//CHECK_FLOAT_SUM:             * _d_x += _r_d0;
//CHECK_FLOAT_SUM:             float _r0 = clad::pop(_EERepl_sum1);
//CHECK_FLOAT_SUM:             _delta_sum += std::abs(_r_d0 * _r0 * {{.+}});
//CHECK_FLOAT_SUM:             _d_sum -= _r_d0;
//CHECK_FLOAT_SUM:         }
//CHECK_FLOAT_SUM:     }
//CHECK_FLOAT_SUM:     _delta_sum += std::abs(_d_sum * _EERepl_sum0 * {{.+}});
//CHECK_FLOAT_SUM:     double _delta_x = 0;
//CHECK_FLOAT_SUM:     _delta_x += std::abs(* _d_x * x * {{.+}});
//CHECK_FLOAT_SUM:     _final_error += _delta_{{x|sum}} + _delta_{{x|sum}};
//CHECK_FLOAT_SUM: }

//-----------------------------------------------------------------------------/
// Demo: Custom Error Estimation Plugin
//-----------------------------------------------------------------------------/
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fcustom-estimation-model \
// RUN:  -Xclang -plugin-arg-clad -Xclang %clad_obj_root/demos/ErrorEstimation/CustomModel/libcladCustomModelPlugin%shlibext \
// RUN:   %S/../../demos/ErrorEstimation/CustomModel/test.cpp \
// RUN: -I%S/../../include -oCustomModelTest.out | FileCheck -check-prefix CHECK_CUSTOM_MODEL %s

// CHECK_CUSTOM_MODEL-NOT: Could not load {{.*}}cladCustomModelPlugin{{.*}}

// RUN: ./CustomModelTest.out | FileCheck -check-prefix CHECK_CUSTOM_MODEL_EXEC %s
// CHECK_CUSTOM_MODEL_EXEC-NOT:{{.*error|warning|note:.*}}
// CHECK_CUSTOM_MODEL_EXEC: The code is:
// CHECK_CUSTOM_MODEL_EXEC-NEXT: void func_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    float _d_z = 0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    double _delta_z = 0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    float _EERepl_z0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    float _EERepl_z1;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    float z;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    _EERepl_z0 = z;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    z = x + y;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    _EERepl_z1 = z;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    goto _label0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:  _label0:
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    _d_z += 1;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    {
// CHECK_CUSTOM_MODEL_EXEC-NEXT:        float _r_d0 = _d_z;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:        * _d_x += _r_d0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:        * _d_y += _r_d0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:        _delta_z += _r_d0 * _EERepl_z1;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:        _d_z -= _r_d0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    }
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    double _delta_x = 0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    _delta_x += * _d_x * x;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    double _delta_y = 0;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    _delta_y += * _d_y * y;
// CHECK_CUSTOM_MODEL_EXEC-NEXT:    _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
// CHECK_CUSTOM_MODEL_EXEC-NEXT: }

//-----------------------------------------------------------------------------/
// Demo: Print Error Estimation Plugin
//-----------------------------------------------------------------------------/
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -fcustom-estimation-model \
// RUN:  -Xclang -plugin-arg-clad -Xclang %clad_obj_root/demos/ErrorEstimation/PrintModel/libcladPrintModelPlugin%shlibext \
// RUN:   %S/../../demos/ErrorEstimation/PrintModel/test.cpp \
// RUN: -I%S/../../include -oPrintModelTest.out | FileCheck -check-prefix CHECK_PRINT_MODEL %s

// CHECK_PRINT_MODEL-NOT: Could not load {{.*}}cladPrintModelPlugin{{.*}}

// RUN: ./PrintModelTest.out | FileCheck -check-prefix CHECK_PRINT_MODEL_EXEC %s
// CHECK_PRINT_MODEL_EXEC-NOT:{{.*error|warning|note:.*}}
// CHECK_PRINT_MODEL_EXEC: The code is:
// CHECK_PRINT_MODEL_EXEC-NEXT: void func_grad(float x, float y, clad::array_ref<float> _d_x, clad::array_ref<float> _d_y, double &_final_error) {
// CHECK_PRINT_MODEL_EXEC-NEXT:    float _d_z = 0;
// CHECK_PRINT_MODEL_EXEC-NEXT:    double _delta_z = 0;
// CHECK_PRINT_MODEL_EXEC-NEXT:    float _EERepl_z0;
// CHECK_PRINT_MODEL_EXEC-NEXT:    float _EERepl_z1;
// CHECK_PRINT_MODEL_EXEC-NEXT:    float z;
// CHECK_PRINT_MODEL_EXEC-NEXT:    _EERepl_z0 = z;
// CHECK_PRINT_MODEL_EXEC-NEXT:    z = x + y;
// CHECK_PRINT_MODEL_EXEC-NEXT:    _EERepl_z1 = z;
// CHECK_PRINT_MODEL_EXEC-NEXT:    goto _label0;
// CHECK_PRINT_MODEL_EXEC-NEXT:  _label0:
// CHECK_PRINT_MODEL_EXEC-NEXT:    _d_z += 1;
// CHECK_PRINT_MODEL_EXEC-NEXT:    {
// CHECK_PRINT_MODEL_EXEC-NEXT:        float _r_d0 = _d_z;
// CHECK_PRINT_MODEL_EXEC-NEXT:        * _d_x += _r_d0;
// CHECK_PRINT_MODEL_EXEC-NEXT:        * _d_y += _r_d0;
// CHECK_PRINT_MODEL_EXEC-NEXT:        _delta_z += clad::getErrorVal(_r_d0, _EERepl_z1, "z");
// CHECK_PRINT_MODEL_EXEC-NEXT:        _d_z -= _r_d0;
// CHECK_PRINT_MODEL_EXEC-NEXT:    }
// CHECK_PRINT_MODEL_EXEC-NEXT:    double _delta_x = 0;
// CHECK_PRINT_MODEL_EXEC-NEXT:    _delta_x += clad::getErrorVal(* _d_x, x, "x");
// CHECK_PRINT_MODEL_EXEC-NEXT:    double _delta_y = 0;
// CHECK_PRINT_MODEL_EXEC-NEXT:    _delta_y += clad::getErrorVal(* _d_y, y, "y");
// CHECK_PRINT_MODEL_EXEC-NEXT:    _final_error += _delta_{{x|y|z}} + _delta_{{x|y|z}} + _delta_{{x|y|z}};
// CHECK_PRINT_MODEL_EXEC-NEXT: }
// CHECK_PRINT_MODEL_EXEC: Error in z : {{.+}}
// CHECK_PRINT_MODEL_EXEC-NEXT: Error in x : {{.+}}
// CHECK_PRINT_MODEL_EXEC-NEXT: Error in y : {{.+}}

//-----------------------------------------------------------------------------/
// Demo: Gradient Descent
//-----------------------------------------------------------------------------/
// RUN: %cladclang %S/../../demos/GradientDescent.cpp -I%S/../../include -oGradientDescent.out | FileCheck -check-prefix CHECK_GRADIENT_DESCENT %s

//CHECK_GRADIENT_DESCENT: void f_pullback(double theta_0, double theta_1, double x, double _d_y, clad::array_ref<double> _d_theta_0, clad::array_ref<double> _d_theta_1, clad::array_ref<double> _d_x) {
//CHECK_GRADIENT_DESCENT-NEXT:     double _t0;
//CHECK_GRADIENT_DESCENT-NEXT:     double _t1;
//CHECK_GRADIENT_DESCENT-NEXT:     _t1 = theta_1;
//CHECK_GRADIENT_DESCENT-NEXT:     _t0 = x;
//CHECK_GRADIENT_DESCENT-NEXT:     goto _label0;
//CHECK_GRADIENT_DESCENT-NEXT:   _label0:
//CHECK_GRADIENT_DESCENT-NEXT:     {
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_theta_0 += _d_y;
//CHECK_GRADIENT_DESCENT-NEXT:         double _r0 = _d_y * _t0;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_theta_1 += _r0;
//CHECK_GRADIENT_DESCENT-NEXT:         double _r1 = _t1 * _d_y;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_x += _r1;
//CHECK_GRADIENT_DESCENT-NEXT:     }
//CHECK_GRADIENT_DESCENT-NEXT: }

//CHECK_GRADIENT_DESCENT-NEXT: void cost_grad(double theta_0, double theta_1, double x, double y, clad::array_ref<double> _d_theta_0, clad::array_ref<double> _d_theta_1, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y) {
//CHECK_GRADIENT_DESCENT-NEXT:     double _t0;
//CHECK_GRADIENT_DESCENT-NEXT:     double _t1;
//CHECK_GRADIENT_DESCENT-NEXT:     double _t2;
//CHECK_GRADIENT_DESCENT-NEXT:     double _d_f_x = 0;
//CHECK_GRADIENT_DESCENT-NEXT:     double _t3;
//CHECK_GRADIENT_DESCENT-NEXT:     double _t4;
//CHECK_GRADIENT_DESCENT-NEXT:     _t0 = theta_0;
//CHECK_GRADIENT_DESCENT-NEXT:     _t1 = theta_1;
//CHECK_GRADIENT_DESCENT-NEXT:     _t2 = x;
//CHECK_GRADIENT_DESCENT-NEXT:     double f_x = f(_t0, _t1, _t2);
//CHECK_GRADIENT_DESCENT-NEXT:     _t4 = (f_x - y);
//CHECK_GRADIENT_DESCENT-NEXT:     _t3 = (f_x - y);
//CHECK_GRADIENT_DESCENT-NEXT:     goto _label0;
//CHECK_GRADIENT_DESCENT-NEXT:   _label0:
//CHECK_GRADIENT_DESCENT-NEXT:     {
//CHECK_GRADIENT_DESCENT-NEXT:         double _r3 = 1 * _t3;
//CHECK_GRADIENT_DESCENT-NEXT:         _d_f_x += _r3;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_y += -_r3;
//CHECK_GRADIENT_DESCENT-NEXT:         double _r4 = _t4 * 1;
//CHECK_GRADIENT_DESCENT-NEXT:         _d_f_x += _r4;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_y += -_r4;
//CHECK_GRADIENT_DESCENT-NEXT:     }
//CHECK_GRADIENT_DESCENT-NEXT:     {
//CHECK_GRADIENT_DESCENT-NEXT:         double _grad0 = 0.;
//CHECK_GRADIENT_DESCENT-NEXT:         double _grad1 = 0.;
//CHECK_GRADIENT_DESCENT-NEXT:         double _grad2 = 0.;
//CHECK_GRADIENT_DESCENT-NEXT:         f_pullback(_t0, _t1, _t2, _d_f_x, &_grad0, &_grad1, &_grad2);
//CHECK_GRADIENT_DESCENT-NEXT:         double _r0 = _grad0;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_theta_0 += _r0;
//CHECK_GRADIENT_DESCENT-NEXT:         double _r1 = _grad1;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_theta_1 += _r1;
//CHECK_GRADIENT_DESCENT-NEXT:         double _r2 = _grad2;
//CHECK_GRADIENT_DESCENT-NEXT:         * _d_x += _r2;
//CHECK_GRADIENT_DESCENT-NEXT:     }
//CHECK_GRADIENT_DESCENT-NEXT: }

//-----------------------------------------------------------------------------/
// Demo: Custom Type Numerical Diff
//-----------------------------------------------------------------------------/
// RUN: %cladnumdiffclang %S/../../demos/CustomTypeNumDiff.cpp -I%S/../../include -oCustomTypeNumDiff.out
// RUN: ./CustomTypeNumDiff.out | FileCheck -check-prefix CHECK_CUSTOM_NUM_DIFF_EXEC %s
// CHECK_CUSTOM_NUM_DIFF_EXEC: Result of df/dx is = 0.07
// CHECK_CUSTOM_NUM_DIFF_EXEC: Result of df/dx is = 0.003

//-----------------------------------------------------------------------------/
// Demo: Arrays.cpp
//-----------------------------------------------------------------------------/
// RUN: %cladclang %S/../../demos/Arrays.cpp -I%S/../../include -oArrays.out 2>&1
// RUN: ./Arrays.out | FileCheck -check-prefix CHECK_ARRAYS_EXEC %s
// CHECK_ARRAYS_EXEC: Forward Mode w.r.t. arr:
// CHECK_ARRAYS_EXEC:  res_arr = 0.17, 0.2, 0.1
// CHECK_ARRAYS_EXEC: Reverse Mode w.r.t. all:
// CHECK_ARRAYS_EXEC:  darr = {0.17, 0.2, 0.1}
// CHECK_ARRAYS_EXEC:  dweights = {0.33, 0.67, 1}
// CHECK_ARRAYS_EXEC: Reverse Mode w.r.t. arr:
// CHECK_ARRAYS_EXEC:  darr = {0.17, 0.2, 0.1}
// CHECK_ARRAYS_EXEC: Hessian Mode w.r.t. to all:
// CHECK_ARRAYS_EXEC:  matrix =
// CHECK_ARRAYS_EXEC:   {0, 0, 0, 0.33, 0, 0}
// CHECK_ARRAYS_EXEC:   {0, 0, 0, 0, 0.33, 0}
// CHECK_ARRAYS_EXEC:   {0, 0, 0, 0, 0, 0.33}
// CHECK_ARRAYS_EXEC:   {0.33, 0, 0, 0, 0, 0}
// CHECK_ARRAYS_EXEC:   {0, 0.33, 0, 0, 0, 0}
// CHECK_ARRAYS_EXEC:   {0, 0, 0.33, 0, 0, 0}
// CHECK_ARRAYS_EXEC: Hessian Mode w.r.t. to arr:
// CHECK_ARRAYS_EXEC:  matrix =
// CHECK_ARRAYS_EXEC:   {0, 0, 0}
// CHECK_ARRAYS_EXEC:   {0, 0, 0}
// CHECK_ARRAYS_EXEC:   {0, 0, 0}
