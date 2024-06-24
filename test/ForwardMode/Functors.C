// RUN: %cladclang %s -I%S/../../include -oFunctors.out 2>&1 | %filecheck %s
// RUN: ./Functors.out | %filecheck_exec %s
// CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"

struct Experiment {
  mutable double x, y;
  Experiment() : x(0), y(0) {}
  Experiment(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) {
    return x*i*j;
  }
  void setX(double val) {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     Experiment _d_this_obj;
  // CHECK-NEXT:     Experiment *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _d_x = 0;
  // CHECK-NEXT:     double _d_y = 0;
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentConst {
  mutable double x, y;
  ExperimentConst() : x(0), y(0) {}
  ExperimentConst(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const {
    return x*i*j;
  }
  void setX(double val) const {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const ExperimentConst _d_this_obj;
  // CHECK-NEXT:     const ExperimentConst *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _d_x = 0;
  // CHECK-NEXT:     double _d_y = 0;
  // CHECK-NEXT:     double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentVolatile {
  mutable double x, y;
  ExperimentVolatile() : x(0), y(0) {}
  ExperimentVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) volatile {
    return x*i*j;
  }
  void setX(double val) volatile {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     volatile ExperimentVolatile _d_this_obj;
  // CHECK-NEXT:     volatile ExperimentVolatile *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _d_x = 0;
  // CHECK-NEXT:     double _d_y = 0;
  // CHECK-NEXT:     volatile double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

struct ExperimentConstVolatile {
  mutable double x, y;
  ExperimentConstVolatile() : x(0), y(0) {}
  ExperimentConstVolatile(double p_x, double p_y) : x(p_x), y(p_y) {}
  double operator()(double i, double j) const volatile {
    return x*i*j;
  }
  void setX(double val) const volatile {
    x = val;
  }

  // CHECK: double operator_call_darg0(double i, double j) const volatile {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     const volatile ExperimentConstVolatile _d_this_obj;
  // CHECK-NEXT:     const volatile ExperimentConstVolatile *_d_this = &_d_this_obj;
  // CHECK-NEXT:     double _d_x = 0;
  // CHECK-NEXT:     double _d_y = 0;
  // CHECK-NEXT:     volatile double &_t0 = this->x;
  // CHECK-NEXT:     double _t1 = _t0 * i;
  // CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * j + _t1 * _d_j;
  // CHECK-NEXT: }
};

namespace outer {
  namespace inner {
    struct ExperimentNNS {
      mutable double x, y;
      ExperimentNNS() : x(0), y(0) {}
      ExperimentNNS(double p_x, double p_y) : x(p_x), y(p_y) {}
      double operator()(double i, double j) {
        return x*i*j;
      }
      void setX(double val) {
        x = val;
      }

      // CHECK: double operator_call_darg0(double i, double j) {
      // CHECK-NEXT:     double _d_i = 1;
      // CHECK-NEXT:     double _d_j = 0;
      // CHECK-NEXT:     outer::inner::ExperimentNNS _d_this_obj;
      // CHECK-NEXT:     outer::inner::ExperimentNNS *_d_this = &_d_this_obj;
      // CHECK-NEXT:     double _d_x = 0;
      // CHECK-NEXT:     double _d_y = 0;
      // CHECK-NEXT:     double &_t0 = this->x;
      // CHECK-NEXT:     double _t1 = _t0 * i;
      // CHECK-NEXT:     return (_d_x * i + _t0 * _d_i) * j + _t1 * _d_j;
      // CHECK-NEXT: }   
    };

    auto lambdaNNS = [](double i, double j) {
      return i*i*j;
    };
  }
}

struct Widget {
  double i, j;
  const char* char_arr[10];
  Widget() : i(0), j(0) {}
  Widget(double p_i, double p_j) : i(p_i), j(p_j) {}
  double operator()() {
    j = i * i;
    j /= i;
    return i*i + j;
  }

  // CHECK:   double operator_call_darg0() {
  // CHECK-NEXT:       Widget _d_this_obj;
  // CHECK-NEXT:       Widget *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double _d_i = 1;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double &_t0 = this->i;
  // CHECK-NEXT:       double &_t1 = this->i;
  // CHECK-NEXT:       _d_j = _d_i * _t1 + _t0 * _d_i;
  // CHECK-NEXT:       this->j = _t0 * _t1;
  // CHECK-NEXT:       double &_t2 = this->j;
  // CHECK-NEXT:       double &_t3 = this->i;
  // CHECK-NEXT:       _d_j = (_d_j * _t3 - _t2 * _d_i) / (_t3 * _t3);
  // CHECK-NEXT:       _t2 /= _t3;
  // CHECK-NEXT:       double &_t4 = this->i;
  // CHECK-NEXT:       double &_t5 = this->i;
  // CHECK-NEXT:       return _d_i * _t5 + _t4 * _d_i + _d_j;
  // CHECK-NEXT:   }

  void setI(double new_i) {
    i = new_i;
  }

  void setJ(double new_j) {
    j = new_j;
  }
};

struct WidgetConstVolatile {
  mutable double i, j;
  char** s;
  char* char_arr[10];
  WidgetConstVolatile() : i(0), j(0) {}
  WidgetConstVolatile(double p_i, double p_j) : i(p_i), j(p_j) {}
  double operator()() const volatile {
    j = i * i;
    j /= i;
    return i*i + j;
  }

  // CHECK:   double operator_call_darg0() const volatile {
  // CHECK-NEXT:       const volatile WidgetConstVolatile _d_this_obj;
  // CHECK-NEXT:       const volatile WidgetConstVolatile *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double _d_i = 1;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       volatile double &_t0 = this->i;
  // CHECK-NEXT:       volatile double &_t1 = this->i;
  // CHECK-NEXT:       _d_j = _d_i * _t1 + _t0 * _d_i;
  // CHECK-NEXT:       this->j = _t0 * _t1;
  // CHECK-NEXT:       volatile double &_t2 = this->j;
  // CHECK-NEXT:       volatile double &_t3 = this->i;
  // CHECK-NEXT:       _d_j = (_d_j * _t3 - _t2 * _d_i) / (_t3 * _t3);
  // CHECK-NEXT:       _t2 /= _t3;
  // CHECK-NEXT:       volatile double &_t4 = this->i;
  // CHECK-NEXT:       volatile double &_t5 = this->i;
  // CHECK-NEXT:       return _d_i * _t5 + _t4 * _d_i + _d_j;
  // CHECK-NEXT:   }

  void setI(double new_i) {
    i = new_i;
  }

  void setJ(double new_j) {
    j = new_j;
  }
};

struct WidgetArr {
  mutable double i, j;
  double arr[10];
  WidgetArr() : i(0), j(0), arr{} {}
  WidgetArr(double p_i, double p_j) : i(p_i), j(p_j) {
    for (int i=0; i<10; ++i)
      arr[i] = i;
  }
  double operator()() {
    double temp=0;
    for (int k=0; k<10; ++k) {
      temp += arr[k];
    }
    i *= arr[3];
    j *= arr[5];

    return i + j + temp;
  }

  // CHECK:   double operator_call_darg2_3() {
  // CHECK-NEXT:       WidgetArr _d_this_obj;
  // CHECK-NEXT:       WidgetArr *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double _d_i = 0;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double _d_arr[10] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0};
  // CHECK-NEXT:       double _d_temp = 0;
  // CHECK-NEXT:       double temp = 0;
  // CHECK-NEXT:       {
  // CHECK-NEXT:           int _d_k = 0;
  // CHECK-NEXT:           for (int k = 0; k < 10; ++k) {
  // CHECK-NEXT:               _d_temp += _d_arr[k];
  // CHECK-NEXT:               temp += this->arr[k];
  // CHECK-NEXT:           }
  // CHECK-NEXT:       }
  // CHECK-NEXT:       double &_t0 = this->i;
  // CHECK-NEXT:       double &_t1 = this->arr[3];
  // CHECK-NEXT:       _d_i = _d_i * _t1 + _t0 * _d_arr[3];
  // CHECK-NEXT:       _t0 *= _t1;
  // CHECK-NEXT:       double &_t2 = this->j;
  // CHECK-NEXT:       double &_t3 = this->arr[5];
  // CHECK-NEXT:       _d_j = _d_j * _t3 + _t2 * _d_arr[5];
  // CHECK-NEXT:       _t2 *= _t3;
  // CHECK-NEXT:       return _d_i + _d_j + _d_temp;
  // CHECK-NEXT:   }

  // CHECK:   double operator_call_darg2_5() {
  // CHECK-NEXT:       WidgetArr _d_this_obj;
  // CHECK-NEXT:       WidgetArr *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double _d_i = 0;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double _d_arr[10] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0};
  // CHECK-NEXT:       double _d_temp = 0;
  // CHECK-NEXT:       double temp = 0;
  // CHECK-NEXT:       {
  // CHECK-NEXT:           int _d_k = 0;
  // CHECK-NEXT:           for (int k = 0; k < 10; ++k) {
  // CHECK-NEXT:               _d_temp += _d_arr[k];
  // CHECK-NEXT:               temp += this->arr[k];
  // CHECK-NEXT:           }
  // CHECK-NEXT:       }
  // CHECK-NEXT:       double &_t0 = this->i;
  // CHECK-NEXT:       double &_t1 = this->arr[3];
  // CHECK-NEXT:       _d_i = _d_i * _t1 + _t0 * _d_arr[3];
  // CHECK-NEXT:       _t0 *= _t1;
  // CHECK-NEXT:       double &_t2 = this->j;
  // CHECK-NEXT:       double &_t3 = this->arr[5];
  // CHECK-NEXT:       _d_j = _d_j * _t3 + _t2 * _d_arr[5];
  // CHECK-NEXT:       _t2 *= _t3;
  // CHECK-NEXT:       return _d_i + _d_j + _d_temp;
  // CHECK-NEXT:   }

  void setI(double new_i) {
    i = new_i;
  }

  void setJ(double new_j) {
    j = new_j;
  }
};

struct WidgetPointer {
  mutable double i, j;
  double* arr;
  WidgetPointer() : i(0), j(0) {}
  WidgetPointer(double p_i, double p_j) : i(p_i), j(p_j) {
    arr = static_cast<double*>(malloc(sizeof(double)*10));
    for (int i=0; i<10; ++i) {
      arr[i] = i;
    }
  }
  // ~WidgetPointer() {
  //   free(arr);
  // }
  double operator()() {
    double temp=0;
    for (int k=0; k<10; ++k) {
      temp += arr[k];
    }
    i *= arr[3]*arr[3];
    j *= arr[5]*arr[5];

    return i + j + temp;
  }

  // CHECK:   double operator_call_darg2_3() {
  // CHECK-NEXT:       WidgetPointer _d_this_obj;
  // CHECK-NEXT:       WidgetPointer *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double _d_i = 0;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double *_d_arr = nullptr;
  // CHECK-NEXT:       double _d_temp = 0;
  // CHECK-NEXT:       double temp = 0;
  // CHECK-NEXT:       {
  // CHECK-NEXT:           int _d_k = 0;
  // CHECK-NEXT:           for (int k = 0; k < 10; ++k) {
  // CHECK-NEXT:               _d_temp += (k == 3.);
  // CHECK-NEXT:               temp += this->arr[k];
  // CHECK-NEXT:           }
  // CHECK-NEXT:       }
  // CHECK-NEXT:       double &_t0 = this->arr[3];
  // CHECK-NEXT:       double &_t1 = this->arr[3];
  // CHECK-NEXT:       double &_t2 = this->i;
  // CHECK-NEXT:       double _t3 = 1. * _t1 + _t0 * 1.;
  // CHECK-NEXT:       double _t4 = _t0 * _t1;
  // CHECK-NEXT:       _d_i = _d_i * _t4 + _t2 * _t3;
  // CHECK-NEXT:       _t2 *= _t4;
  // CHECK-NEXT:       double &_t5 = this->arr[5];
  // CHECK-NEXT:       double &_t6 = this->arr[5];
  // CHECK-NEXT:       double &_t7 = this->j;
  // CHECK-NEXT:       double _t8 = 0 * _t6 + _t5 * 0;
  // CHECK-NEXT:       double _t9 = _t5 * _t6;
  // CHECK-NEXT:       _d_j = _d_j * _t9 + _t7 * _t8;
  // CHECK-NEXT:       _t7 *= _t9;
  // CHECK-NEXT:       return _d_i + _d_j + _d_temp;
  // CHECK-NEXT:   }

  // CHECK:   double operator_call_darg2_5() {
  // CHECK-NEXT:       WidgetPointer _d_this_obj;
  // CHECK-NEXT:       WidgetPointer *_d_this = &_d_this_obj;
  // CHECK-NEXT:       double _d_i = 0;
  // CHECK-NEXT:       double _d_j = 0;
  // CHECK-NEXT:       double *_d_arr = nullptr;
  // CHECK-NEXT:       double _d_temp = 0;
  // CHECK-NEXT:       double temp = 0;
  // CHECK-NEXT:       {
  // CHECK-NEXT:           int _d_k = 0;
  // CHECK-NEXT:           for (int k = 0; k < 10; ++k) {
  // CHECK-NEXT:               _d_temp += (k == 5.);
  // CHECK-NEXT:               temp += this->arr[k];
  // CHECK-NEXT:           }
  // CHECK-NEXT:       }
  // CHECK-NEXT:       double &_t0 = this->arr[3];
  // CHECK-NEXT:       double &_t1 = this->arr[3];
  // CHECK-NEXT:       double &_t2 = this->i;
  // CHECK-NEXT:       double _t3 = 0 * _t1 + _t0 * 0;
  // CHECK-NEXT:       double _t4 = _t0 * _t1;
  // CHECK-NEXT:       _d_i = _d_i * _t4 + _t2 * _t3;
  // CHECK-NEXT:       _t2 *= _t4;
  // CHECK-NEXT:       double &_t5 = this->arr[5];
  // CHECK-NEXT:       double &_t6 = this->arr[5];
  // CHECK-NEXT:       double &_t7 = this->j;
  // CHECK-NEXT:       double _t8 = 1. * _t6 + _t5 * 1.;
  // CHECK-NEXT:       double _t9 = _t5 * _t6;
  // CHECK-NEXT:       _d_j = _d_j * _t9 + _t7 * _t8;
  // CHECK-NEXT:       _t7 *= _t9;
  // CHECK-NEXT:       return _d_i + _d_j + _d_temp;
  // CHECK-NEXT:   }

  void setI(double new_i) {
    i = new_i;
  }

  void setJ(double new_j) {
    j = new_j;
  }
};

#define INIT(E, ARG)\
auto d_##E = clad::differentiate(&E, ARG);\
auto d_##E##Ref = clad::differentiate(E, ARG);

#define TEST(E)\
printf("%.2f %.2f\n", d_##E.execute(7, 9), d_##E##Ref.execute(7, 9));

#define TEST_2(W, I, J)             \
  W.setI(I);                        \
  W.setJ(J);                        \
  printf("%.2f ", d_##W.execute()); \
  W.setI(I);                        \
  W.setJ(J);                        \
  printf("%.2f\n", d_##W##Ref.execute());

double x=3;

int main() {
  Experiment E(3, 5);
  auto E_Again = E;
  const ExperimentConst E_Const(3, 5);
  volatile ExperimentVolatile E_Volatile(3, 5);
  const volatile ExperimentConstVolatile E_ConstVolatile(3, 5);
  outer::inner::ExperimentNNS E_NNS(3, 5);
  auto E_NNS_Again = E_NNS;
  auto lambda = [](double i, double j) {
    return i*i*j;
  };

  // CHECK: inline double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double _t0 = i * i;
  // CHECK-NEXT:     return (_d_i * i + i * _d_i) * j + _t0 * _d_j;
  // CHECK-NEXT: }

  auto lambdaWithCapture = [&](double i, double jj) {
    return x*i*jj;
  };

  // CHECK: inline double operator_call_darg0(double i, double jj) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_jj = 0;
  // CHECK-NEXT:     double _t0 = x * i;
  // CHECK-NEXT:     return (0 * i + x * _d_i) * jj + _t0 * _d_jj;
  // CHECK-NEXT: }

  auto lambdaNNS = outer::inner::lambdaNNS;

  // CHECK: inline double operator_call_darg0(double i, double j) const {
  // CHECK-NEXT:     double _d_i = 1;
  // CHECK-NEXT:     double _d_j = 0;
  // CHECK-NEXT:     double _t0 = i * i;
  // CHECK-NEXT:     return (_d_i * i + i * _d_i) * j + _t0 * _d_j;
  // CHECK-NEXT: }

  Widget W(3, 5);
  WidgetConstVolatile W_ConstVolatile(3, 5);
  WidgetArr W_Arr_3(3, 5), W_Arr_5(3, 5);
  WidgetPointer W_Pointer_3(3, 5), W_Pointer_5(3, 5);

  INIT(E, "i");
  INIT(E_Again, "i");
  INIT(E_Const, "i");
  INIT(E_Volatile, "i");
  INIT(E_ConstVolatile, "i");
  INIT(E_NNS, "i");
  INIT(E_NNS_Again, "i");
  INIT(W, 0);
  INIT(W_ConstVolatile, 0);
  INIT(W_Arr_3, "arr[3]");
  INIT(W_Arr_5, "arr[5]");
  INIT(W_Pointer_3, "arr[3]");
  INIT(W_Pointer_5, "arr[5]");
  INIT(lambda, "i");
  INIT(lambdaWithCapture, "i");
  INIT(lambdaNNS, "i");

  TEST(E);                        // CHECK-EXEC: 27.00 27.00
  TEST(E_Again);                  // CHECK-EXEC: 27.00 27.00
  TEST(E_Const);                  // CHECK-EXEC: 27.00 27.00
  TEST(E_Volatile);               // CHECK-EXEC: 27.00 27.00
  TEST(E_ConstVolatile);          // CHECK-EXEC: 27.00 27.00
  TEST(E_NNS);                    // CHECK-EXEC: 27.00 27.00
  TEST(E_NNS_Again);              // CHECK-EXEC: 27.00 27.00
  TEST(lambda);                   // CHECK-EXEC: 126.00 126.00
  TEST(lambdaWithCapture);        // CHECK-EXEC: 27.00 27.00
  TEST(lambdaNNS);                // CHECK-EXEC: 126.00 126.00
  TEST_2(W, 3, 5);                // CHECK-EXEC: 7.00 7.00
  TEST_2(W_ConstVolatile, 3, 5);  // CHECK-EXEC: 7.00 7.00
  TEST_2(W_Arr_3, 3, 5);          // CHECK-EXEC: 4.00 4.00
  TEST_2(W_Arr_5, 3, 5);          // CHECK-EXEC: 6.00 6.00
  TEST_2(W_Pointer_3, 3, 5);      // CHECK-EXEC: 19.00 19.00
  TEST_2(W_Pointer_5, 3, 5);      // CHECK-EXEC: 51.00 51.00

  E.setX(6);
  E_Again.setX(6);
  E_Const.setX(6);
  E_Volatile.setX(6);
  E_ConstVolatile.setX(6);
  E_NNS.setX(6);
  E_NNS_Again.setX(6);
  x = 6;

  TEST(E);                        // CHECK-EXEC: 54.00 54.00
  TEST(E_Again);                  // CHECK-EXEC: 54.00 54.00
  TEST(E_Const);                  // CHECK-EXEC: 54.00 54.00
  TEST(E_Volatile);               // CHECK-EXEC: 54.00 54.00
  TEST(E_ConstVolatile);          // CHECK-EXEC: 54.00 54.00
  TEST(E_NNS);                    // CHECK-EXEC: 54.00 54.00
  TEST(E_NNS_Again);              // CHECK-EXEC: 54.00 54.00
  TEST(lambdaWithCapture);        // CHECK-EXEC: 54.00 54.00
  TEST_2(W, 6, 5);                // CHECK-EXEC: 13.00 13.00
  TEST_2(W_ConstVolatile, 6, 5);  // CHECK-EXEC: 13.00 13.00
  TEST_2(W_Arr_3, 6, 5);          // CHECK-EXEC: 7.00 7.00
  TEST_2(W_Arr_5, 6, 5);          // CHECK-EXEC: 6.00 6.00
  TEST_2(W_Pointer_3, 6, 5);      // CHECK-EXEC: 37.00 37.00
  TEST_2(W_Pointer_5, 6, 5);      // CHECK-EXEC: 51.00 51.00
}
