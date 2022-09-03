// RUN: %cladclang %s -I%S/../../include -oReverseMode.out | FileCheck %s
// RUN: ./ReverseMode.out | FileCheck -check-prefix=CHECK-EXEC %s
// CHECK-NOT: {{.*error|warning|note:.*}}
// REQUIRES: Enzyme

#include "clad/Differentiator/Differentiator.h"

double f1(double* arr) { return arr[0] * arr[1]; }

// CHECK: void f1_grad_enzyme(double *arr, clad::array_ref<double> _d_arr) {
// CHECK-NEXT:    double *d_arr = _d_arr.ptr();
// CHECK-NEXT:    __enzyme_autodiff_f1(f1, arr, d_arr);
// CHECK-NEXT:}

double f2(double x, double y, double z){
    return x * y * z;
}

// CHECK: void f2_grad_enzyme(double x, double y, double z, clad::array_ref<double> _d_x, clad::array_ref<double> _d_y, clad::array_ref<double> _d_z) {
// CHECK-NEXT:    clad::EnzymeGradient<3> grad = __enzyme_autodiff_f2(f2, x, y, z);
// CHECK-NEXT:    * _d_x = grad.d_arr[0U];
// CHECK-NEXT:    * _d_y = grad.d_arr[1U];
// CHECK-NEXT:    * _d_z = grad.d_arr[2U];
// CHECK-NEXT:}

double f3(double* arr, int n){
    double sum=0;
    for(int i=0;i<n;i++){
        sum+=arr[i]*arr[i];
    }
    return sum;
}

// CHECK: void f3_grad_enzyme(double *arr, int n, clad::array_ref<double> _d_arr, clad::array_ref<int> _d_n) {
// CHECK-NEXT:     double *d_arr = _d_arr.ptr();
// CHECK-NEXT:     __enzyme_autodiff_f3(f3, arr, d_arr, n);
// CHECK-NEXT: }

double f4(double* arr1, int n,  double*arr2, int m){
    double sum=0;
    for(int i=0;i<n;i++){
        sum+=arr1[i]*arr1[i];
    }
    for(int i=0;i<m;i++){
        sum+=arr2[i]*arr2[i];
    }
    return sum;
}

// CHECK: void f4_grad_enzyme(double *arr1, int n, double *arr2, int m, clad::array_ref<double> _d_arr1, clad::array_ref<int> _d_n, clad::array_ref<double> _d_arr2, clad::array_ref<int> _d_m) {
// CHECK-NEXT:     double *d_arr1 = _d_arr1.ptr();
// CHECK-NEXT:     double *d_arr2 = _d_arr2.ptr();   
// CHECK-NEXT:     __enzyme_autodiff_f4(f4, arr1, d_arr1, n, arr2, d_arr2, m);
// CHECK-NEXT: }

double f5(double arr[], double x,int n,double y){
    double res=0;
    for(int i=0;i<n;i++){
        res+=(arr[i]*x*y);
    }
    return res;
}

// CHECK: void f5_grad_enzyme(double arr[], double x, int n, double y, clad::array_ref<double> _d_arr, clad::array_ref<double> _d_x, clad::array_ref<int> _d_n, clad::array_ref<double> _d_y) {
// CHECK-NEXT:     double *d_arr = _d_arr.ptr();
// CHECK-NEXT:     clad::EnzymeGradient<2> grad = __enzyme_autodiff_f5(f5, arr, d_arr, x, n, y);
// CHECK-NEXT:     * _d_x = grad.d_arr[0U];
// CHECK-NEXT:     * _d_y = grad.d_arr[1U];
// CHECK-NEXT: }


int main() {
  auto f1_grad = clad::gradient<clad::opts::use_enzyme>(f1);
  double f1_v[2] = {3, 4};
  double f1_g[2] = {0};
  f1_grad.execute(f1_v, f1_g);
  printf("d_x = %.2f, d_y = %.2f\n", f1_g[0], f1_g[1]); 
  // CHECK-EXEC: d_x = 4.00, d_y = 3.00

  auto f2_grad=clad::gradient<clad::opts::use_enzyme>(f2);
  double f2_res[3];
  double f2_x=3,f2_y=4,f2_z=5;
  f2_grad.execute(f2_x,f2_y,f2_z,&f2_res[0],&f2_res[1],&f2_res[2]);
  printf("d_x = %.2f, d_y = %.2f, d_z = %.2f\n", f2_res[0], f2_res[1], f2_res[2]); 
  //CHECK-EXEC: d_x = 20.00, d_y = 15.00, d_z = 12.00

  auto f3_grad=clad::gradient<clad::opts::use_enzyme>(f3);
  double f3_list[3]={3,4,5};
  double f3_res[3]={0};
  int f3_dn=0;
  f3_grad.execute(f3_list,3,f3_res,&f3_dn);
  printf("d_x1 = %.2f, d_x2 = %.2f, d_x3 = %.2f, d_n = %d\n",f3_res[0],f3_res[1],f3_res[2],f3_dn); 
  //CHECK-EXEC: d_x1 = 6.00, d_x2 = 8.00, d_x3 = 10.00, d_n = 0

  auto f4_grad=clad::gradient<clad::opts::use_enzyme>(f4);
  double f4_list1[3]={3,4,5};
  double f4_list2[2]={1,2};
  double f4_res1[3]={0};
  double f4_res2[2]={0};
  int f4_dn1=0,f4_dn2=0;
  f4_grad.execute(f4_list1,3,f4_list2,2,f4_res1,&f4_dn1,f4_res2,&f4_dn2);
  printf("d_x1 = %.2f, d_x2 = %.2f, d_x3 = %.2f, d_n1 = %d\n",f4_res1[0],f4_res1[1],f4_res1[2],f4_dn1); 
  //CHECK-EXEC: d_x1 = 6.00, d_x2 = 8.00, d_x3 = 10.00, d_n1 = 0
  printf("d_y1 = %.2f, d_y2 = %.2f, d_n2 = %d\n",f4_res2[0],f4_res2[1],f4_dn2); 
  //CHECK-EXEC: d_y1 = 2.00, d_y2 = 4.00, d_n2 = 0

  auto f5_grad=clad::gradient<clad::opts::use_enzyme>(f5);
  double f5_list[3]={3,4,5};
  double f5_res[3]={0};
  double f5_x=10.0,f5_dx=0,f5_y=5,f5_dy;
  int f5_dn=0;
  f5_grad.execute(f5_list,f5_x,3,f5_y,f5_res,&f5_dx,&f5_dn,&f5_dy);
  printf("d_x1 = %.2f, d_x2 = %.2f, d_x3 = %.2f, d_n1 = %d, d_x = %.2f, d_y = %.2f\n",f5_res[0],f5_res[1],f5_res[2],f5_dn, f5_dx, f5_dy); 
  //CHECK-EXEC: d_x1 = 50.00, d_x2 = 50.00, d_x3 = 50.00, d_n1 = 0, d_x = 60.00, d_y = 120.00

}
