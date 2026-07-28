// RUN: %cladclang -std=c++17 -I%S/../../include %s -o %t
// RUN: %t | %filecheck_exec %s

#include <cassert>
#include "clad/Differentiator/Differentiator.h"

double func(double x){
    assert(x>0.0);
    return x*x*x;
}
double func1(double x) {
    const char* file = __builtin_FILE();
    int line = __builtin_LINE();
    return x*x*x;
}

void temp() {}
double temp_func(double x) {
    x > 0.0 ? temp() : temp();
    return x * x * x;
}

int main(){
    auto d_func=clad::gradient(func);
    double dx=0;
    d_func.execute(3.0,&dx);
    printf("Diff result: %.2f\n", dx);
    //CHECK-EXEC: Diff result: 27.00
    
    auto d_func_forw=clad::differentiate(func);
    auto res_forw=d_func_forw.execute(3.0);
    printf("Diff result: %.2f\n", res_forw);
    // CHECK-EXEC: Diff result: 27.00

    auto d_func1=clad::gradient(func1);
    double dx1=0;
    d_func1.execute(3.0,&dx1);
    printf("Diff result: %.2f\n", dx1);
    //CHECK-EXEC: Diff result: 27.00

    auto d_func1_forw = clad::differentiate(func1);
    auto res1_forw = d_func1_forw.execute(3.0);
    printf("Diff result: %.2f\n", res1_forw);
    //CHECK-EXEC: Diff result: 27.00

    auto d_temp_func = clad::differentiate(temp_func);
    auto res_void = d_temp_func.execute(3.0);
    printf("Diff result: %.2f\n", res_void);
    //CHECK-EXEC: Diff result: 27.00

    return 0;
}