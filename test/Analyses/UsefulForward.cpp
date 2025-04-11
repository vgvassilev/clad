// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oUseful.out 2>&1 | %filecheck %s
// RUN: ./Useful.out | %filecheck_exec %s
// RUN: %cladclang -Xclang -plugin-arg-clad -Xclang -enable-ua -Xclang -plugin-arg-clad -Xclang -disable-tbr %s -I%S/../../include -oUseful.out
// RUN: ./Useful.out | %filecheck_exec %s
//CHECK-NOT: {{.*error|warning|note:.*}}

#include "clad/Differentiator/Differentiator.h"
#include "../TestUtils.h"

double f1(double x){
    double b = 1;
    return x;
}

// CHECK: double f1_darg0(double x) {
// CHECK-NEXT:    double _d_x = 1;
// CHECK-NEXT:    double b = 1;
// CHECK-NEXT:    return _d_x;
// CHECK-NEXT:}

int main(){
    INIT_DIFFERENTIATE_UA(f1, "x");

    TEST_DIFFERENTIATE(f1, 3); // CHECK-EXEC: {1.00}
    return 0;
}