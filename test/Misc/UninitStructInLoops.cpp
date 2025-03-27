#include "clad/Differentiator/Differentiator.h"
#include <stdio.h>

typedef struct {
    double val;
} Struct;

void fn1(double a) {
    for (int i=0;i<1;i++){
        Struct s;
    }
}

void fn2 (int a) {
    int i = 1;
    while (i--) {
        Struct s1 = {0};
        Struct s2;
    }
}

int main(){
    auto grad = clad::gradient(fn1);
    auto grad2 = clad::gradient(fn2);
    return 0;
}