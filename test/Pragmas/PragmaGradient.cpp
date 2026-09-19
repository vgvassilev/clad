// RUN: %cladclang -c %s -o %t.o
// RUN: %clang %t.o -o %t.exe
// RUN: %t.exe | %FileCheck %s

extern "C" int printf(const char*, ...);

double add(double i) {
    return i + i;
}

#pragma clad gradient add
#pragma clad differentiate add

// Declare the signature of the generated derivatives
extern "C" {
    void add_grad(double i, double* _grad);
    double add_darg0(double i);
}

int main() {
    double grad = 0.0;
    add_grad(3.0, &grad);
    double df = add_darg0(3.0);
    printf("grad = %.1f, df = %.1f\n", grad, df);
    // CHECK: grad = 2.0, df = 2.0
    return 0;
}
