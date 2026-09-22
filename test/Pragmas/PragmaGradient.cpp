// RUN: %cladclang %s -o %t
// RUN: %t | %filecheck_exec %s

extern "C" int printf(const char*, ...);

double mul(double x) {
    return x * x;
}

double add(double i) {
    return mul(i) + i;
}

#pragma clad gradient add
#pragma clad differentiate add

int main() {
    double grad = 0.0;
    add_grad(3.0, &grad);
    double df = add_darg0(3.0);
    printf("grad = %.1f, df = %.1f\n", grad, df);
    // CHECK-EXEC: grad = 7.0, df = 7.0
    return 0;
}
