// RUN: %cladclang %s -o %t
// RUN: %t | %filecheck_exec %s

extern "C" int printf(const char*, ...);

double add(double a, double b) {
    return a + b;
}

#pragma clad gradient add
#pragma clad differentiate add

int main() {
    double grad[2] = {0.0, 0.0};
    add_grad(3.0, 4.0, grad);
    double df = add_darg0(3.0, 4.0);
    printf("grad = %.1f, %.1f, df = %.1f\n", grad[0], grad[1], df);
    // CHECK-EXEC: grad = 1.0, 1.0, df = 1.0
    return 0;
}
