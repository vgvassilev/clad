// expected-no-diagnostics
// RUN:
double single_arg(double x) {
    return 2 * x;
}

double multi_arg(double x, double y) {
    return x + y;
}

void noNumDiff(double& x) {
    x += x;
}
