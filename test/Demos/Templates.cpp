// Clad differentiates the specialisation the compiler picked, not the primary
// template. Equation<long double> computes twice what the general one does, so
// its derivative has to carry that factor through. The demo prints nothing, so
// the code clad wrote is the only thing to read.
//
// RUN: %cladclang %S/../../demos/Templates.cpp -I%S/../../include -o%t 2>&1 \
// RUN:     | %filecheck %s

// CHECK: double operator_call_darg0(double i, double j) {
// CHECK: double &_t0 = this->m_x;
// CHECK: return (_d_m_x * i + _t0 * _d_i) * i + _t1 * _d_i + (_d_m_y * j + _t2 * _d_j) * j + _t3 * _d_j;

// The factor of two belongs to the long double specialisation alone.
// CHECK: long double operator_call_darg1(long double i, long double j) {
// CHECK: long double _t0 = 2 * this->m_x;
// CHECK: return ((2 * _d_m_x) * i + _t0 * _d_i) * i + _t1 * _d_i + ((2 * _d_m_y) * j + _t2 * _d_j) * j + _t3 * _d_j;

// A function template is differentiated per instantiation in the same way.
// CHECK: double kinetic_energy_darg0(double mass, double velocity) {
// CHECK: long double kinetic_energy_darg1(long double mass, long double velocity) {
