namespace clad {
namespace custom_derivatives {
inline void opaque_pullback(double& x, double _d_y, double* _d_x) {
  *_d_x += 2. * _d_y;
}
} // namespace custom_derivatives
} // namespace clad
