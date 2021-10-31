
///\returns the x squared.
inline double pow2(double x) { return x * x; }

///\returns the sum of the elements in \p
inline double sum(double* p, int dim) {
  double r = 0.0;
  for (int i = 0; i < dim; i++)
    r += p[i];
  return r;
}

///\returns the gaussian distribution. We need always_inline to improve the
/// performance of reverse mode.
__attribute__((always_inline)) inline double
gaus(double* x, double* p /*means*/, double sigma, int dim) {
  double t = 0;
  for (int i = 0; i < dim; i++)
    t += (x[i] - p[i]) * (x[i] - p[i]);
  t = -t / (2 * sigma * sigma);
  return std::pow(2 * M_PI, -dim / 2.0) * std::pow(sigma, -0.5) * std::exp(t);
}
