
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
gaus(const double* x, double* p /*means*/, double sigma, int dim) {
  double t = 0;
  for (int i = 0; i < dim; i++)
    t += (x[i] - p[i]) * (x[i] - p[i]);
  t = -t / (2 * sigma * sigma);
  return std::pow(2 * M_PI, -dim / 2.0) * std::pow(sigma, -0.5) * std::exp(t);
}

///\returns the sum of elements in an array multiplied by scalars x and y
inline double addArrayAndMultiplyWithScalars(double arr[], double x, double y,
                                             int n) {
  double res = 0;
  for (int i = 0; i < n; i++) {
    res += (arr[i] * x * y);
  }
  return res;
}

///\returns the product of the elements in \p
inline double product(double p[], int n) {
  double prod = 1;
  for (int i = 0; i < n; i++) {
    prod *= p[i];
  }
  return prod;
}

///\returns the weighted sum of the elements in \p
inline double weightedSum(double p[], double w[], int n) {
  double sum = 0;
  for (int i = 0; i < n; i++)
    sum += p[i] * w[i];
  return sum;
}