
///\returns the x squared.
inline double pow2(double x) { return x * x; }

///\returns the sum of the elements in \p
inline double sum(double* p, int dim) {
  double r = 0.0;
  for (int i = 0; i < dim; i++)
    r += p[i];
  return r;
}

