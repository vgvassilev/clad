#ifndef CLAD_NUMERICAL_DIFF_H
#define CLAD_NUMERICAL_DIFF_H

#include <cmath>
#include <limits>

namespace numerical_diff {
  /// A function to make sure the step size being used is machine representable.
  /// It is likely that if do not have a similar construct, we may end up with
  /// catastrophic cancellations, hence resulting into a 0 derivative over a
  /// large interval.
  ///
  /// \param[in] x Input to the target function.
  /// \param[in] h The h to make representable.
  ///
  /// \returns A value of h that does not result in catastrohic cancellation.
  double make_h_representable(double x, double h) {
    double xph = x + h;
    double dx = xph - x;

    // If x+h ~ x, we should look for the next closest value so that (x+h) - x
    // != 0
    if (dx == 0)
      h = std::nextafter(x, std::numeric_limits<double>::max()) - x;

    return h;
  }

  /// A funtion to calculate the derivative of a function using the central
  /// difference formula. Note: we do not propogate errors resulting in the
  /// following function, it is likely the errors are large enough to be of
  /// significance, hence it is only wise to use these methods when it is
  /// absolutely necessary.
  ///
  /// \param[in] f The function to calculate the derivative of.
  /// \param[in] x The input value at which the function's derivative is
  /// required.
  ///
  /// \returns A first order derivative of 'f' caculated using the five-point
  /// stencil method.
  // TODO: Exten this for function with multiple params.
  template <typename F> double central_difference(F f, double x) {
    // Maximum error in h = eps^4/5
    double h = std::pow(11.25 * std::numeric_limits<double>::epsilon(),
                        (double)1.0 / 5.0);
    h = make_h_representable(x, h);

    // calculate f[x+h, x-h]
    double xah = x + h;
    double xbh = x - h;
    double xf1 = (f(xah) - f(xbh)) / (xah - xbh);

    // calculate f[x+2h, x-2h]
    xah += h;
    xbh -= h;
    double xf2 = (f(xah) - f(xbh)) / (xah - xbh);

    // five-point sentcil formula = (4f[x+h, x-h] - f[x+2h, x-2h])/3
    double dx = 4.0 * xf1 / 3.0 - xf2 / 3.0;

    // FIXME: We should allow some diagnostic printing of truncation errors...
    return dx;
  }

} // namespace numerical_diff

#endif
