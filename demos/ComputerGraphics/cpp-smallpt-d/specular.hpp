#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "rng.hpp"
#include "vector.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

  [[nodiscard]]
  constexpr double Reflectance0(double n1, double n2) noexcept {
    const double sqrt_R0 = (n1 - n2) / (n1 + n2);
    return sqrt_R0 * sqrt_R0;
  }

  [[nodiscard]]
  constexpr double SchlickReflectance(double n1, double n2, double c) noexcept {
    const double R0 = Reflectance0(n1, n2);
    return R0 + (1.0 - R0) * c * c * c * c * c;
  }

  [[nodiscard]]
  constexpr const Vector3 IdealSpecularReflect(const Vector3& d, const Vector3& n) noexcept {
    return d - 2.0 * n.Dot(d) * n;
  }

  [[nodiscard]]
  inline const Vector3 IdealSpecularTransmit(const Vector3& d, const Vector3& n, double n_out,
                        double n_in, double& pr, RNG& rng) noexcept {
    const Vector3 d_Re = IdealSpecularReflect(d, n);

    const bool out_to_in = (0.0 > n.Dot(d));
    const Vector3 nl = out_to_in ? n : -n;
    const double nn = out_to_in ? n_out / n_in : n_in / n_out;
    const double cos_theta = d.Dot(nl);
    const double cos2_phi = 1.0 - nn * nn * (1.0 - cos_theta * cos_theta);

    // Total Internal Reflection
    if (0.0 > cos2_phi) {
      pr = 1.0;
      return d_Re;
    }

    const Vector3 d_Tr = Normalize(nn * d - nl * (nn * cos_theta + sqrt(cos2_phi)));
    const double c = 1.0 - (out_to_in ? -cos_theta : d_Tr.Dot(n));

    const double Re = SchlickReflectance(n_out, n_in, c);
    const double p_Re = 0.25 + 0.5 * Re;
    if (rng.Uniform() < p_Re) {
      pr = (Re / p_Re);
      return d_Re;
    } else {
      const double Tr = 1.0 - Re;
      const double p_Tr = 1.0 - p_Re;
      pr = (Tr / p_Tr);
      return d_Tr;
    }
  }
} // namespace smallpt
