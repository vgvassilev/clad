#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "math.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <iostream>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

  //-------------------------------------------------------------------------
  // Vector3
  //-------------------------------------------------------------------------

  struct Vector3 {

  public:
    //---------------------------------------------------------------------
    // Constructors and Destructors
    //---------------------------------------------------------------------

    constexpr explicit Vector3(double xyz = 0.0) noexcept
        : Vector3(xyz, xyz, xyz) {}
    constexpr Vector3(double x, double y, double z) noexcept
        : m_x(x), m_y(y), m_z(z) {}
    constexpr Vector3(const Vector3& v) noexcept = default;
    constexpr Vector3(Vector3&& v) noexcept = default;
    ~Vector3() = default;

    //---------------------------------------------------------------------
    // Assignment Operators
    //---------------------------------------------------------------------

    Vector3& operator=(const Vector3& v) = default;
    Vector3& operator=(Vector3&& v) = default;

    //---------------------------------------------------------------------
    // Member Methods
    //---------------------------------------------------------------------

    [[nodiscard]] bool HasNaNs() const noexcept {
      return std::isnan(m_x) || std::isnan(m_y) || std::isnan(m_z);
    }

    [[nodiscard]] constexpr const Vector3 operator-() const noexcept {
      return {-m_x, -m_y, -m_z};
    }

    [[nodiscard]] constexpr const Vector3
    operator+(const Vector3& v) const noexcept {
      return {m_x + v.m_x, m_y + v.m_y, m_z + v.m_z};
    }

    [[nodiscard]] constexpr const Vector3
    operator-(const Vector3& v) const noexcept {
      return {m_x - v.m_x, m_y - v.m_y, m_z - v.m_z};
    }

    [[nodiscard]] constexpr const Vector3
    operator*(const Vector3& v) const noexcept {
      return {m_x * v.m_x, m_y * v.m_y, m_z * v.m_z};
    }

    [[nodiscard]] constexpr const Vector3
    operator/(const Vector3& v) const noexcept {
      return {m_x / v.m_x, m_y / v.m_y, m_z / v.m_z};
    }

    [[nodiscard]] constexpr const Vector3 operator+(double a) const noexcept {
      return {m_x + a, m_y + a, m_z + a};
    }

    [[nodiscard]] constexpr const Vector3 operator-(double a) const noexcept {
      return {m_x - a, m_y - a, m_z - a};
    }

    [[nodiscard]] constexpr const Vector3 operator*(double a) const noexcept {
      return {m_x * a, m_y * a, m_z * a};
    }

    [[nodiscard]] constexpr const Vector3 operator/(double a) const noexcept {
      const double inv_a = 1.0 / a;
      return {m_x * inv_a, m_y * inv_a, m_z * inv_a};
    }

    Vector3& operator+=(const Vector3& v) noexcept {
      m_x += v.m_x;
      m_y += v.m_y;
      m_z += v.m_z;
      return *this;
    }

    Vector3& operator-=(const Vector3& v) noexcept {
      m_x -= v.m_x;
      m_y -= v.m_y;
      m_z -= v.m_z;
      return *this;
    }

    Vector3& operator*=(const Vector3& v) noexcept {
      m_x *= v.m_x;
      m_y *= v.m_y;
      m_z *= v.m_z;
      return *this;
    }

    Vector3& operator/=(const Vector3& v) noexcept {
      m_x /= v.m_x;
      m_y /= v.m_y;
      m_z /= v.m_z;
      return *this;
    }

    Vector3& operator+=(double a) noexcept {
      m_x += a;
      m_y += a;
      m_z += a;
      return *this;
    }

    Vector3& operator-=(double a) noexcept {
      m_x -= a;
      m_y -= a;
      m_z -= a;
      return *this;
    }

    Vector3& operator*=(double a) noexcept {
      m_x *= a;
      m_y *= a;
      m_z *= a;
      return *this;
    }

    Vector3& operator/=(double a) noexcept {
      const double inv_a = 1.0 / a;
      m_x *= inv_a;
      m_y *= inv_a;
      m_z *= inv_a;
      return *this;
    }

    [[nodiscard]] constexpr double Dot(const Vector3& v) const noexcept {
      return m_x * v.m_x + m_y * v.m_y + m_z * v.m_z;
    }

    [[nodiscard]] constexpr const Vector3
    Cross(const Vector3& v) const noexcept {
      return {m_y * v.m_z - m_z * v.m_y, m_z * v.m_x - m_x * v.m_z,
              m_x * v.m_y - m_y * v.m_x};
    }

    [[nodiscard]] constexpr bool operator==(const Vector3& rhs) const {
      return m_x == rhs.m_x && m_y == rhs.m_y && m_z == rhs.m_z;
    }

    [[nodiscard]] constexpr bool operator!=(const Vector3& rhs) const {
      return !(*this == rhs);
    }

    [[nodiscard]] double& operator[](std::size_t i) noexcept {
      return (&m_x)[i];
    }

    [[nodiscard]] constexpr double operator[](std::size_t i) const noexcept {
      return (&m_x)[i];
    }

    [[nodiscard]] constexpr std::size_t MinDimension() const noexcept {
      return (m_x < m_y && m_x < m_z) ? 0u : ((m_y < m_z) ? 1u : 2u);
    }

    [[nodiscard]] constexpr std::size_t MaxDimension() const noexcept {
      return (m_x > m_y && m_x > m_z) ? 0u : ((m_y > m_z) ? 1u : 2u);
    }

    [[nodiscard]] constexpr double Min() const noexcept {
      return std::min(m_x, std::min(m_y, m_z));
    }
    [[nodiscard]] constexpr double Max() const noexcept {
      return std::max(m_x, std::max(m_y, m_z));
    }

    [[nodiscard]] constexpr double Norm2_squared() const noexcept {
      return m_x * m_x + m_y * m_y + m_z * m_z;
    }

    [[nodiscard]] double Norm2() const noexcept {
      return std::sqrt(Norm2_squared());
    }

    void Normalize() noexcept {
      const double a = 1.0 / Norm2();
      m_x *= a;
      m_y *= a;
      m_z *= a;
    }

    //---------------------------------------------------------------------
    // Member Variables
    //---------------------------------------------------------------------

    double m_x, m_y, m_z;
  };

  //-------------------------------------------------------------------------
  // Vector3 Utilities
  //-------------------------------------------------------------------------

  std::ostream& operator<<(std::ostream& os, const Vector3& v) {
    os << '[' << v.m_x << ' ' << v.m_y << ' ' << v.m_z << ']';
    return os;
  }

  [[nodiscard]] constexpr const Vector3 operator+(double a,
                                                  const Vector3& v) noexcept {
    return {a + v.m_x, a + v.m_y, a + v.m_z};
  }

  [[nodiscard]] constexpr const Vector3 operator-(double a,
                                                  const Vector3& v) noexcept {
    return {a - v.m_x, a - v.m_y, a - v.m_z};
  }

  [[nodiscard]] constexpr const Vector3 operator*(double a,
                                                  const Vector3& v) noexcept {
    return {a * v.m_x, a * v.m_y, a * v.m_z};
  }

  [[nodiscard]] constexpr const Vector3 operator/(double a,
                                                  const Vector3& v) noexcept {
    return {a / v.m_x, a / v.m_y, a / v.m_z};
  }

  [[nodiscard]] inline const Vector3 Sqrt(const Vector3& v) noexcept {
    return {std::sqrt(v.m_x), std::sqrt(v.m_y), std::sqrt(v.m_z)};
  }

  [[nodiscard]] inline const Vector3 Pow(const Vector3& v, double a) noexcept {
    return {std::pow(v.m_x, a), std::pow(v.m_y, a), std::pow(v.m_z, a)};
  }

  [[nodiscard]] inline const Vector3 Abs(const Vector3& v) noexcept {
    return {std::abs(v.m_x), std::abs(v.m_y), std::abs(v.m_z)};
  }

  [[nodiscard]] constexpr const Vector3 Min(const Vector3& v1,
                                            const Vector3& v2) noexcept {
    return {std::min(v1.m_x, v2.m_x), std::min(v1.m_y, v2.m_y),
            std::min(v1.m_z, v2.m_z)};
  }

  [[nodiscard]] constexpr const Vector3 Max(const Vector3& v1,
                                            const Vector3& v2) noexcept {
    return {std::max(v1.m_x, v2.m_x), std::max(v1.m_y, v2.m_y),
            std::max(v1.m_z, v2.m_z)};
  }

  [[nodiscard]] inline const Vector3 Round(const Vector3& v) noexcept {
    return {std::round(v.m_x), std::round(v.m_y), std::round(v.m_z)};
  }

  [[nodiscard]] inline const Vector3 Floor(const Vector3& v) noexcept {
    return {std::floor(v.m_x), std::floor(v.m_y), std::floor(v.m_z)};
  }

  [[nodiscard]] inline const Vector3 Ceil(const Vector3& v) noexcept {
    return {std::ceil(v.m_x), std::ceil(v.m_y), std::ceil(v.m_z)};
  }

  [[nodiscard]] inline const Vector3 Trunc(const Vector3& v) noexcept {
    return {std::trunc(v.m_x), std::trunc(v.m_y), std::trunc(v.m_z)};
  }

  [[nodiscard]] constexpr const Vector3
  Clamp(const Vector3& v, double low = 0.0, double high = 1.0) noexcept {

    return {clamp(v.m_x, low, high), clamp(v.m_y, low, high),
            clamp(v.m_z, low, high)};
  }
  [[nodiscard]] constexpr const Vector3 Lerp(double a, const Vector3& v1,
                                             const Vector3& v2) noexcept {
    return v1 + a * (v2 - v1);
  }

  template <std::size_t X, std::size_t Y, std::size_t Z>
  [[nodiscard]] constexpr const Vector3 Permute(const Vector3& v) noexcept {
    return {v[X], v[Y], v[Z]};
  }

  [[nodiscard]] inline const Vector3 Normalize(const Vector3& v) noexcept {
    const double a = 1.0 / v.Norm2();
    return a * v;
  }
} // namespace smallpt
