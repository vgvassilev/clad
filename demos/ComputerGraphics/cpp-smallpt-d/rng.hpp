#pragma once

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <cstdint>
#include <random>
#include <limits>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

  class RNG {
  public:
    //---------------------------------------------------------------------
    // Constructors and Destructors
    //---------------------------------------------------------------------

    explicit RNG(std::uint32_t seed = 606418532u) noexcept
        : m_generator(), m_distribution() {
      Seed(seed);
    }
    RNG(const RNG& rng) noexcept = default;
    RNG(RNG&& rng) noexcept = default;
    ~RNG() = default;

    //---------------------------------------------------------------------
    // Assignment Operators
    //---------------------------------------------------------------------

    RNG& operator=(const RNG& rng) = delete;
    RNG& operator=(RNG&& rng) = delete;

    //---------------------------------------------------------------------
    // Member Methods
    //---------------------------------------------------------------------

    void Seed(uint32_t seed) noexcept { /*m_generator.seed(seed);*/ }

    double Uniform() noexcept { return m_distribution(m_generator); }

  private:
    //---------------------------------------------------------------------
    // Member Variables
    //---------------------------------------------------------------------

    std::default_random_engine m_generator;
    std::uniform_real_distribution<double> m_distribution;
  };
} // namespace smallpt
