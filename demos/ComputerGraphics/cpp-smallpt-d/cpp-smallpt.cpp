//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to use clad in simple Differentiable Path tracer.
//
// Author:  Alexander Penev <alexander_penev-at-yahoo.com>, 2022
//          Based on smallpt, a Path Tracer by Kevin Beason, 2008
// Based on: C++ modification of Kevin Baeson's 99 line C++ path tracer
//           https://github.com/matt77hias/cpp-smallpt
//----------------------------------------------------------------------------//

// To compile the demo please type:
// path/to/clang -O3 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so -I../../include/ -x c++ -std=c++11 -lstdc++ -lm \
// cpp-smallpt-d.cpp -fopenmp=libiomp5 -o SmallPT
//
// To run the demo please type:
// ./cpp-smallpt-d 500 && xv image.ppm

// A typical invocation would be:
// ../../../../../obj/Debug+Asserts/bin/clang -O3 -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../../obj/Debug+Asserts/lib/libclad.dylib \
// -I../../include/ -x c++ -std=c++11 -lstdc++ -lm cpp-smalpt-d.cpp -fopenmp=libiomp5 \
// -o cpp-smallpt-d
// ./cpp-smallpt-d 500 && xv image.ppm


//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "imageio.hpp"
#include "sampling.hpp"
#include "specular.hpp"
#include "sphere.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <iterator>
#include <memory>

#pragma endregion

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

#define REFRACTIVE_INDEX_OUT 1.0
#define REFRACTIVE_INDEX_IN 1.5

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

  /*constexpr*/ Sphere* scene[] = {
      new Sphere(1e5,  Vector3(1e5 + 1, 40.8, 81.6),   Vector3(),   Vector3(0.75, 0.25, 0.25), Reflection_t::Diffuse),   // Left
      new Sphere(1e5,  Vector3(-1e5 + 99, 40.8, 81.6), Vector3(),   Vector3(0.25, 0.25, 0.75), Reflection_t::Diffuse),   // Right
      new Sphere(1e5,  Vector3(50, 40.8, 1e5),         Vector3(),   Vector3(0.75),             Reflection_t::Diffuse),   // Back
      new Sphere(1e5,  Vector3(50, 40.8, -1e5 + 170),  Vector3(),   Vector3(),                 Reflection_t::Diffuse),   // Front
      new Sphere(1e5,  Vector3(50, 1e5, 81.6),         Vector3(),   Vector3(0.75),             Reflection_t::Diffuse),   // Bottom
      new Sphere(1e5,  Vector3(50, -1e5 + 81.6, 81.6), Vector3(),   Vector3(0.75),             Reflection_t::Diffuse),   // Top
      new Sphere(16.5, Vector3(27, 16.5, 47),          Vector3(),   Vector3(0.999),            Reflection_t::Refractive),  // Glass
//      new Sphere(16.5, Vector3(55, 30, 57),          Vector3(),   Vector3(0.999),            Reflection_t::Refractive),  // Glassr
//      new Sphere(16.5, Vector3(80, 60, 67),          Vector3(),   Vector3(0.999),            Reflection_t::Refractive),  // Glass
      new Sphere(16.5, Vector3(73, 16.5, 78),          Vector3(),   Vector3(0.999),            Reflection_t::Specular),// Mirror
      new Sphere(600,  Vector3(50, 681.6 - .27, 81.6), Vector3(12), Vector3(),                 Reflection_t::Diffuse)    // Light
  };

  [[nodiscard]]
  /*constexpr*/ size_t Intersect(Sphere* g_scene[], const size_t n_scene, const Ray& ray) noexcept {
    size_t hit = SIZE_MAX;
    for (size_t i = 0u; i < n_scene; ++i) {
      if (g_scene[i]->Intersect(ray)) {
        hit = i;
      }
    }

    return hit;
  }

  [[nodiscard]]
  static const Vector3 Radiance(Sphere* g_scene[], const size_t n_scene, const Ray& ray, RNG& rng) noexcept {
    Ray r = ray;
    Vector3 L;
    Vector3 F(1.0);

    while (true) {
      const auto hit = Intersect(g_scene, n_scene, r);
      if (hit == SIZE_MAX) {
        return L;
      }

      const Sphere* shape = g_scene[hit];
      const Vector3 p = r(r.m_tmax);
      const Vector3 n = Normalize(p - shape->m_p);

      L += F * shape->m_e;
      F *= shape->m_f;

      // Russian roulette
      if (4u < r.m_depth) {
        const double continue_probability = shape->m_f.Max();
        if (rng.Uniform() >= continue_probability) {
          return L;
        }
        F /= continue_probability;
      }

      // Next path segment
      switch (shape->m_reflection_t) {

        case Reflection_t::Specular: {
          const Vector3 d = IdealSpecularReflect(r.m_d, n);
          r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
          break;
        }

        case Reflection_t::Refractive: {
          double pr;
          const Vector3 d = IdealSpecularTransmit(r.m_d, n,
                                                  REFRACTIVE_INDEX_OUT,
                                                  REFRACTIVE_INDEX_IN, pr, rng);
          F *= pr;
          r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
          break;
        }

        default: {
          const Vector3 w = (0.0 > n.Dot(r.m_d)) ? n : -n;
          const Vector3 u = Normalize((std::abs(w.m_x) > 0.1
                                           ? Vector3(0.0, 1.0, 0.0)
                                           : Vector3(1.0, 0.0, 0.0))
                                          .Cross(w));
          const Vector3 v = w.Cross(u);

          const Vector3
              sample_d = CosineWeightedSampleOnHemisphere(rng.Uniform(),
                                                          rng.Uniform());
          const Vector3 d = Normalize(sample_d.m_x * u + sample_d.m_y * v +
                                      sample_d.m_z * w);
          r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
          break;
        }
      }
    }
  }

  static void Render(
      Sphere* g_scene[], const size_t n_scene, // Geometry, Lights
      double Vx, double Vy, double Vz, // Params - Center of one sphere // must be Vector3()
      const std::uint32_t w, const std::uint32_t h, std::uint32_t nb_samples, size_t fr, // Camera
      Vector3 Ls[], const char* fileName // Result
    ) noexcept {

    RNG rng;

    // Camera params
    const Vector3 eye = {50.0, 52.0, 295.6};
    const Vector3 gaze = Normalize(Vector3(0.0, -0.042612, -1.0));
    const double fov = 0.5135;
    const Vector3 cx = {w * fov / h, 0.0, 0.0};
    const Vector3 cy = Normalize(cx.Cross(gaze)) * fov;

    // Change unfixed geometry center
    g_scene[6]->m_p = Vector3(Vx, Vy, Vz);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < static_cast<int>(h); ++y) { // pixel row
      for (std::size_t x = 0u; x < w; ++x) { // pixel column
        for (std::size_t sy = 0u, i = (h - 1u - y) * w + x; sy < 2u; ++sy) { // 2 subpixel row
          for (std::size_t sx = 0u; sx < 2u; ++sx) { // 2 subpixel column
            Vector3 L;

            for (std::size_t s = 0u; s < nb_samples; ++s) { // samples per subpixel
              const double u1 = 2.0 * rng.Uniform();
              const double u2 = 2.0 * rng.Uniform();
              const double dx = u1 < 1.0 ? sqrt(u1) - 1.0 : 1.0 - sqrt(2.0 - u1);
              const double dy = u2 < 1.0 ? sqrt(u2) - 1.0 : 1.0 - sqrt(2.0 - u2);
              const Vector3 d = cx * (((sx + 0.5 + dx) * 0.5 + x) / w - 0.5) +
                                cy * (((sy + 0.5 + dy) * 0.5 + y) / h - 0.5) +
                                gaze;

              L += Radiance(g_scene, n_scene, Ray(eye + d * 130.0, Normalize(d), EPSILON_SPHERE), rng) * (1.0 / nb_samples);
            }

            Ls[i] += 0.25 * Clamp(L);
          }
        }
      }
    }

    WritePPM(w, h, Ls, fileName);
  }
} // namespace smallpt

#ifndef CPP_EMBED_SMALLPT
using namespace smallpt;
int main(int argc, char* argv[]) {
  const std::uint32_t nb_samples = (2 == argc) ? atoi(argv[1]) / 4 : 1;
  const std::uint32_t w = 1024;
  const std::uint32_t h = 768;

  smallpt::Render(
    scene, *(&scene + 1) - scene, // Geometry, Lights
    27, 16.5, 47, // Params - Center of one sphere // must be Vector3()
    w, h, nb_samples, 0, // Camera
    new Vector3[w*h], "image.ppm" // Result
  );

  return 0;
}
#endif
