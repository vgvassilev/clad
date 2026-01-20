// RUN: %cladclang -std=c++20 -ostl_cmath.o -I%S/../../include %s 2>&1 | %filecheck %s
// RUN: ./stl_cmath.o | %filecheck_exec %s
// XFAIL: valgrind
// CHECK-EXEC-NOT: FAIL

// This test checks if the functions available in <cmath> are supported by clad.
//----------------------------------- LEGEND -----------------------------------
// D - Differentiable
// N - Differentiation is either undefined (discontinuous ops) or not practical.
// M - Function produces classification/bit-level results and differentiation is
//     conceptually meaningless
// S - Supported in Clad
// P - Partially supported in Clad
//
//---------------------------- Absolute value ----------------------------------
// DS abs/ fabs/ fabsf/ fabsl           (C++11) absolute value |x|
//
//---------------------- Floating-point remainder ------------------------------
// D fmod/ fmodf/ fmodl                (C++11) remainder of division
// D remainder/ remainderf/ remainderl (C++11) signed remainder
// N remquo/ remquof/ remquol          (C++11) remainder + quotient bits
//
//-------------------------- Fused/Min/Max/Diff/NaN ----------------------------
// DS fma/ fmaf/ fmal                   (C++11) fused multiply-add
// DS fmax/ fmaxf/ fmaxl                (C++11) larger of two values
// DS fmin/ fminf/ fminl                (C++11) smaller of two values
// DS fdim/ fdimf/ fdiml                (C++11) positive difference max(0,x−y)
// N nan/ nanf/ nanl                   (C++11) generate NaN
// D lerp                              (C++20) linear interpolation
//
//-------------------------- Exponential / Logarithmic -------------------------
// DS exp/ expf/ expl                   (C++11) e^x
// DS exp2/ exp2f/ exp2l                (C++11) 2^x
// DS expm1/ expm1f/ expm1l             (C++11) e^x − 1
// DS log/ logf/ logl                   (C++11) natural log ln(x)
// DS log10/ log10f/ log10l             (C++11) log base 10
// DS log2/ log2f/ log2l                (C++11) log base 2
// DS log1p/ log1pf/ log1pl             (C++11) ln(1+x)
//
//----------------------------- Power / Roots ----------------------------------
// DS pow/ powf/ powl                  (C++11) x^y
// DS sqrt/ sqrtf/ sqrtl               (C++11) square root
// DS cbrt/ cbrtf/ cbrtl               (C++11) cube root
// DS hypot/ hypotf/ hypotl            (C++11) sqrt(x^2+y^2) [+ z^2 since C++17]
//
//---------------------------- Trigonometric -----------------------------------
// DS sin/ sinf/ sinl                   (C++11) sine
// DS cos/ cosf/ cosl                   (C++11) cosine
// DS tan/ tanf/ tanl                   (C++11) tangent
// DS asin/ asinf/ asinl                (C++11) arc-sine
// DS acos/ acosf/ acosl                (C++11) arc-cosine
// DS atan/ atanf/ atanl                (C++11) arc-tangent
// DS atan2/ atan2f/ atan2l             (C++11) quadrant-aware arc-tangent
//
//----------------------------- Hyperbolic -------------------------------------
// DS sinh/ sinhf/ sinhl                (C++11) hyperbolic sine
// DS cosh/ coshf/ coshl                (C++11) hyperbolic cosine
// DS tanh/ tanhf/ tanhl                (C++11) hyperbolic tangent
// DS asinh/ asinhf/ asinhl             (C++11) inverse hyperbolic sine
// DP acosh/ acoshf/ acoshl             (C++11) inverse hyperbolic cosine
// DS atanh/ atanhf/ atanhl             (C++11) inverse hyperbolic tangent
//
//------------------------ Error / Gamma functions -----------------------------
// D erf/ erff/ erfl                   (C++11) error function
// D erfc/ erfcf/ erfcl                (C++11) complementary error
// D tgamma/ tgammaf/ tgammal          (C++11) gamma function
// D lgamma/ lgammaf/ lgammal          (C++11) log gamma
//
//------------------------ Elliptic integrals -----------------------------
// D comp_ellint_1 / comp_ellint_1f / comp_ellint_1l (C++17) complete elliptic integral of the first kind
// D comp_ellint_2 / comp_ellint_2f / comp_ellint_2l (C++17) complete elliptic integral of the second kind
// D comp_ellint_3 (C++17) complete elliptic integral of the third kind
//
//---------------------- Nearest integer operations ----------------------------
// N ceil/ ceilf/ ceill                (C++11) smallest integer >= x
// N floor/ floorf/ floorl             (C++11) largest integer <= x
// N trunc/ truncf/ truncl             (C++11) nearest integer toward 0
// N round/ roundf/ roundl ...         (C++11) round away from 0
// N nearbyint/ nearbyintf/ nearbyintl (C++11) round using current mode
// N rint/ rintf/ rintl ...            (C++11) round using mode (may trap)
//
//--------------- Floating-point decomposition / manipulation ------------------
// M frexp/ frexpf/ frexpl             (C++11) split into mantissa & exponent
// M ldexp/ ldexpf/ ldexpl             (C++11) x * 2^exp
// M modf/ modff/ modfl                (C++11) split int & frac
// M scalbn/scalbnf/scalbnl ...        (C++11) x * FLT_RADIX^n
// M ilogb/ ilogbf/ ilogbl             (C++11) exponent as int
// M logb/ logbf/ logbl                (C++11) exponent as float
// M nextafter/ nexttoward ...         (C++11) next representable value
// M copysign/ copysignf/ copysignl    (C++11) copy sign
//
//------------- Classification / comparison (predicates) -----------------------
// M fpclassify                        (C++11) category of value
// M isfinite                          (C++11) finite?
// M isinf                             (C++11) infinite?
// M isnan                             (C++11) NaN?
// M isnormal                          (C++11) normal?
// M signbit                           (C++11) sign is negative?
// M isgreater/ isgreaterequal         (C++11) ordered comparisons
// M isless/ islessequal/ islessgreater (C++11) ordered comparisons
// M isunordered                       (C++11) unordered (NaN?)
//
//----------------------- Mathematical special functions -----------------------
// D assoc_laguerre / f / l    (C++17) associated Laguerre polynomials
// D assoc_legendre/ f / l     (C++17) associated Legendre polynomials
// D beta/ betaf/ betal        (C++17) beta function
// D comp_ellint_1/ f / l      (C++17) complete elliptic integral (1st kind)
// D comp_ellint_2/ f / l      (C++17) complete elliptic integral (2nd kind)
// D comp_ellint_3/ f / l      (C++17) complete elliptic integral (3rd kind)
// D cyl_bessel_i/ f / l       (C++17) modified cylindrical Bessel (regular)
// D cyl_bessel_j/ f / l       (C++17) cylindrical Bessel functions (1st kind)
// D cyl_bessel_k/ f / l       (C++17) modified cylindrical Bessel (irregular)
// D cyl_neumann/ f / l        (C++17) cylindrical Neumann functions
// D ellint_1/ f / l           (C++17) incomplete elliptic integral (1st kind)
// D ellint_2/ f / l           (C++17) incomplete elliptic integral (2nd kind)
// D ellint_3/ f / l           (C++17) incomplete elliptic integral (3rd kind)
// D expint/ expintf/ expintl  (C++17) exponential integral
// D hermite/ hermitef/ hermitel (C++17) Hermite polynomials
// D legendre/ legendref/ legendrel (C++17) Legendre polynomials
// D laguerre/ laguerref/ laguerrel (C++17) Laguerre polynomials
// D riemann_zeta/ f / l       (C++17) Riemann zeta function
// D sph_bessel/ f / l         (C++17) spherical Bessel functions (1st kind)
// D sph_legendre/ f / l       (C++17) spherical associated Legendre functions
// D sph_neumann/ f / l        (C++17) spherical Neumann functions

#include <clad/Differentiator/Differentiator.h>

#include <cmath>
#include <iostream>
#include <iomanip>

template <typename T>
T get_tolerance() {
    if constexpr (std::is_same_v<T,float>) return T(1e-3);
    else if constexpr (std::is_same_v<T,long double>) return T(1e-10);
    else return T(1e-6);
}

// step for numerical derivative
template <typename T>
T get_h() {
    if constexpr (std::is_same_v<T,float>) return T(1e-3);
    else if constexpr (std::is_same_v<T,long double>) return T(1e-8);
    else return T(1e-6);
}
// Central finite difference approximation
template<typename T>
T numerical_derivative(T (*f)(T), T x) {
    const T h = get_h<T>();
    return (f(x + h) - f(x - h)) / (2 * h);
}

// Test function
template<typename T>
void test_func(const std::string &name,
               T (*f)(T), const clad::CladFunction<T(*)(T)> &clad_fwd,
               const decltype(clad::gradient((T (*)(T))nullptr)) &clad_rev,
               const T (&test_points)[7] = {-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0}) {
    const T tol = get_tolerance<T>();
    std::cerr << std::setprecision(10);
    for (T x : test_points) {
        T dfwd_val = clad_fwd.execute(x);
        T drev_val = T();
        clad_rev.execute(x, &drev_val);
        T df_num = numerical_derivative(f, x);
        if (dfwd_val != drev_val || std::abs(dfwd_val - df_num) > tol) {
            std::cerr << "FAIL: " << name << " at x=" << x
                      << ": dfwd=" << dfwd_val
                      << ", drev=" << drev_val
                      << ", numerical=" << df_num << "\n";
        } else {
            std::cout << "PASS: " << name << " at x=" << x
                      << " dfwd=" << dfwd_val << ", drev=" << drev_val << "\n";
        }
    }
}

#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#define CHECK_F(X) test_func<float>(STRINGIFY(X), \
                           X, clad::differentiate(X), \
                           clad::gradient(X));
#define CHECK_D(X) test_func<double>(STRINGIFY(X), \
                           X, clad::differentiate(X), \
                           clad::gradient(X));
#define CHECK_LD(X) test_func<long double>(STRINGIFY(X), \
                           X, clad::differentiate(X), \
                           clad::gradient(X));
#define CHECK_RANGE_F(X, ...)                           \
  {                                                     \
    float arr[] = __VA_ARGS__;                          \
    test_func<float>(STRINGIFY(X),                      \
                     X, clad::differentiate(X),         \
                     clad::gradient(X), arr);           \
  }
#define CHECK_RANGE_D(X, ...)                           \
  {                                                     \
    double arr[] = __VA_ARGS__;                         \
    test_func<double>(STRINGIFY(X),                     \
                      X, clad::differentiate(X),        \
                      clad::gradient(X), arr);          \
  }
#define CHECK_RANGE_LD(X, ...)                          \
  {                                                     \
    long double arr[] = __VA_ARGS__;                    \
    test_func<long double>(STRINGIFY(X),                \
                           X, clad::differentiate(X),   \
                           clad::gradient(X), arr);     \
  }

#define CHECK(X)            CHECK_LD(f_##X<long double>);               \
                            CHECK_D(f_##X<double>);                     \
                            CHECK_F(f_##X<float>);
#define CHECK_RANGE(X, ...) CHECK_RANGE_F(f_##X<float>, __VA_ARGS__);       \
                            CHECK_RANGE_D(f_##X<double>, __VA_ARGS__);      \
                            CHECK_RANGE_LD(f_##X<long double>, __VA_ARGS__);

#define CHECK_ALL(X) CHECK(X); CHECK_F(f_##X##f); CHECK_LD(f_##X##l);
#define CHECK_ALL_RANGE(X, ...) CHECK_RANGE(X, __VA_ARGS__);            \
                                CHECK_RANGE_F(f_##X##f, __VA_ARGS__);   \
                                CHECK_RANGE_LD(f_##X##l, __VA_ARGS__)

#define DEFINE_FUNCTIONS(F)                                             \
  template <typename T>                                                 \
  T f_##F(T x) { return std::F(x); }                                    \
  inline float f_##F##f(float x) { return F##f(x); }                    \
  inline long double f_##F##l(long double x) { return F##l(x); }

// ----------------------- Differentiable functions ----------------------------
//
//---------------------------- Absolute value ----------------------------------
//
template<typename T> T f_abs(T x) { return std::abs(x); }

DEFINE_FUNCTIONS(fabs)
double f_fabs(double x) { return fabs(x); }
//
//-------------------------- Fused/Min/Max/Diff/NaN ----------------------------
//
// x in (-inf,+inf)discontinuous at x == y
template<typename T> T f_fmax(T x){ return std::fmax(x,(T)1.1); }
float f_fmaxf(float x){ return fmaxf(x,1.1f); }
long double f_fmaxl(long double x){ return fmaxl(x,1.0L); }
// x in (-inf,+inf)discontinuous at x == y
template<typename T> T f_fmin(T x){ return std::fmin(x,(T)1.1); }
float f_fminf(float x){ return fminf(x,1.0f); }
long double f_fminl(long double x){ return fminl(x,1.0L); }
// x in (-inf,+inf)discontinuous at x == y
template<typename T> T f_fdim(T x){ return std::fdim(x,(T)0.51); }
float f_fdimf(float x){ return fdimf(x,0.51f); }
long double f_fdiml(long double x){ return fdiml(x,0.51L); }
// x in (-inf,+inf)
template<typename T> T f_fma(T x){ return std::fma(x,(T)2.0,(T)1.0); }
float f_fmaf(float x){ return fmaf(x,2.0f,1.0f); }
long double f_fmal(long double x){ return fmal(x,2.0L, 1.0L); }
//
//-------------------------- Exponential / Logarithmic -------------------------
//
DEFINE_FUNCTIONS(exp)   // x in (-inf,+inf)
DEFINE_FUNCTIONS(exp2)  // x in (-inf,+inf)
DEFINE_FUNCTIONS(expm1) // x in (-inf,+inf)
DEFINE_FUNCTIONS(log)   // x in (0,+inf)
DEFINE_FUNCTIONS(log10) // x in (0,+inf)
DEFINE_FUNCTIONS(log2)  // x in (0,+inf)
DEFINE_FUNCTIONS(log1p) // x in (-1,+inf)
//
//----------------------------- Power / Roots ----------------------------------
//
template<typename T> T f_pow(T x){ return std::pow(x,2.0); } // x in (-inf,+inf)
float f_powf(float x){ return powf(x,2.0f); }
long double f_powl(long double x){ return powl(x,2.0L); }
DEFINE_FUNCTIONS(sqrt) // x in [0,+inf)
DEFINE_FUNCTIONS(cbrt) // x in (-inf,+inf)
template<typename T> T f_hypot(T x){ return std::hypot(x,(T)1.0); } // x in (-inf,+inf)
float f_hypotf(float x){ return hypotf(x,2.0f); }
long double f_hypotl(long double x){ return hypotl(x,2.0L); }
//
//---------------------------- Trigonometric -----------------------------------
//
DEFINE_FUNCTIONS(sin)  // x in (-inf,+inf)
DEFINE_FUNCTIONS(cos)  // x in (-inf,+inf)
DEFINE_FUNCTIONS(tan)  // x != pi/2 + k*pi
DEFINE_FUNCTIONS(asin) // x in [-1,1]
DEFINE_FUNCTIONS(acos) // x in [-1,1]
DEFINE_FUNCTIONS(atan) // x in (-inf,+inf)
template<typename T> T f_atan2(T x){ return std::atan2(x,(T)1.0); } // x in (-inf,+inf)
inline float f_atan2f(float x){ return atan2f(x, 1.0f); }
inline long double f_atan2l(long double x){ return atan2l(x, 1.0L); }
//
//----------------------------- Hyperbolic -------------------------------------
//
DEFINE_FUNCTIONS(sinh)  // x in (-inf,+inf)
DEFINE_FUNCTIONS(cosh)  // x in (-inf,+inf)
DEFINE_FUNCTIONS(tanh)  // x in (-inf,+inf)
DEFINE_FUNCTIONS(asinh) // x in (-inf,+inf)
DEFINE_FUNCTIONS(acosh) // x in [1,inf)
DEFINE_FUNCTIONS(atanh) // x in [-1,1]
//
//------------------------ Error / Gamma functions -----------------------------
//
DEFINE_FUNCTIONS(erf)  // x in (-inf,+inf)

//------------------------ Elliptic integrals -----------------------------
//
// Domain: k in (-1, 1)
template<typename T> T f_comp_ellint_1(T k) { return std::comp_ellint_1(k); }
float f_comp_ellint_1f(float k) { return std::comp_ellint_1(k); }
long double f_comp_ellint_1l(long double k) { return std::comp_ellint_1(k); }

// Domain: k in (-1, 1)
template<typename T> T f_comp_ellint_2(T k) { return std::comp_ellint_2(k); }
float f_comp_ellint_2f(float k) { return std::comp_ellint_2(k); }
long double f_comp_ellint_2l(long double k) { return std::comp_ellint_2(k); }

// Domain: k in (-1, 1). Fixed nu = 0.5 for testing.
template<typename T> T f_comp_ellint_3(T k) { return std::comp_ellint_3(k, (T)0.5); }
float f_comp_ellint_3f(float k) { return std::comp_ellint_3(k, 0.5f); }
long double f_comp_ellint_3l(long double k) { return std::comp_ellint_3(k, 0.5L); }


int main() {
  // Absolute value
  CHECK(abs);
  CHECK(fabs); CHECK_F(f_fabsf); CHECK_LD(f_fabsl);
  //CHECK_RANGE_D(f_fabs, {-2.0,-1.0,-0.5,0.01,0.5,1.0,2.0});

  // Fused/Min/Max/Diff/NaN
  CHECK_ALL(fmax);
  CHECK_ALL(fmin);
  CHECK_ALL(fdim);
  CHECK_ALL(fma);

  // Exponential / Logarithmic
  CHECK_ALL(exp);
  CHECK_ALL(exp2);
  CHECK_ALL(expm1);
  CHECK_ALL(log);
  CHECK_ALL(log10);
  CHECK_ALL(log2);
  CHECK_ALL(log1p);

  // Power / Roots
  CHECK_ALL(pow);
  CHECK_ALL_RANGE(sqrt, {0.01, 0.1, 0.25, 1.0, 2.0, 4.0, 9.0});
  // x=0 should return inf but numerical diff goes crazy.
  CHECK_ALL_RANGE(cbrt, {-2.0, -1.0, -0.5, 0.05, 0.5, 1.0, 2.0});
  CHECK_ALL(hypot);


  // Trigonometric
  CHECK_ALL(sin);
  CHECK_ALL(cos);
  CHECK_ALL(tan);

  // x in [-1,1]
  CHECK_ALL_RANGE(asin, {-0.95, -0.9, -0.5, 0.0, 0.5, 0.9, 0.95});

  // x in [-1,1]
  CHECK_ALL_RANGE(acos, {-0.95, -0.9, -0.5, 0.0, 0.5, 0.9, 0.95});

  CHECK_ALL(atan);
  CHECK_ALL(atan2);

  // Hyperbolic
  CHECK_ALL(sinh);
  CHECK_ALL(cosh);
  CHECK_ALL(tanh);
  CHECK_ALL(asinh);

  // x in [1,inf)
  CHECK_RANGE(acosh, {1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0});
  //CHECK_RANGE_F(acoshf, {1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0});
  //CHECK_RANGE_LD(acoshl, {1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0});
  CHECK_ALL(atanh);

  // Error / Gamma functions
  CHECK_ALL(erf);

  // Elliptic Integrals
  CHECK_ALL_RANGE(comp_ellint_1, -0.9, 0.9);
  CHECK_ALL_RANGE(comp_ellint_2, -0.9, 0.9);
  CHECK_ALL_RANGE(comp_ellint_3, -0.9, 0.9);

  return 0;
}