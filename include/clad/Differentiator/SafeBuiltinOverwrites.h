#ifndef CLAD_SAFE_BUILTIN_OVERRIDES_H
#define CLAD_SAFE_BUILTIN_OVERRIDES_H

#ifndef CLAD_SAFE_MATH // Must be defined here or when compiling to enable safe math functions and pushforwards. 
#define CLAD_SAFE_MATH //It is not defined by default to avoid performance overhead in cases where it is not needed.
#endif

#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>

namespace clad {

template <typename T, typename U> struct ValueAndPushforward {
  T value;
  U pushforward;

  // Define the cast operator from ValueAndPushforward<T, U> to
  // ValueAndPushforward<V, w> where V is convertible to T and W is
  // convertible to U.
  template <typename V = T, typename W = U>
  operator ValueAndPushforward<V, W>() const {
    return {static_cast<V>(value), static_cast<W>(pushforward)};
  }
};
// Warning helpers

#ifdef CLAD_SAFE_MATH

inline void warn_domain(const char* func, double x) {
    std::fprintf(
        stderr,
        "[CLAD_SAFE_MATH] Domain violation in %s(x=%f)\n",
        func,
        x
    );
}

inline void warn_domain2(const char* func, double x, double y) {
    std::fprintf(
        stderr,
        "[CLAD_SAFE_MATH] Domain violation in %s(x=%f, y=%f)\n",
        func,
        x,
        y
    );
}

#endif

// Safe basic functions

inline double safe_sqrt(double x) {
#ifdef CLAD_SAFE_MATH
    if (x < 0.0) {
        warn_domain("sqrt", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::sqrt(x);
}

inline double safe_log(double x) {
#ifdef CLAD_SAFE_MATH
    if (x <= 0.0) {
        warn_domain("log", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::log(x);
}

inline double safe_log1p(double x) {
#ifdef CLAD_SAFE_MATH
    if (x <= -1.0) {
        warn_domain("log1p", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::log1p(x);
}

inline double safe_log10(double x) {
#ifdef CLAD_SAFE_MATH
    if (x <= 0.0) {
        warn_domain("log10", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::log10(x);
}

inline double safe_log2(double x) {
#ifdef CLAD_SAFE_MATH
    if (x <= 0.0) {
        warn_domain("log2", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::log2(x);
}

inline double safe_acos(double x) {
#ifdef CLAD_SAFE_MATH
    if (x < -1.0 || x > 1.0) {
        warn_domain("acos", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::acos(x);
}

inline double safe_asin(double x) {
#ifdef CLAD_SAFE_MATH
    if (x < -1.0 || x > 1.0) {
        warn_domain("asin", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::asin(x);
}

inline double safe_atanh(double x) {
#ifdef CLAD_SAFE_MATH
    if (x <= -1.0 || x >= 1.0) {
        warn_domain("atanh", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::atanh(x);
}

inline double safe_acosh(double x) {
#ifdef CLAD_SAFE_MATH
    if (x < 1.0) {
        warn_domain("acosh", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return std::acosh(x);
}

inline double safe_div(double x, double y) {
#ifdef CLAD_SAFE_MATH
    if (y == 0.0) {
        warn_domain2("division", x, y);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif
    return x / y;
}

//Pushforwards

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_sqrt_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x < 0) {
        warn_domain("sqrt_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    T val = std::sqrt(x);

    return {
        val,
        (1.0 / (2.0 * val)) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_log_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0) {
        warn_domain("log_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {
        std::log(x),
        (1.0 / x) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_log10_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0) {
        warn_domain("log10_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {
        std::log10(x),
        (1.0 / (x * std::log(10))) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_log2_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0) {
        warn_domain("log2_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {
        std::log2(x),
        (1.0 / (x * std::log(2))) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_log1p_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= -1.0) {
        warn_domain("log1p_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {
        std::log1p(x),
        (1.0 / (x + 1.0)) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_acos_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x < -1 || x > 1) {
        warn_domain("acos_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {
        std::acos(x),
        (-1.0 / std::sqrt(1 - x*x)) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_asin_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x < -1 || x > 1) {
        warn_domain("asin_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {
        std::asin(x),
        (1.0 / std::sqrt(1 - x*x)) * d_x
    };
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_acosh_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x < 1) {
        warn_domain("acosh_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {::std::acosh(x), d_x / ::std::sqrt(x * x - 1)};
}

template<typename T, typename dT>
inline ValueAndPushforward<T,dT>
safe_atanh_pushforward(T x, dT d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= -1 || x >= 1) {
        warn_domain("atanh_pushforward", x);
        return {
            std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<dT>::quiet_NaN()
        };
    }
#endif

    return {::std::atanh(x), d_x / (1 - x * x)};
}

//Pullbacks

template<typename T, typename dT>
inline void safe_sqrt_pullback(T x, dT d_y, dT* d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0) {
        warn_domain("sqrt_pullback", x);
        *d_x += std::numeric_limits<dT>::quiet_NaN();
        return;
    }
#endif

    *d_x += (1.0 / (2.0 * std::sqrt(x))) * d_y;
}

template<typename T, typename dT>
inline void safe_log_pullback(T x, dT d_y, dT* d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0) {
        warn_domain("log_pullback", x);
        *d_x += std::numeric_limits<dT>::quiet_NaN();
        return;
    }
#endif

    *d_x += (1.0 / x) * d_y;
}

template<typename T, typename dT>
inline void safe_acos_pullback(T x, dT d_y, dT* d_x) {

#ifdef CLAD_SAFE_MATH
    if (x < -1 || x > 1) {
        warn_domain("acos_pullback", x);
        *d_x += std::numeric_limits<dT>::quiet_NaN();
        return;
    }
#endif

    *d_x += (-1.0 / std::sqrt(1 - x*x)) * d_y;
}

template<typename T, typename dT>
inline void safe_asin_pullback(T x, dT d_y, dT* d_x) {

#ifdef CLAD_SAFE_MATH
    if (x < -1 || x > 1) {
        warn_domain("asin_pullback", x);
        *d_x += std::numeric_limits<dT>::quiet_NaN();
        return;
    }
#endif

    *d_x += (1.0 / std::sqrt(1 - x*x)) * d_y;
}

template<typename T, typename dT>
inline void safe_atanh_pullback(T x, dT d_y, dT* d_x) {

#ifdef CLAD_SAFE_MATH
    if (x <= -1 || x >= 1) {
        warn_domain("atanh_pullback", x);
        *d_x += std::numeric_limits<dT>::quiet_NaN();
        return;
    }
#endif

    *d_x += (1.0 / (1 - x*x)) * d_y;
}

// Safe derivatives

inline double safe_sqrt_derivative(double x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0.0) {
        warn_domain("sqrt derivative", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif

    return 1.0 / (2.0 * std::sqrt(x));
}

inline double safe_log_derivative(double x) {

#ifdef CLAD_SAFE_MATH
    if (x <= 0.0) {
        warn_domain("log derivative", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif

    return 1.0 / x;
}

inline double safe_acos_derivative(double x) {

#ifdef CLAD_SAFE_MATH
    if (x <= -1.0 || x >= 1.0) {
        warn_domain("acos derivative", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif

    return -1.0 / std::sqrt(1 - x*x);
}

inline double safe_asin_derivative(double x) {

#ifdef CLAD_SAFE_MATH
    if (x <= -1.0 || x >= 1.0) {
        warn_domain("asin derivative", x);
        return std::numeric_limits<double>::quiet_NaN();
    }
#endif

    return 1.0 / std::sqrt(1 - x*x);
}

} 

#endif
