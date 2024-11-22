#include <cmath>

extern "C" double CND(double d);

double d1(double S, double X, double T) {
  return (log(S / X) + (RISKFREE + 0.5 * VOLATILITY * VOLATILITY) * T) /
         (VOLATILITY * sqrt(T));
}

double N_prime(double d) {
  const double RSQRT2PI =
      0.39894228040143267793994605993438; // 1 / sqrt(2 * PI)
  return RSQRT2PI * exp(-0.5 * d * d);
}

enum Greek { Delta, dX, Theta };

enum OptionType { Call, Put };

template <OptionType opt> const char* getNameofOpt() {
  if constexpr (opt == Call)
    return "Call";
  if constexpr (opt == Put)
    return "Put";
}

template <Greek greek> const char* getNameOfGreek() {
  if constexpr (greek == Delta)
    return "Delta";
  if constexpr (greek == dX)
    return "dStrike";
  if constexpr (greek == Theta)
    return "Theta";
}

template <OptionType opt, Greek greek>
void computeL1norm(float* S, float* X, float* T, float* d) {
  double delta, ref, sum_delta, sum_ref;
  sum_delta = 0;
  sum_ref = 0;
  for (int i = 0; i < OPT_N; i++) {
    if constexpr (opt == Call) {
      if constexpr (greek == Delta) {
        double d1_val = d1(S[i], X[i], T[i]);
        ref = CND(d1_val);
      } else if constexpr (greek == dX) {
        double T_val = T[i];
        double d1_val = d1(S[i], X[i], T_val);
        double d2_val = d1_val - VOLATILITY * sqrt(T_val);
        double expRT = exp(-RISKFREE * T_val);
        ref = -expRT * CND(d2_val);
      } else if constexpr (greek == Theta) {
        double S_val = S[i], X_val = X[i], T_val = T[i];
        double d1_val = d1(S_val, X_val, T_val);
        double d2_val = d1_val - VOLATILITY * sqrt(T_val);
        double expRT = exp(-RISKFREE * T_val);
        ref = (S_val * N_prime(d1_val) * VOLATILITY) / (2 * sqrt(T_val)) +
              RISKFREE * X_val * expRT *
                  CND(d2_val); // theta is with respect to t, so -theta is the
                               // approximation of the derivative with respect
                               // to T
      }
    } else if constexpr (opt == Put) {
      if constexpr (greek == Delta) {
        double d1_val = d1(S[i], X[i], T[i]);
        ref = CND(d1_val) - 1.0;
      } else if constexpr (greek == dX) {
        double T_val = T[i];
        double d1_val = d1(S[i], X[i], T_val);
        double d2_val = d1_val - VOLATILITY * sqrt(T_val);
        double expRT = exp(-RISKFREE * T_val);
        ref = expRT * CND(-d2_val);
      } else if constexpr (greek == Theta) {
        double S_val = S[i], X_val = X[i], T_val = T[i];
        double d1_val = d1(S_val, X_val, T_val);
        double d2_val = d1_val - VOLATILITY * sqrt(T_val);
        double expRT = exp(-RISKFREE * T_val);
        ref = (S_val * N_prime(d1_val) * VOLATILITY) / (2 * sqrt(T_val)) -
              RISKFREE * X_val * expRT * CND(-d2_val);
      }
    }

    delta = fabs(d[i] - ref);
    sum_delta += delta;
    sum_ref += fabs(ref);
  }

  double L1norm = sum_delta / sum_ref;
  printf("L1norm of %s for %s option = %E\n", getNameOfGreek<greek>(),
         getNameofOpt<opt>(), L1norm);
  if (L1norm > 1e-5) {
    printf(
        "Gradient test failed: Difference between %s's computed and "
        "approximated theoretical values for %s option is larger than expected",
        getNameOfGreek<greek>(), getNameofOpt<opt>());
    exit(EXIT_FAILURE);
  }
}
