#pragma once

#include <algorithm>
#include <cassert> // Needed for assert() macro
#include <cmath>   // Needed for pow()
#include <cstdio>  // Needed for printf()
#include <cstdlib> // Needed for exit() and ato*()
#include <random>

/** Zipf-like random distribution.
 *
 * "Rejection-inversion to generate variates from monotone discrete
 * distributions", Wolfgang Hörmann and Gerhard Derflinger
 * ACM TOMACS 6.3 (1996): 169-184
 */
template <class IntType = unsigned long, class RealType = double> class zipf_distribution {
public:
  typedef RealType input_type;
  typedef IntType result_type;

  static_assert(std::numeric_limits<IntType>::is_integer, "");
  static_assert(!std::numeric_limits<RealType>::is_integer, "");

  zipf_distribution(const IntType n = std::numeric_limits<IntType>::max(), const RealType q = 1.0)
      : n(n), q(q), H_x1(H(1.5) - 1.0), H_n(H(n + 0.5)), dist(H_x1, H_n) {}

  IntType operator()(std::mt19937& rng) {
    while(true) {
      const RealType u = dist(rng);
      const RealType x = H_inv(u);
      const IntType k = clamp<IntType>(std::round(x), 1, n);
      if(u >= H(k + 0.5) - h(k)) {
        return k;
      }
    }
  }

private:
  /** Clamp x to [min, max]. */
  template <typename T> static constexpr T clamp(const T x, const T min, const T max) {
    return std::max(min, std::min(max, x));
  }

  /** exp(x) - 1 / x */
  static double expxm1bx(const double x) {
    return (std::abs(x) > epsilon) ? std::expm1(x) / x
                                   : (1.0 + x / 2.0 * (1.0 + x / 3.0 * (1.0 + x / 4.0)));
  }

  /** H(x) = log(x) if q == 1, (x^(1-q) - 1)/(1 - q) otherwise.
   * H(x) is an integral of h(x).
   *
   * Note the numerator is one less than in the paper order to work with all
   * positive q.
   */
  const RealType H(const RealType x) {
    const RealType log_x = std::log(x);
    return expxm1bx((1.0 - q) * log_x) * log_x;
  }

  /** log(1 + x) / x */
  static RealType log1pxbx(const RealType x) {
    return (std::abs(x) > epsilon) ? std::log1p(x) / x
                                   : 1.0 - x * ((1 / 2.0) - x * ((1 / 3.0) - x * (1 / 4.0)));
  }

  /** The inverse function of H(x) */
  const RealType H_inv(const RealType x) {
    const RealType t = std::max(-1.0, x * (1.0 - q));
    return std::exp(log1pxbx(t) * x);
  }

  /** That hat function h(x) = 1 / (x ^ q) */
  const RealType h(const RealType x) { return std::exp(-q * std::log(x)); }

  static constexpr RealType epsilon = 1e-8;

  IntType n;                                     ///< Number of elements
  RealType q;                                    ///< Exponent
  RealType H_x1;                                 ///< H(x_1)
  RealType H_n;                                  ///< H(n)
  std::uniform_real_distribution<RealType> dist; ///< [H(x_1), H(n)]
};

//----- Constants -----------------------------------------------------------
#define FALSE 0 // Boolean false //NOLINT
#define TRUE 1  // Boolean true //NOLINT

//----- Function prototypes -------------------------------------------------
[[maybe_unused]] int zipf(double alpha, int n); // Returns a Zipf random variable
[[maybe_unused]] double rand_val(int seed);     // Jain's RNG

//===========================================================================
//=  Function to generate Zipf (power law) distributed random variables     =
//=    - Input: alpha and N                                                 =
//=    - Output: Returns with Zipf distributed random variable              =
//===========================================================================
[[maybe_unused]] inline int zipf(double alpha, int n) {
  static int first = TRUE;  // Static first time flag
  static double c = 0;      // Normalization constant
  static double* sum_probs; // Pre-calculated sum of probabilities //NOLINT
  double z;                 // Uniform random number (0 < z < 1) //NOLINT
  int zipf_value = 0;       // Computed exponential value to be returned //NOLINT
  int i;                    // Loop counter //NOLINT
  int low, high, mid;       // Binary-search bounds //NOLINT

  // Compute normalization constant on first call only
  if(first == TRUE) {
    for(i = 1; i <= n; i++) {
      c = c + (1.0 / pow((double)i, alpha));
    }
    c = 1.0 / c;

    sum_probs = static_cast<double*>(malloc((n + 1) * sizeof(*sum_probs))); // NOLINT
    sum_probs[0] = 0;
    for(i = 1; i <= n; i++) {
      sum_probs[i] = sum_probs[i - 1] + c / pow((double)i, alpha);
    }
    first = FALSE;
  }

  // Pull a uniform random number (0 < z < 1)
  do {
    z = rand_val(0);
  } while((z == 0) || (z == 1));

  // Map z to the value
  low = 1, high = n, mid;
  do {
    mid = floor((low + high) / 2); // NOLINT
    if(sum_probs[mid] >= z && sum_probs[mid - 1] < z) {
      zipf_value = mid;
      break;
    } else if(sum_probs[mid] >= z) { // NOLINT
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  } while(low <= high); // NOLINT

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >= 1) && (zipf_value <= n)); // NOLINT

  return (zipf_value);
}

//=========================================================================
//= Multiplicative LCG for generating uniform(0.0, 1.0) random numbers    =
//=   - x_n = 7^5*x_(n-1)mod(2^31 - 1)                                    =
//=   - With x seeded to 1 the 10000th x value should be 1043618065       =
//=   - From R. Jain, "The Art of Computer Systems Performance Analysis," =
//=     John Wiley & Sons, 1991. (Page 443, Figure 26.2)                  =
//=========================================================================
[[maybe_unused]] inline double rand_val(int seed) {
  const long a = 16807;      // Multiplier
  const long m = 2147483647; // Modulus
  const long q = 127773;     // m div a
  const long r = 2836;       // m mod a
  static long x;             // Random int value
  long x_div_q;              // x divided by q //NOLINT
  long x_mod_q;              // x modulo q     //NOLINT
  long x_new;                // New x value    //NOLINT

  // Set the seed if argument is non-zero and then return zero
  if(seed > 0) {
    x = seed;
    return (0.0);
  }

  // RNG using integer arithmetic
  x_div_q = x / q;
  x_mod_q = x % q;
  x_new = (a * x_mod_q) - (r * x_div_q);
  if(x_new > 0) {
    x = x_new;
  } else {
    x = x_new + m;
  }

  // Return a random value between 0.0 and 1.0
  return ((double)x / m);
}