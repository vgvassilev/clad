//#define CLAD_SAFE_MATH

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

template <typename T, typename U>
ValueAndPushforward<T, U> make_value_and_pushforward(T value, U pushforward) {
  return {value, pushforward};
}
}