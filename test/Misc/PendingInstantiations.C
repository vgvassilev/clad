// RUN: %cladclang -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// Clad calls Sema::PerformPendingInstantiations before starting to build
// derivatives. That is problematic generally and this example demonstrates that
// when instantiating member functions. There we call the listener and Clad
// gets informed about a new function upon which it forces instantiating the
// pending (due to the ongoing instantiation) instantiations. That breaks the
// language semantics and results in 'error: incomplete type' errors.

template <int a> struct b {
  static constexpr int c = 1;
  typedef int d;
  constexpr operator d() const { return c; }
};
template <int> struct h { };
constexpr b<true> f(bool) { return {}; }
template <typename g> struct j : b<0> {
  static_assert(f(g{}), "");
};
template <typename, typename i> struct k {
  void operator=(h<j<i>::c>);
};
namespace clad {}
template <typename> struct l {
  using ap = int;
};
template <typename T> struct n {
  k<typename l<T>::ap, bool> m_fn1() { return {}; }
};
void bb() {
  n<char> a;
  a.m_fn1();
}
