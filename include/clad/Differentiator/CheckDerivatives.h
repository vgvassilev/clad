//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
//------------------------------------------------------------------------------

/// Checking the derivatives clad generates against finite differences.
///
/// Porting a code base to AD leaves one question open at every call site:
/// whether the derivative clad produced is the derivative of the primal as
/// written. The numerical differentiation clad already carries answers it
/// approximately -- run both on the same inputs and compare:
///
/// \code
///   auto grad = clad::gradient(f);
///   clad::check_report r = clad::check_gradient(f, grad, x, y);
///   if (!r.ok())
///     r.print();
/// \endcode
///
/// Finite differences are an approximation, so a disagreement is a reason to
/// look rather than a proof of a wrong derivative. They are known to disagree
/// with a correct derivative where the primal is not smooth (`abs`, `min`,
/// `max`, a branch, `floor` or `round` near a step), where an input is badly
/// scaled for a fixed step, and where the primal has state or side effects,
/// since the stencil calls it several times. An integer or enum input takes
/// no small step at all; those are carried in the report as unchecked rather
/// than counted as failures.

#ifndef CLAD_CHECK_DERIVATIVES_H
#define CLAD_CHECK_DERIVATIVES_H

// Finite differences are the thing being compared against, so there is
// nothing to offer where they have been compiled out.
#ifndef CLAD_NO_NUM_DIFF

#include "Array.h"
#include "ArrayRef.h"
#include "FunctionTraits.h"
#include "NumericalDiff.h"
// clad::tape, for the tape numerical_diff writes the finite differences into.
// The use is through an alias template, which include-cleaner does not
// attribute to the header declaring it.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "Tape.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
// std::tuple and std::get, for the per-input storage the gradient writes
// through.
// NOLINTNEXTLINE(misc-include-cleaner)
#include <tuple>
#include <type_traits>
// std::vector, for the report's entries and the flattened derivatives.
// NOLINTNEXTLINE(misc-include-cleaner)
#include <vector>

namespace clad {

/// How closely a derivative has to agree with the finite difference beside it.
///
/// The two tolerances are combined as
/// `|ad - fd| <= abs_tolerance + rel_tolerance * |fd|`: the absolute term
/// carries derivatives near zero, where asking for a relative agreement asks
/// for more than a finite difference can give, and the relative term carries
/// large ones, where an absolute agreement asks the same.
struct check_options {
  double abs_tolerance = 1e-6;
  double rel_tolerance = 1e-5;
  /// Forward mode only: which input the derivative being checked is of.
  /// `clad::differentiate(f)` takes the first one, and so does this.
  std::size_t wrt = 0;
};

/// What one input's two derivatives came out as.
struct check_entry {
  /// Position of the parameter in the primal's signature.
  std::size_t arg = 0;
  /// Element within that parameter; 0 for a scalar input.
  std::size_t index = 0;
  /// What clad's derivative gave, and what the finite difference gave.
  double ad = 0;
  double fd = 0;
  /// Whether the two were compared at all. An integer or enum input cannot
  /// take a small step, so there is no finite difference to compare against.
  bool checked = false;
  /// Whether they agreed, within the tolerances. Always true for an input
  /// that was not checked -- it is not evidence either way.
  bool agrees = true;

  [[nodiscard]] double abs_error() const { return std::fabs(ad - fd); }
  /// Relative to the finite difference, which is the value being compared
  /// against. Zero there leaves nothing to be relative to, so the absolute
  /// error stands in and the absolute tolerance is what judges it.
  [[nodiscard]] double rel_error() const {
    double denom = std::fabs(fd);
    return denom == 0 ? abs_error() : abs_error() / denom;
  }
};

/// The result of checking a derivative: one entry per input, in the order the
/// primal takes them.
class check_report {
public:
  check_report() = default;
  explicit check_report(const check_options& options)
      : m_Entries(), m_Options(options) {}

  /// Whether every input that could be checked agreed.
  [[nodiscard]] bool ok() const { return m_Mismatches == 0; }
  /// How many inputs disagreed, and how many could not be checked.
  [[nodiscard]] std::size_t mismatches() const { return m_Mismatches; }
  [[nodiscard]] std::size_t unchecked() const { return m_Unchecked; }
  [[nodiscard]] const std::vector<check_entry>& entries() const {
    return m_Entries;
  }
  [[nodiscard]] const check_options& options() const { return m_Options; }

  /// Judge one input against the tolerances and keep it.
  void add(check_entry e) {
    if (e.checked) {
      e.agrees =
          e.abs_error() <=
          m_Options.abs_tolerance + (m_Options.rel_tolerance * std::fabs(e.fd));
      if (!e.agrees)
        ++m_Mismatches;
    } else {
      e.agrees = true;
      ++m_Unchecked;
    }
    m_Entries.push_back(e);
  }

  /// Report what disagreed, and what could not be checked, to \p out.
  ///
  /// A disagreement is printed with both derivatives and both errors, because
  /// which of the two is wrong is not something this can tell: a finite
  /// difference across a kink is wrong about a derivative clad has right.
  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg)
  void print(std::FILE* out = stdout) const {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const std::size_t total = m_Entries.size();
    if (ok() && m_Unchecked == 0) {
      std::fprintf(out,
                   "clad: %zu of %zu derivatives agree with finite "
                   "differences\n",
                   total, total);
      return;
    }
    if (ok())
      std::fprintf(out,
                   "clad: %zu of %zu derivatives agree with finite "
                   "differences, %zu not checked\n",
                   total - m_Unchecked, total, m_Unchecked);
    else
      std::fprintf(out,
                   "clad: %zu of %zu derivatives disagree with finite "
                   "differences (abs tol %g, rel tol %g)\n",
                   m_Mismatches, total, m_Options.abs_tolerance,
                   m_Options.rel_tolerance);
    for (const check_entry& e : m_Entries) {
      if (e.checked && e.agrees)
        continue;
      std::fprintf(out, "  arg %zu", e.arg);
      if (e.index != 0)
        std::fprintf(out, "[%zu]", e.index);
      if (!e.checked) {
        std::fprintf(out, ": not checked, an integer or enum input takes no "
                          "step\n");
        continue;
      }
      std::fprintf(out, ": ad %g, fd %g, abs %g, rel %g\n", e.ad, e.fd,
                   e.abs_error(), e.rel_error());
    }
    if (!ok())
      std::fprintf(out, "  which of the two is wrong is not decided here: a "
                        "finite difference across a kink, or of a badly "
                        "scaled input, disagrees with a correct derivative\n");
  }
  // NOLINTEND(cppcoreguidelines-pro-type-vararg)

private:
  std::vector<check_entry> m_Entries{};
  check_options m_Options;
  std::size_t m_Mismatches = 0;
  std::size_t m_Unchecked = 0;
};

namespace check_detail {

/// Whether a finite difference can step an input of this type. An integer or
/// an enum cannot take a small step: perturbing one either changes nothing or
/// changes everything, so there is no finite difference to compare the
/// derivative of such an input against.
template <typename T> struct steppable : std::is_floating_point<T> {};

/// Where one input's derivative is collected from the gradient, and what it
/// contributes to the comparison. An input the primal takes by value or by
/// reference has a single derivative, written through a pointer to its own
/// type -- which is what clad gives the gradient as that input's adjoint
/// parameter.
template <typename T> class slot {
  using value_type = std::decay_t<T>;
  value_type m_Adjoint = value_type();

public:
  explicit slot(const T& /*input*/) {}
  value_type* adjoint() { return &m_Adjoint; }
  [[nodiscard]] std::size_t size() const { return 1; }
  [[nodiscard]] double at(std::size_t /*i*/) const {
    return static_cast<double>(m_Adjoint);
  }
  /// An integer or enum input cannot take a small step: perturbing it
  /// either changes nothing or changes everything, so there is no finite
  /// difference to compare its derivative against.
  static bool checkable() { return steppable<value_type>::value; }
};

/// An array input, whose length the array_ref carries. One derivative per
/// element, written through a pointer to the first, as clad's gradient
/// expects.
template <typename T> class slot<array_ref<T>> {
  array<T> m_Adjoint;

public:
  explicit slot(const array_ref<T>& input) : m_Adjoint(input.size()) {}
  T* adjoint() { return m_Adjoint.ptr(); }
  [[nodiscard]] std::size_t size() const { return m_Adjoint.size(); }
  [[nodiscard]] double at(std::size_t i) const {
    return static_cast<double>(m_Adjoint[static_cast<std::ptrdiff_t>(i)]);
  }
  static bool checkable() { return steppable<T>::value; }
};

/// False, and dependent on T, so that a static_assert in a template fires
/// when the template is instantiated rather than when it is parsed.
template <typename T> struct never : std::false_type {};

/// A bare pointer says where an array starts and not how long it is, and
/// the check needs the length to know how many derivatives to compare.
template <typename T> class slot<T*> {
  static_assert(never<T>::value,
                "pass an array input as clad::array_ref<T>(ptr, n): a "
                "pointer alone does not say how many derivatives to check");
};

/// Whether a parameter is one a finite difference can perturb. The stencil
/// calls the primal with perturbed copies, which a parameter bound by
/// non-const reference has nowhere to put -- numerical_diff does not
/// compile for such a primal, and saying so here beats saying it from
/// inside the stencil.
template <typename... Ts> struct perturbable;
template <> struct perturbable<> : std::true_type {};
template <typename T, typename... Rest>
struct perturbable<T, Rest...>
    : std::integral_constant<
          bool, !(std::is_lvalue_reference<T>::value &&
                  !std::is_const<std::remove_reference_t<T>>::value) &&
                    perturbable<Rest...>::value> {};

template <typename L> struct perturbable_inputs;
template <typename... Ts>
struct perturbable_inputs<list<Ts...>> : perturbable<Ts...> {};

/// The parameters of \p F, whether it is a function, a pointer to one, or
/// something with an operator().
template <typename F>
struct primal_inputs : perturbable_inputs<argument_types_t<std::decay_t<F>>> {};

/// What the primal and the derivative are called with. An array_ref stands
/// for the pointer the primal declares; everything else goes as it is.
template <typename T> T& pass(T& v) { return v; }
template <typename T> T* pass(array_ref<T>& a) { return a.ptr(); }

/// The scalar value of an input, for the step size a forward difference
/// picks. Zero for an input that has no single value; such an input is
/// never the one being differentiated here.
template <typename T> double value_of(const T& v, std::true_type) {
  return static_cast<double>(v);
}
template <typename T> double value_of(const T& /*v*/, std::false_type) {
  return 0;
}
template <typename T> double value_of(const T& v) {
  return value_of(v, std::is_arithmetic<T>{});
}

/// Flatten one input's slot into the parallel arrays the comparison walks.
template <typename S>
void collect(const S& s, std::vector<double>& ad,
             std::vector<std::size_t>& lengths, std::vector<char>& checkable) {
  lengths.push_back(s.size());
  checkable.push_back(S::checkable() ? 1 : 0);
  for (std::size_t i = 0; i < s.size(); ++i)
    ad.push_back(s.at(i));
}

/// Run the gradient with each input's adjoint pointer appended, which is the
/// signature clad gives it: `f_grad(args..., _d_args...)`.
template <typename CladFn, typename Tuple, std::size_t... Is, typename... Args>
void run_gradient(const CladFn& grad, Tuple& slots, IndexSequence<Is...>,
                  Args&... args) {
  grad.execute(pass(args)..., std::get<Is>(slots).adjoint()...);
}

template <typename Tuple, std::size_t... Is>
void collect_all(const Tuple& slots, IndexSequence<Is...>,
                 std::vector<double>& ad, std::vector<std::size_t>& lengths,
                 std::vector<char>& checkable) {
  // The expansion is for its effects; the array is what gives it somewhere
  // to happen in order.
  const std::array<int, sizeof...(Is) + 1> sequence = {
      0, (collect(std::get<Is>(slots), ad, lengths, checkable), 0)...};
  (void)sequence;
}

template <std::size_t... Is, typename... Args>
void collect_values(std::vector<double>& values, IndexSequence<Is...>,
                    const Args&... args) {
  const std::array<int, sizeof...(Is) + 1> sequence = {
      0, (values.push_back(value_of(args)), 0)...};
  (void)sequence;
}

} // namespace check_detail

/// Compare the gradient clad generated for \p f against central differences of
/// \p f itself, at \p args.
///
/// The inputs are taken by value: the stencil calls the primal several times
/// per input, and one call's writes must not reach the next. An array input is
/// passed as `clad::array_ref<T>(ptr, n)`, which says how many derivatives it
/// has; its elements are the caller's and are not copied.
template <typename F, typename CladFn, typename... Args>
check_report check_gradient(const check_options& options, F f,
                            const CladFn& grad, Args... args) {
  static_assert(sizeof...(Args) != 0,
                "a gradient of no inputs has nothing to check");
  static_assert(check_detail::primal_inputs<F>::value,
                "a finite difference calls the primal with perturbed copies "
                "of its inputs, so every input has to be taken by value or by "
                "const reference");
  using sequence = MakeIndexSequence<sizeof...(Args)>;

  // What clad says. The gradient accumulates into its adjoints, so they start
  // at zero, which is how every slot is built.
  std::tuple<check_detail::slot<std::decay_t<Args>>...> slots{
      check_detail::slot<std::decay_t<Args>>(args)...};
  check_detail::run_gradient(grad, slots, sequence{}, args...);

  std::vector<double> ad;
  std::vector<std::size_t> lengths;
  std::vector<char> checkable;
  check_detail::collect_all(slots, sequence{}, ad, lengths, checkable);

  // What the central differences say. numerical_diff reads the length of each
  // input from the tape entry standing for it, so the two sides agree on the
  // shape of the answer by construction.
  array<double> fd(ad.size());
  tape<array_ref<double>> fdTape = {};
  std::size_t offset = 0;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    fdTape.emplace_back(fd.ptr() + offset, lengths[i]);
    offset += lengths[i];
  }
  numerical_diff::central_difference(f, fdTape, /*printErrors=*/false,
                                     check_detail::pass(args)...);

  check_report report(options);
  offset = 0;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    for (std::size_t j = 0; j < lengths[i]; ++j) {
      check_entry e;
      e.arg = i;
      e.index = j;
      e.ad = ad[offset + j];
      e.fd = fd[static_cast<std::ptrdiff_t>(offset + j)];
      e.checked = checkable[i] != 0;
      report.add(e);
    }
    offset += lengths[i];
  }
  return report;
}

template <typename F, typename CladFn, typename... Args>
check_report check_gradient(F f, const CladFn& grad, Args... args) {
  return check_gradient(check_options(), f, grad, args...);
}

/// Compare the forward-mode derivative clad generated for \p f against a
/// central difference of \p f itself, at \p args.
///
/// `options.wrt` is the input the derivative is of, and has to be the one
/// `clad::differentiate` was asked for; it defaults to the first, as
/// `clad::differentiate(f)` does. An array input is checked through
/// check_gradient instead, which has a length to work from.
template <typename F, typename CladFn, typename... Args>
check_report check_derivative(const check_options& options, F f,
                              const CladFn& derivative, Args... args) {
  static_assert(sizeof...(Args) != 0,
                "a derivative of no inputs has nothing to check");
  static_assert(check_detail::primal_inputs<F>::value,
                "a finite difference calls the primal with perturbed copies "
                "of its inputs, so every input has to be taken by value or by "
                "const reference");
  using sequence = MakeIndexSequence<sizeof...(Args)>;

  std::vector<double> values;
  check_detail::collect_values(values, sequence{}, args...);

  check_report report(options);
  check_entry e;
  e.arg = options.wrt;
  if (options.wrt >= sizeof...(Args)) {
    // Nothing to check against: the report says so rather than comparing
    // against an input that is not there.
    report.add(e);
    return report;
  }

  e.ad = static_cast<double>(derivative.execute(check_detail::pass(args)...));
  e.fd = numerical_diff::forward_central_difference(
      f, values[options.wrt], options.wrt, /*printErrors=*/false,
      check_detail::pass(args)...);
  // The same rule as a gradient's: an input that takes no small step has no
  // finite difference to be compared against.
  const std::array<bool, sizeof...(Args)> steppable = {
      check_detail::steppable<std::decay_t<Args>>::value...};
  e.checked = steppable[options.wrt];
  report.add(e);
  return report;
}

template <typename F, typename CladFn, typename... Args>
check_report check_derivative(F f, const CladFn& derivative, Args... args) {
  return check_derivative(check_options(), f, derivative, args...);
}

} // namespace clad

#endif // CLAD_NO_NUM_DIFF
#endif // CLAD_CHECK_DERIVATIVES_H
