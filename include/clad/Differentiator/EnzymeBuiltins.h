//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// Enzyme activity markers and the carrier for its returned gradients.
//
// Enzyme recognises the markers by name and consumes the uses, so they need a
// declaration but never a definition. Declaring them here lets clad emit
// activity-annotated __enzyme_autodiff calls, and saves every caller of the
// Enzyme backend from redeclaring them by hand.
//
// Enzyme's own <enzyme/utils> declares the same names, and duplicate
// declarations are harmless, so this header does not look for it. Including
// it when present would make every clad translation unit pull in a third
// party header on machines that happen to have Enzyme installed and not on
// those that do not.
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_ENZYMEBUILTINS_H
#define CLAD_DIFFERENTIATOR_ENZYMEBUILTINS_H

// The markers are non-const with external linkage because that is the
// interface Enzyme matches; a namespace-scope `const int` would have internal
// linkage and Enzyme would never see it. They hold no state -- only their
// addresses are ever taken.

/// The argument is not differentiated; it is passed through unchanged.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int enzyme_const;
/// The argument is a pointer whose pointee is differentiated. It is followed
/// by a shadow pointer that receives the adjoint.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int enzyme_dup;
/// The argument is passed by value and differentiated; its adjoint comes back
/// in the returned gradient struct.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
extern int enzyme_out;

namespace clad {
/// Carries the adjoints of the by-value real arguments back from
/// __enzyme_autodiff, which returns them as a struct rather than through
/// shadow pointers.
template <unsigned N> struct EnzymeGradient {
  // Matches the struct Enzyme returns; std::array would change the layout it
  // writes through.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  double d_arr[N];
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_ENZYMEBUILTINS_H
