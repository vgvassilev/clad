#ifndef CLAD_DIFFERENTIATOR_DERIVEDFNCOLLECTOR_H
#define CLAD_DIFFERENTIATOR_DERIVEDFNCOLLECTOR_H

#include "clad/Differentiator/DerivedFnInfo.h"

#include "clang/AST/Decl.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clad {
/// This class is designed to store collection of `DerivedFnInfo` objects.
/// It's purpose is to avoid repeated generation of same derivatives by
/// making it possible to reuse previously computed derivatives.
class DerivedFnCollector {
  using DerivedFns = llvm::SmallVector<DerivedFnInfo, 16>;
  using DerivativeSet = llvm::SmallSet<const clang::FunctionDecl*, 16>;
  /// Mapping to efficiently find out information about all the derivatives of
  /// a function.
  llvm::DenseMap<const clang::FunctionDecl*, DerivedFns>
      m_DerivedFnInfoCollection;
  /// Set to keep track of all the functions that are derivatives
  /// functions produced by Clad.
  DerivativeSet m_DerivativeSet;

  /// Set to keep track of all the functions that are custom derivatives
  /// functions provided by the user.
  DerivativeSet m_CustomDerivativeSet;

public:
  /// Adds a derived function to the collection.
  void Add(const DerivedFnInfo& DFI);

  /// Adds a function to derivative set.
  void AddToDerivativeSet(const clang::FunctionDecl* FD);

  /// Adds a function to custom derivative set.
  void AddToCustomDerivativeSet(const clang::FunctionDecl* FD);

  /// Finds a `DerivedFnInfo` object in the collection that satisfies the
  /// given differentiation request.
  DerivedFnInfo Find(const DiffRequest& request) const;

  /// Returns true if the function is a Clad-generated derivative.
  bool IsCladDerivative(const clang::FunctionDecl* FD) const;

  /// Returns true if the function is a custom derivative.
  bool IsCustomDerivative(const clang::FunctionDecl* FD) const;

private:
  /// Returns true if the collection already contains a `DerivedFnInfo`
  /// object that represents the same derivative object as the provided
  /// argument `DFI`.
  bool AlreadyExists(const DerivedFnInfo& DFI) const;
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DERIVEDFNCOLLECTOR_H