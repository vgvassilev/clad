#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffPlanner.h"

#include "clang/AST/Decl.h"

#include "llvm/Support/Casting.h"

namespace clad {
void DerivedFnCollector::Add(const DerivedFnInfo& DFI) {
  assert(!AlreadyExists(DFI) &&
         "We are generating same derivative more than once, or calling "
         "`DerivedFnCollector::Add` more than once for the same derivative "
         ". Ideally, we shouldn't do either.");
  m_DerivedFnInfoCollection[DFI.OriginalFn()].push_back(DFI);
  AddToDerivativeSet(DFI.DerivedFn());
}

void DerivedFnCollector::AddToDerivativeSet(const clang::FunctionDecl* FD) {
  m_DerivativeSet.insert(FD);
}

void DerivedFnCollector::AddToCustomDerivativeSet(
    const clang::FunctionDecl* FD) {
  m_CustomDerivativeSet.insert(FD);
}

bool DerivedFnCollector::AlreadyExists(const DerivedFnInfo& DFI) const {
  auto subCollectionIt = m_DerivedFnInfoCollection.find(DFI.OriginalFn());
  if (subCollectionIt == m_DerivedFnInfoCollection.end())
    return false;
  const auto& subCollection = subCollectionIt->second;
  const auto* it =
      std::find_if(subCollection.begin(), subCollection.end(),
                   [&DFI](const DerivedFnInfo& info) {
                     return DerivedFnInfo::RepresentsSameDerivative(DFI, info);
                   });
  return it != subCollection.end();
}

DerivedFnInfo DerivedFnCollector::Find(const DiffRequest& request) const {
  auto subCollectionIt = m_DerivedFnInfoCollection.find(request.Function);
  if (subCollectionIt == m_DerivedFnInfoCollection.end())
    return DerivedFnInfo();
  const auto& subCollection = subCollectionIt->second;
  const auto* it = std::find_if(subCollection.begin(), subCollection.end(),
                                [&request](const DerivedFnInfo& DFI) {
                                  return DFI.SatisfiesRequest(request);
                                });
  if (it == subCollection.end())
    return DerivedFnInfo();
  if (request.RequestedDerivativeOrder > 1) {
    // Lookup the next order derivative.
    DiffRequest highOrderRequest = request;
    highOrderRequest.Function = it->DerivedFn();
    --highOrderRequest.RequestedDerivativeOrder;
    assert(request.DVI.size() == 1 &&
           "Only differentiation w.r.t. one parameter is supported in high "
           "order derivatives");
    const clang::ValueDecl*& decl = highOrderRequest.DVI[0].param;
    unsigned idx =
        llvm::cast<clang::ParmVarDecl>(decl)->getFunctionScopeIndex();
    decl = highOrderRequest.Function->getParamDecl(idx);
    return Find(highOrderRequest);
  }
  return *it;
}

bool DerivedFnCollector::IsCladDerivative(const clang::FunctionDecl* FD) const {
  return m_DerivativeSet.count(FD);
}

bool DerivedFnCollector::IsCustomDerivative(
    const clang::FunctionDecl* FD) const {
  return m_CustomDerivativeSet.count(FD);
}
} // namespace clad