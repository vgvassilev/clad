#include "clad/Differentiator/DerivedTypesHandler.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivedTypeInitialiser.h"

#include "clang/Sema/Sema.h"

using namespace clang;

namespace clad {
  DerivedTypesHandler::DerivedTypesHandler(ASTConsumer& consumer, Sema& semaRef)
      : m_Consumer(consumer), m_Sema(semaRef),
        m_Context(semaRef.getASTContext()) {}

  void DerivedTypesHandler::SetDTE(clang::QualType yType, clang::QualType xType,
                                   DerivedTypeEssentials DTE) {
    m_DerivedTypesEssentials[{yType, xType}] = DTE;
  }

  void DerivedTypesHandler::InitialiseDerivedType(QualType yQType,
                                                  QualType xQType) {
    auto DTI = DerivedTypeInitialiser(m_Consumer, m_Sema, *this, yQType,
                                      xQType);
    auto DTE = DTI.CreateDerivedTypeEssentials();
    SetDTE(yQType, xQType, DTE);
  }

  QualType DerivedTypesHandler::GetDerivedType(clang::QualType yQType,
                                               clang::QualType xQType) {
    auto it = m_DerivedTypesEssentials.find({yQType, xQType});
    if (it != m_DerivedTypesEssentials.end())
      return it->second.GetTangentQType();
    // assert("We should never reach here");
    return QualType();
  }

  static std::pair<QualType, QualType>
  ComputeYandXQTypes(clang::QualType derivedQType) {
    std::pair<QualType, QualType> types;
    if (auto TS = dyn_cast<ClassTemplateSpecializationDecl>(
            derivedQType->getAsCXXRecordDecl())) {
      auto& templateArgs = TS->getTemplateArgs();
      types.first = templateArgs.get(0).getAsType();
      types.second = templateArgs.get(1).getAsType();
    }
    return types;
  }

  clang::QualType DerivedTypesHandler::GetYType(clang::QualType derivedQType) {
    auto it = m_DerivedTypesEssentials.find(ComputeYandXQTypes(derivedQType));
    if (it == m_DerivedTypesEssentials.end()) {
      return QualType();
    }
    return it->second.GetYQType();
  }

  DerivedTypeEssentials
  DerivedTypesHandler::GetDTE(clang::QualType derivedQType) {
    auto it = m_DerivedTypesEssentials.find(ComputeYandXQTypes(derivedQType));
    if (it == m_DerivedTypesEssentials.end()) {
      return DerivedTypeEssentials();
    }
    return it->second;
  }
} // namespace clad