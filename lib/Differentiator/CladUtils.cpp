#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
namespace clad {
  namespace utils {
    std::string ComputeEffectiveFnName(const clang::FunctionDecl* FD) {
      // TODO: Add cases for more operators
      switch (FD->getOverloadedOperator()) {
        case clang::OverloadedOperatorKind::OO_Call: return "operator_call";
        default: return FD->getNameAsString();
      }
    }

    std::pair<std::string, std::string>
    ComputeYAndXTypeNames(llvm::StringRef cladDerivedTypeName) {
      std::pair<std::string, std::string> typeNames;
      if (!cladDerivedTypeName.startswith("__clad_"))
        return typeNames;
      std::size_t prefixLength = 7;
      auto wrtPos = cladDerivedTypeName.find("_wrt_");
      std::size_t wrtSeparaterLength = 5;
      if (wrtPos == llvm::StringRef::npos)
        return typeNames;
      typeNames.first = cladDerivedTypeName.substr(prefixLength,
                                                   wrtPos - prefixLength).str();
      typeNames.second = cladDerivedTypeName.substr(wrtPos +
                                                    wrtSeparaterLength).str();
      return typeNames;
    }

    std::string CreateDerivedTypeName(llvm::StringRef YName,
                                      llvm::StringRef XName) {
      return "__clad_" + YName.str() + "_wrt_" + XName.str();
    }

    std::string GetRecordName(clang::QualType qType) {
      auto name = qType.getAsString();
      auto pos = name.rfind(" ");
      return name.substr(pos+1);
    }
  } // namespace utils
} // namespace clad