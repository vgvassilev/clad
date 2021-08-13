#include "clad/Differentiator/CladUtils.h"

#include "clad/Differentiator/Compatibility.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

using namespace clang;

namespace clad {
  namespace utils {
    static SourceLocation noLoc;

    std::string ComputeEffectiveFnName(const FunctionDecl* FD) {
      // TODO: Add cases for more operators
      switch (FD->getOverloadedOperator()) {
        case OverloadedOperatorKind::OO_Call: return "operator_call";
        default: return FD->getNameAsString();
      }
    }

    StringLiteral* CreateStringLiteral(ASTContext& C, llvm::StringRef str) {
      // Copied and adapted from clang::Sema::ActOnStringLiteral.
      QualType CharTyConst = C.CharTy.withConst();
      QualType
          StrTy = clad_compat::getConstantArrayType(C, CharTyConst,
                                                    llvm::APInt(/*numBits=*/32,
                                                                str.size() + 1),
                                                    /*SizeExpr=*/nullptr,
                                                    /*ASM=*/ArrayType::Normal,
                                                    /*IndexTypeQuals*/ 0);
      StringLiteral* SL = StringLiteral::Create(C, str,
                                                /*Kind=*/StringLiteral::Ascii,
                                                /*Pascal=*/false, StrTy, noLoc);
      return SL;
    }
  } // namespace utils
} // namespace clad