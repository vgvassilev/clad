#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/SourceLocation.h"
#include "clad/Differentiator/Compatibility.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
namespace clad {
  namespace utils {
    static SourceLocation noLoc{};
    
    std::string ComputeEffectiveFnName(const FunctionDecl* FD) {
      // TODO: Add cases for more operators
      switch (FD->getOverloadedOperator()) {
        case OverloadedOperatorKind::OO_Call: return "operator_call";
        default: return FD->getNameAsString();
      }
    }

    CompoundStmt* PrependAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                               Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      block.push_back(S);
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(initial))
        block.append(CS->body_begin(), CS->body_end());
      else 
        block.push_back(initial);
      auto stmtsRef = llvm::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef, noLoc, noLoc);
    }

    CompoundStmt* AppendAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                              Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      assert(isa<CompoundStmt>(initial) &&
             "initial should be of type `clang::CompoundStmt`");
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(initial))
        block.append(CS->body_begin(), CS->body_end());
      block.push_back(S);
      auto stmtsRef = llvm::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef, noLoc, noLoc);
    }
  } // namespace utils
} // namespace clad