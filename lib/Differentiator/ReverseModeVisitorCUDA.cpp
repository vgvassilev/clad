#include "ConstantFolder.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"

using namespace clang;

namespace clad {

static void CloneCUDASharedAttr(const clang::VarDecl* OriginalVD,
                                clang::VarDecl* VDClone) {
  if (const auto* attr = OriginalVD->getAttr<clang::CUDASharedAttr>())
    VDClone->addAttr(attr->clone(OriginalVD->getASTContext()));
}

void ReverseModeVisitor::HandleCUDASharedMemoryDecl(
    const clang::VarDecl* VD, clang::VarDecl* VDForward,
    clang::VarDecl* VDDerived,
    llvm::SmallVectorImpl<clang::Stmt*>& memsetCalls) {

  bool isDynamicSharedMem = VD->getType()->isIncompleteArrayType();

  if (!isDynamicSharedMem) {
    CloneCUDASharedAttr(VD, VDDerived);
    VDDerived->setStorageClass(clang::SC_Static);

    CloneCUDASharedAttr(VD, VDForward);
    VDForward->setStorageClass(clang::SC_Static);

    llvm::SmallVector<Expr*, 1> args = {BuildDeclRef(VDDerived)};
    Stmt* initCall = GetCladZeroInit(args);
    if (initCall)
      memsetCalls.push_back(initCall);
  } else {
    CloneCUDASharedAttr(VD, VDForward);
    Expr* derivedRef = BuildDeclRef(VDDerived);
    Expr* zeroIdx =
        ConstantFolder::synthesizeLiteral(m_Context.IntTy, m_Context, 0);
    Expr* arraySub = BuildArraySubscript(derivedRef, {zeroIdx});
    QualType elemType = VDDerived->getType()->getPointeeType();
    Expr* assignZero = BuildOp(BO_Assign, arraySub, getZeroInit(elemType));
    memsetCalls.push_back(assignZero);
  }
}
} // namespace clad
