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
    llvm::SmallVector<Expr*, 1> args = {derivedRef};
    Stmt* clearCall = GetFunctionCall("clear_dynamic_smem", "clad", args);
    memsetCalls.push_back(clearCall);
  }

  // A barrier is strictly required here to prevent race conditions.
  // Since zero-initialization is performed by threads in the block, we must
  // ensure all threads finish zeroing the shared memory before any thread
  // proceeds to the reverse sweep and starts accumulating into it.
  llvm::SmallVector<Expr*, 0> syncArgs;
  Stmt* syncCall = GetFunctionCall("__syncthreads", "", syncArgs);
  memsetCalls.push_back(syncCall);
}
} // namespace clad
