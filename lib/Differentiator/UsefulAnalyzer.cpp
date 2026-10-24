#include "UsefulAnalyzer.h"

#include "AnalysisBase.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

#include <cstddef>

#include <memory>

using namespace clang;

namespace clad {

void UsefulAnalyzer::Analyze(const FunctionDecl* FD) {
  m_Function = FD;
  if (!m_AnalysisDC->getCFG())
    return;
  m_BlockData.resize(m_AnalysisDC->getCFG()->size());
  // Useful analysis starts at the exit and propagates to predecessors.
  CFGBlock& exit = m_AnalysisDC->getCFG()->getExit();
  m_CurBlockID = exit.getBlockID();
  m_BlockData[m_CurBlockID] = std::make_unique<VarsData>();
  // What a function writes into a parameter passed by non-const reference or
  // pointer, its caller reads: such a parameter is an output, as the return
  // value is.
  for (const ParmVarDecl* PVD : FD->parameters()) {
    QualType T = PVD->getType();
    if ((T->isReferenceType() || T->isPointerType()) &&
        !T->getPointeeType().isConstQualified())
      markUseful(PVD);
  }
  // Add the exit block to the queue.
  m_CFGQueue.insert(m_CurBlockID);

  // Visit CFG blocks in the queue until it's empty.
  while (!m_CFGQueue.empty()) {
    auto IDIter = m_CFGQueue.begin();
    m_CurBlockID = *IDIter;
    m_CFGQueue.erase(IDIter);
    CFGBlock& nextBlock = *getCFGBlockByID(m_AnalysisDC, m_CurBlockID);
    AnalyzeCFGBlock(nextBlock);
  }
}

bool UsefulAnalyzer::isUseful(const VarDecl* VD) {
  return getVarDataFromDecl(VD) != nullptr;
}

void UsefulAnalyzer::markUseful(const clang::VarDecl* VD) {
  // A useful declaration needs one leaf, not its field or pointee state.
  if (!getVarDataFromDecl(VD))
    getCurBlockVarsData()[VD] = VarData(m_AnalysisDC->getASTContext().BoolTy);
  m_UsefulDecls.insert(VD);
}

void UsefulAnalyzer::AnalyzeCFGBlock(const CFGBlock& block) {
  for (const auto* it = block.rbegin(); it != block.rend(); ++it) {
    if (it->getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = it->castAs<clang::CFGStmt>().getStmt();
      // The const_cast is inevitable, since there is no
      // ConstRecursiveASTVisitor.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }

  for (const clang::CFGBlock::AdjacentBlock pred : block.preds()) {
    if (!pred)
      continue;
    auto& predData = m_BlockData[pred->getBlockID()];
    auto* currentData = m_BlockData[block.getBlockID()].get();
    if (!predData) {
      predData = std::make_unique<VarsData>();
      predData->m_Prev = currentData;
    }

    // Discovery links form a forest. Other edges require a growing merge.
    if (predData->m_Prev == currentData || merge(predData.get(), currentData))
      m_CFGQueue.insert(pred->getBlockID());
  }
}

bool UsefulAnalyzer::VisitBinaryOperator(BinaryOperator* BinOp) {
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();
  const auto opCode = BinOp->getOpcode();
  if (BinOp->isAssignmentOp()) {
    m_Useful = false;
    TraverseStmt(L);
    m_Marking = m_Useful;
    TraverseStmt(R);
    m_Marking = false;
  } else if (opCode == BO_Add || opCode == BO_Sub || opCode == BO_Mul ||
             opCode == BO_Div) {
    for (auto* subexpr : BinOp->children())
      if (!isa<BinaryOperator>(subexpr))
        TraverseStmt(subexpr);
  }
  return true;
}

bool UsefulAnalyzer::VisitDeclStmt(DeclStmt* DS) {
  for (Decl* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      if (isUseful(VD)) {
        m_Useful = true;
        m_Marking = true;
      }
      if (Expr* init = dyn_cast<VarDecl>(D)->getInit())
        TraverseStmt(init);
      m_Marking = false;
    }
  }
  return true;
}

bool UsefulAnalyzer::VisitReturnStmt(ReturnStmt* RS) {
  m_Useful = true;
  m_Marking = true;
  auto* rv = RS->getRetValue();
  TraverseStmt(rv);
  m_Marking = false;
  return true;
}

bool UsefulAnalyzer::VisitCallExpr(CallExpr* CE) { return true; }

bool UsefulAnalyzer::namesUseful(const Expr* E) {
  class Finder : public RecursiveASTVisitor<Finder> {
    UsefulAnalyzer& m_UA;

  public:
    bool Found = false;
    explicit Finder(UsefulAnalyzer& UA) : m_UA(UA) {}
    bool VisitDeclRefExpr(DeclRefExpr* DRE) {
      if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Found |= m_UA.isUseful(VD);
      return !Found;
    }
  } F(*this);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  F.TraverseStmt(const_cast<Expr*>(E));
  return F.Found;
}

bool UsefulAnalyzer::writesUseful(const CallExpr* CE) {
  const FunctionDecl* FD = CE->getDirectCallee();
  std::size_t skip = 0;
  if (const auto* MD = dyn_cast_or_null<CXXMethodDecl>(FD))
    if (MD->isInstance()) {
      const Expr* Obj = nullptr;
      if (const auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
        Obj = MCE->getImplicitObjectArgument();
      else if (isa<CXXOperatorCallExpr>(CE)) {
        Obj = CE->getArg(0);
        skip = 1;
      }
      if (Obj && !MD->isConst() && namesUseful(Obj))
        return true;
    }
  for (std::size_t i = skip, e = CE->getNumArgs(); i != e; ++i) {
    // Without a declaration to read the parameter from, any argument may be
    // written.
    bool mayWrite = true;
    if (FD && i - skip < FD->getNumParams()) {
      QualType T = FD->getParamDecl(i - skip)->getType();
      mayWrite = (T->isReferenceType() || T->isPointerType()) &&
                 !T->getPointeeType().isConstQualified();
    }
    if (mayWrite && namesUseful(CE->getArg(i)))
      return true;
  }
  return false;
}

template <class F>
bool UsefulAnalyzer::traverseCall(const CallExpr* CE, F Base) {
  // Otherwise the call is read as any other expression is: the left side of
  // an assignment finds out through it whether it writes a useful variable.
  if (!writesUseful(CE)) {
    Base();
    return true;
  }
  // The call writes something useful, and so does an assignment to what it
  // returns, as `r(1) = t` does: m_Useful stays set for the assignment to see.
  m_Useful = true;
  bool savedMarking = m_Marking;
  m_Marking = true;
  Base();
  m_Marking = savedMarking;
  return true;
}

bool UsefulAnalyzer::TraverseCallExpr(CallExpr* CE) {
  return traverseCall(
      CE, [&] { RecursiveASTVisitor<UsefulAnalyzer>::TraverseCallExpr(CE); });
}

bool UsefulAnalyzer::TraverseCXXMemberCallExpr(CXXMemberCallExpr* CE) {
  return traverseCall(CE, [&] {
    RecursiveASTVisitor<UsefulAnalyzer>::TraverseCXXMemberCallExpr(CE);
  });
}

bool UsefulAnalyzer::TraverseCXXOperatorCallExpr(CXXOperatorCallExpr* CE) {
  return traverseCall(CE, [&] {
    RecursiveASTVisitor<UsefulAnalyzer>::TraverseCXXOperatorCallExpr(CE);
  });
}

bool UsefulAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
  auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return true;

  if (isUseful(VD))
    m_Useful = true;

  if (m_Useful && m_Marking)
    markUseful(VD);

  return true;
}

} // namespace clad
