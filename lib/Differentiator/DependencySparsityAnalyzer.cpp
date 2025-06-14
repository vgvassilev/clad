#include "DependencySparsityAnalyzer.h"
#include "clang/AST/Type.h"

using namespace clang;

namespace clad {

void DependencySparsityAnalyzer::Analyze(const FunctionDecl* FD) {
  // Build the CFG (control-flow graph) of FD.
  clang::CFG::BuildOptions Options;
  m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), &m_Context, Options);

  // Set current block ID to the ID of entry the block.
  CFGBlock* entry = &m_CFG->getEntry();
  m_CurBlockID = entry->getBlockID();

  int i = 0;
  for (const auto& par : FD->parameters()) {
    if (!utils::isArrayOrPointerType(par->getType())) {
      m_ParameterNum[dyn_cast<VarDecl>(par)] = i;
      i++;
    }
  }

  // Add the entry block to the queue.
  m_CFGQueue.insert(m_CurBlockID);

  // Visit CFG blocks in the queue until it's empty.
  while (!m_CFGQueue.empty()) {
    auto IDIter = std::prev(m_CFGQueue.end());
    m_CurBlockID = *IDIter;
    m_CFGQueue.erase(IDIter);
    CFGBlock& nextBlock = *getCFGBlockByID(m_CurBlockID);
    AnalyzeCFGBlock(nextBlock);
  }
}

void DependencySparsityAnalyzer::AnalyzeCFGBlock(const CFGBlock& block) {
  // Visit all the statements inside the block.
  for (const clang::CFGElement& Element : block) {
    if (Element.getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = Element.castAs<clang::CFGStmt>().getStmt();
      // The const_cast is inevitable, since there is no
      // ConstRecusiveASTVisitor.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }

  for (const clang::CFGBlock::AdjacentBlock succ : block.succs()) {
    if (!succ)
      continue;
    m_CFGQueue.insert(succ->getBlockID());
  }
}

CFGBlock* DependencySparsityAnalyzer::getCFGBlockByID(unsigned ID) {
  return *(m_CFG->begin() + ID);
}

bool DependencySparsityAnalyzer::VisitBinaryOperator(BinaryOperator* BinOp) {
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();
  const auto opCode = BinOp->getOpcode();
  if (BinOp->isAssignmentOp()) {
    TraverseStmt(L);
    // m_MarkingMode = true;
    TraverseStmt(R);
  } else if (opCode == BO_Add || opCode == BO_Sub || opCode == BO_Mul ||
             opCode == BO_Div) {
    for (auto* subexpr : BinOp->children())
      if (!isa<BinaryOperator>(subexpr))
        TraverseStmt(subexpr);
  }
  return true;
}

bool DependencySparsityAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
  if (m_MarkingMode) {
    auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (m_ParameterNum.find(VD) != m_ParameterNum.end()) {
      auto parr =
          std::make_pair(m_ParameterNum.find(VD)->second, m_CurrOutputInd);
      m_OutputDependencySet.insert(parr);
    }
  }
  return true;
}

bool DependencySparsityAnalyzer::VisitArraySubscriptExpr(
    ArraySubscriptExpr* ASE) {
  auto* dASE = dyn_cast<DeclRefExpr>(ASE->getBase()->IgnoreImpCasts());
  llvm::StringRef Name = dASE->getDecl()->getName();

  if (Name.contains("_clad_out_")) {
    const Expr* idx = ASE->getIdx();
    if (isa<IntegerLiteral>(idx)) {
      m_CurrOutputInd =
          dyn_cast<IntegerLiteral>(idx)->getValue().getLimitedValue();
      m_MarkingMode = true;
    }
  }
  return true;
}
} // namespace clad
