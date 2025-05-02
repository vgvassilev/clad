#include "UsefulAnalyzer.h"

using namespace clang;

namespace clad {

void UsefulAnalyzer::Analyze(const FunctionDecl* FD) {
  // Build the CFG (control-flow graph) of FD.
  clang::CFG::BuildOptions Options;
  m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), &m_Context, Options);

  m_BlockData.resize(m_CFG->size());
  // Set current block ID to the ID of entry the block.
  CFGBlock* exit = &m_CFG->getExit();
  m_CurBlockID = exit->getBlockID();
  m_BlockData[m_CurBlockID] = createNewVarsData({});
  // Add the entry block to the queue.
  m_CFGQueue.insert(m_CurBlockID);

  // Visit CFG blocks in the queue until it's empty.
  while (!m_CFGQueue.empty()) {
    auto IDIter = m_CFGQueue.begin();
    m_CurBlockID = *IDIter;
    m_CFGQueue.erase(IDIter);
    CFGBlock& nextBlock = *getCFGBlockByID(m_CurBlockID);
    AnalyzeCFGBlock(nextBlock);
  }
}

CFGBlock* UsefulAnalyzer::getCFGBlockByID(unsigned ID) {
  return *(m_CFG->begin() + ID);
}

bool UsefulAnalyzer::isUseful(const VarDecl* VD) const {
  const VarsData& curBranch = getCurBlockVarsData();
  return curBranch.find(VD) != curBranch.end();
}

void UsefulAnalyzer::copyVarToCurBlock(const clang::VarDecl* VD) {
  VarsData& curBranch = getCurBlockVarsData();
  curBranch.insert(VD);
}

static void mergeVarsData(std::set<const clang::VarDecl*>* targetData,
                          std::set<const clang::VarDecl*>* mergeData) {
  for (const clang::VarDecl* i : *mergeData)
    targetData->insert(i);
  *mergeData = *targetData;
}

void UsefulAnalyzer::AnalyzeCFGBlock(const CFGBlock& block) {
  for (auto ib = block.end(); ib != block.begin() - 1; ib--) {
    if (ib->getKind() == clang::CFGElement::Statement) {

      const clang::Stmt* S = ib->castAs<clang::CFGStmt>().getStmt();
      // The const_cast is inevitable, since there is no
      // ConstRecusiveASTVisitor.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }

  for (const clang::CFGBlock::AdjacentBlock pred : block.preds()) {
    if (!pred)
      continue;
    auto& predData = m_BlockData[pred->getBlockID()];
    if (!predData)
      predData = createNewVarsData(*m_BlockData[block.getBlockID()]);

    bool shouldPushPred = true;
    if (pred->getBlockID() < block.getBlockID()) {
      if (m_LoopMem == *m_BlockData[block.getBlockID()])
        shouldPushPred = false;

      for (const VarDecl* i : *m_BlockData[block.getBlockID()])
        m_LoopMem.insert(i);
    }

    if (shouldPushPred)
      m_CFGQueue.insert(pred->getBlockID());

    mergeVarsData(predData.get(), m_BlockData[block.getBlockID()].get());
  }

  for (const VarDecl* i : *m_BlockData[block.getBlockID()])
    m_UsefulDecls.insert(i);
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

bool UsefulAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
  auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return true;

  if (isUseful(VD))
    m_Useful = true;

  if (m_Useful && m_Marking)
    copyVarToCurBlock(VD);

  return true;
}

} // namespace clad
