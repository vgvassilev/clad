#include "ActivityAnalyzer.h"

using namespace clang;

namespace clad {

void VariedAnalyzer::Analyze(const FunctionDecl* FD) {
  // Build the CFG (control-flow graph) of FD.
  clang::CFG::BuildOptions Options;
  m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), &m_Context, Options);

  m_BlockData.resize(m_CFG->size());
  // Set current block ID to the ID of entry the block.
  CFGBlock* entry = &m_CFG->getEntry();
  m_CurBlockID = entry->getBlockID();
  m_BlockData[m_CurBlockID] = createNewVarsData({});
  for (const VarDecl* i : m_VariedDecls)
    m_BlockData[m_CurBlockID]->insert(i);
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

CFGBlock* VariedAnalyzer::getCFGBlockByID(unsigned ID) {
  return *(m_CFG->begin() + ID);
}

void VariedAnalyzer::AnalyzeCFGBlock(const CFGBlock& block) {
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

  for (const auto succ : block.succs()) {
    if (!succ)
      continue;
    auto& succData = m_BlockData[succ->getBlockID()];

    if (!succData) {
      succData = new VarsData(*m_BlockData[block.getBlockID()]);
      succData->m_Prev = m_BlockData[block.getBlockID()];
    }

    if (succ->getBlockID() > block.getBlockID()) {
      if (m_LoopMem == *m_BlockData[block.getBlockID()])
        m_shouldPushSucc = false;

      // has to be changed
      for (const auto& i : *m_BlockData[block.getBlockID()])
        m_LoopMem.insert(i);
    }

    if (m_shouldPushSucc)
      m_CFGQueue.insert(succ->getBlockID());
    m_shouldPushSucc = true;

    mergeVarsData(succData.get(), m_BlockData[block.getBlockID()].get());
  }

  // has to be changed
  for (const auto& i : *m_BlockData[block.getBlockID()])
    m_VariedDecls.insert(i);
}

bool VariedAnalyzer::isVaried(const VarDecl* VD) const {
  const VarsData& curBranch = getCurBlockVarsData();
  return curBranch.find(VD) != curBranch.end();
}

void VariedAnalyzer::copyVarToCurBlock(const clang::VarDecl* VD) {
  auto& curBranch = getCurBlockVarsData();
  curBranch.insert(VD);
}

bool VariedAnalyzer::VisitBinaryOperator(BinaryOperator* BinOp) {
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();

  if (BinOp->isAssignmentOp()) {
    m_Varied = false;
    TraverseStmt(R);
    m_Marking = m_Varied;
    TraverseStmt(L);
    m_Marking = false;
  } else {
    TraverseStmt(L);
    TraverseStmt(R);
  }
  return true;
}

bool VariedAnalyzer::VisitConditionalOperator(clang::ConditionalOperator* CO) {
  return true;
}

bool VariedAnalyzer::VisitDeclStmt(DeclStmt* DS) {
  for (Decl* D : DS->decls()) {
    if (!isa<VarDecl>(D))
      continue;
    if (Expr* init = cast<VarDecl>(D)->getInit()) {
      m_Varied = false;
      TraverseStmt(init);
      m_Marking = true;
      if (m_Varied)
        copyVarToCurBlock(cast<VarDecl>(D));
      m_Marking = false;
    }
  }
  return true;
}

bool VariedAnalyzer::VisitUnaryOperator(UnaryOperator* UnOp) {
  Expr* E = UnOp->getSubExpr();
  TraverseStmt(E);
  return true;
}

bool VariedAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
  if (isVaried(dyn_cast<VarDecl>(DRE->getDecl())))
    m_Varied = true;

  if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    if (m_Varied && m_Marking)
      copyVarToCurBlock(VD);
  }
  return true;
}
} // namespace clad
