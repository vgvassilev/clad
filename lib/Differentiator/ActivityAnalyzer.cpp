#include "ActivityAnalyzer.h"

using namespace clang;

namespace clad {

void VariedAnalyzer::Analyze(const FunctionDecl* FD) {
  // Build the CFG (control-flow graph) of FD.
  clang::CFG::BuildOptions Options;
  m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), &m_Context, Options);

  m_BlockData.resize(m_CFG->size());
  m_BlockPassCounter.resize(m_CFG->size(), 0);

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
    VisitCFGBlock(nextBlock);
  }
}

CFGBlock* VariedAnalyzer::getCFGBlockByID(unsigned ID) {
  return *(m_CFG->begin() + ID);
}

void VariedAnalyzer::VisitCFGBlock(const CFGBlock& block) {
  // Visit all the statements inside the block.
  for (const clang::CFGElement& Element : block) {
    if (Element.getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = Element.castAs<clang::CFGStmt>().getStmt();
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }

  for (const clang::CFGBlock::AdjacentBlock succ : block.succs()) {
    if (!succ)
      continue;
    auto& succData = m_BlockData[succ->getBlockID()];

    if (!succData)
      succData = createNewVarsData(*m_BlockData[block.getBlockID()]);

    bool shouldPushSucc = true;
    if (succ->getBlockID() > block.getBlockID()) {
      if (m_LoopMem == *m_BlockData[block.getBlockID()])
        shouldPushSucc = false;

      for (const VarDecl* i : *m_BlockData[block.getBlockID()])
        m_LoopMem.insert(i);
    }

    if (shouldPushSucc)
      m_CFGQueue.insert(succ->getBlockID());

    merge(succData.get(), m_BlockData[block.getBlockID()].get());
  }
  // FIXME: Information about the varied variables is stored in the last block,
  // so we should be able to get it form there
  for (const VarDecl* i : *m_BlockData[block.getBlockID()])
    m_VariedDecls.insert(i);
}

bool VariedAnalyzer::isVaried(const VarDecl* VD) {
  VarsData& curBranch = getCurBlockVarsData();
  return curBranch.find(VD) != curBranch.end();
}

void VariedAnalyzer::merge(VarsData* targetData, VarsData* mergeData) {
  for (const VarDecl* i : *mergeData)
    targetData->insert(i);
  for (const VarDecl* i : *targetData)
    mergeData->insert(i);
}

void VariedAnalyzer::copyVarToCurBlock(const clang::VarDecl* VD) {
  VarsData& curBranch = getCurBlockVarsData();
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

// add branching merging
bool VariedAnalyzer::VisitConditionalOperator(ConditionalOperator* CO) {
  TraverseStmt(CO->getCond());
  TraverseStmt(CO->getTrueExpr());
  TraverseStmt(CO->getFalseExpr());
  return true;
}

bool VariedAnalyzer::VisitCallExpr(CallExpr* CE) {
  FunctionDecl* FD = CE->getDirectCallee();
  bool noHiddenParam = (CE->getNumArgs() == FD->getNumParams());
  std::set<const clang::VarDecl*> variedParam;
  if (noHiddenParam) {
    MutableArrayRef<ParmVarDecl*> FDparam = FD->parameters();
    for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
      clang::Expr* par = CE->getArg(i);
      TraverseStmt(par);
      if (m_Varied) {
        m_VariedDecls.insert(FDparam[i]);
        m_Varied = false;
      }
    }
  }
  return true;
}

bool VariedAnalyzer::VisitDeclStmt(DeclStmt* DS) {
  for (Decl* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      if (Expr* init = VD->getInit()) {
        m_Varied = false;
        TraverseStmt(init);
        m_Marking = true;
        VarsData& curBranch = getCurBlockVarsData();
        if (m_Varied && curBranch.find(VD) == curBranch.end())
          copyVarToCurBlock(VD);
        m_Marking = false;
      }
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
    VarsData& curBranch = getCurBlockVarsData();
    if (m_Varied && m_Marking && curBranch.find(VD) == curBranch.end())
      copyVarToCurBlock(VD);
  }
  return true;
}
} // namespace clad
