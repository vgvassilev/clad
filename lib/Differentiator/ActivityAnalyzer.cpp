#include "ActivityAnalyzer.h"
#include "clang/AST/Type.h"

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

void mergeVarsData(std::set<const clang::VarDecl*>* targetData,
                   std::set<const clang::VarDecl*>* mergeData) {
  for (const clang::VarDecl* i : *mergeData)
    targetData->insert(i);
  *mergeData = *targetData;
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

    mergeVarsData(succData.get(), m_BlockData[block.getBlockID()].get());
  }
  // FIXME: Information about the varied variables is stored in the last block,
  // so we should be able to get it form there
  for (const VarDecl* i : *m_BlockData[block.getBlockID()])
    m_VariedDecls.insert(i);
}

bool VariedAnalyzer::isVaried(const VarDecl* VD) const {
  const VarsData& curBranch = getCurBlockVarsData();
  return curBranch.find(VD) != curBranch.end();
}

void VariedAnalyzer::copyVarToCurBlock(const clang::VarDecl* VD) {
  VarsData& curBranch = getCurBlockVarsData();
  curBranch.insert(VD);
}

bool VariedAnalyzer::VisitBinaryOperator(BinaryOperator* BinOp) {
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();
  const auto opCode = BinOp->getOpcode();
  if (BinOp->isAssignmentOp()) {
    m_Varied = false;
    TraverseStmt(R);
    m_Marking = m_Varied;
    TraverseStmt(L);
    m_Marking = false;
  } else if (opCode == BO_Add || opCode == BO_Sub || opCode == BO_Mul ||
             opCode == BO_Div) {
    for (auto* subexpr : BinOp->children())
      if (!isa<BinaryOperator>(subexpr))
        TraverseStmt(subexpr);
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
  if (noHiddenParam) {
    MutableArrayRef<ParmVarDecl*> FDparam = FD->parameters();
    for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
      clang::Expr* arg = CE->getArg(i);

      QualType parType = FDparam[i]->getType();
      QualType innerMostType = parType;
      while (innerMostType->isPointerType())
        innerMostType = innerMostType->getPointeeType();
      if ((utils::isArrayOrPointerType(parType) &&
           !innerMostType.isConstQualified()) ||
          (parType->isReferenceType() &&
           !parType.getNonReferenceType().isConstQualified())) {
        m_Marking = true;
        m_Varied = true;
      }

      TraverseStmt(arg);

      if (m_Varied)
        m_VariedDecls.insert(FDparam[i]);

      m_Marking = false;
      m_Varied = false;
    }
  }
  return true;
}

bool VariedAnalyzer::VisitDeclStmt(DeclStmt* DS) {
  for (Decl* D : DS->decls()) {
    QualType VDTy = cast<VarDecl>(D)->getType();
    if (utils::isArrayOrPointerType(VDTy)) {
      copyVarToCurBlock(cast<VarDecl>(D));
      continue;
    }
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
  const auto opCode = UnOp->getOpcode();
  Expr* E = UnOp->getSubExpr();
  if (opCode == UO_AddrOf || opCode == UO_Deref) {
    m_Varied = true;
    m_Marking = true;
  }
  TraverseStmt(E);
  m_Marking = false;
  return true;
}

bool VariedAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
  auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return true;

  if (isVaried(VD))
    m_Varied = true;

  if (m_Varied && m_Marking)
    copyVarToCurBlock(VD);
  return true;
}
} // namespace clad
