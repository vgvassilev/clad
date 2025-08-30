#include "ActivityAnalyzer.h"
#include "AnalysisBase.h"

#include "clang/AST/Type.h"

using namespace clang;

namespace clad {

void VariedAnalyzer::Analyze() {
  m_BlockData.resize(m_AnalysisDC->getCFG()->size());
  // Set current block ID to the ID of entry the block.

  CFGBlock& entry = m_AnalysisDC->getCFG()->getEntry();
  m_CurBlockID = entry.getBlockID();

  m_BlockData[m_CurBlockID] = std::unique_ptr<VarsData>(new VarsData());

  for (auto* i : m_DiffReq.getVariedDecls()) {
    addVar(i, /*forceNonRefType=*/true);
    setIsRequired(getVarDataFromDecl(i));
  }

  // Add the entry block to the queue.
  m_CFGQueue.insert(m_CurBlockID);

  // Visit CFG blocks in the queue until it's empty.
  while (!m_CFGQueue.empty()) {
    auto IDIter = std::prev(m_CFGQueue.end());
    m_CurBlockID = *IDIter;
    m_CFGQueue.erase(IDIter);
    CFGBlock& nextBlock = *getCFGBlockByID(m_AnalysisDC, m_CurBlockID);
    AnalyzeCFGBlock(nextBlock);
  }
}

void VariedAnalyzer::AnalyzeCFGBlock(const CFGBlock& block) {
  VarsData blockCopy;
  for (auto& i : *m_BlockData[block.getBlockID()])
    blockCopy[i.first] = i.second.copy();

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

  if (blockCopy == *m_BlockData[block.getBlockID()]) {
    if (m_LoopQueueMod.find(block.getBlockID()) != m_LoopQueueMod.end())
      m_LoopQueueMod[block.getBlockID()].m_isModified = 1;
  }

  for (const clang::CFGBlock::AdjacentBlock succ : block.succs()) {
    if (!succ)
      continue;
    auto& succData = m_BlockData[succ->getBlockID()];

    // Create VarsData for the succ branch if it hasn't been done previously.
    // If the successor doesn't have a VarsData, assign it and attach the
    // current block as previous.
    if (!succData) {
      succData = std::unique_ptr<VarsData>(new VarsData());
      succData->m_Prev = m_BlockData[block.getBlockID()].get();
    }

    bool shouldPushSucc = true;

    if (succ->getBlockID() > block.getBlockID()) {
      auto refIt = m_LoopQueueMod.find(block.getBlockID());
      int loopToRemove = refIt->second.m_LoopID;

      int order = getOrder();
      if (areBlocksFixed(order)) {
        shouldPushSucc = false;

        if (refIt == m_LoopQueueMod.end())
          return;

        for (auto it = m_LoopQueueMod.begin(); it != m_LoopQueueMod.end();)
          if (it->second.m_LoopID == loopToRemove)
            it = m_LoopQueueMod.erase(it);
          else
            ++it;
      } else {
        for (unsigned int i = block.getBlockID(); i < succ->getBlockID() + 1;
             i++) {
          m_LoopQueueMod[i].m_isModified = 0;
          m_LoopQueueMod[i].m_LoopID = order + 1;
        }
      }
    }

    if (shouldPushSucc)
      m_CFGQueue.insert(succ->getBlockID());

    if (succData->m_Prev != m_BlockData[block.getBlockID()].get())
      merge(succData.get(), m_BlockData[block.getBlockID()].get());
  }
}

void VariedAnalyzer::setVaried(const clang::Expr* E, bool isVaried) {
  llvm::SmallVector<ProfileID, 2> IDSequence;
  const VarDecl* VD = nullptr;
  bool sequenceFound = getIDSequence(E, VD, IDSequence);
  std::set<const clang::VarDecl*> vars;
  if (sequenceFound)
    vars.insert(VD);
  else if (isVaried)
    getDependencySet(E, vars);
  // Make sure the current branch has a copy of VarDecl for VD
  auto& curBranch = getCurBlockVarsData();
  for (const VarDecl* iterVD : vars) {
    if (isVaried || sequenceFound) {
      if (curBranch.find(iterVD) == curBranch.end()) {
        if (VarData* data = getVarDataFromDecl(iterVD))
          curBranch[iterVD] = data->copy();
        else
          // If this variable was not found in predecessors, add it.
          addVar(iterVD);
      }

      VarData* data = getVarDataFromDecl(iterVD);
      setIsRequired(data, isVaried, IDSequence);
    }
  }
}

bool VariedAnalyzer::TraverseBinaryOperator(BinaryOperator* BinOp) {
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();
  const auto opCode = BinOp->getOpcode();
  if (BinOp->isAssignmentOp()) {
    m_Varied = false;
    TraverseStmt(R);
    m_Marking = true;
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

bool VariedAnalyzer::TraverseConditionalOperator(ConditionalOperator* CO) {
  TraverseStmt(CO->getCond());
  TraverseStmt(CO->getTrueExpr());
  TraverseStmt(CO->getFalseExpr());
  return true;
}

bool VariedAnalyzer::TraverseCallExpr(CallExpr* CE) {
  bool hasVariedArg = false;
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
      m_Marking = true;

      TraverseStmt(arg);
      if (m_Varied) {
        hasVariedArg = true;
        m_DiffReq.addVariedDecl(FDparam[i]);
      }

      m_Marking = false;
      m_Varied = false;
    }
  }
  // is that so?
  m_Varied = hasVariedArg;
  return true;
}

bool VariedAnalyzer::TraverseDeclStmt(DeclStmt* DS) {
  for (Decl* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      addVar(VD);
      if (Expr* init = cast<VarDecl>(D)->getInit()) {
        m_Varied = false;
        TraverseStmt(init);
        m_Marking = true;
        if (m_Varied) {
          m_DiffReq.addVariedDecl(VD);
          setIsRequired(getVarDataFromDecl(VD));
        }
        m_Marking = false;
      }
    }
  }
  return true;
}

bool VariedAnalyzer::TraverseUnaryOperator(UnaryOperator* UnOp) {
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

// Move to CladUtils
// static const DeclStmt* getDeclStmtFromVarDecl(AnalysisDeclContext* ADC,
// VarDecl* VD){
//   class DeclStmtFinder: public RecursiveASTVisitor<DeclStmtFinder> {
//     clang::AnalysisDeclContext* m_ADC;
//     clang::VarDecl* m_VD;
//     clang::DeclStmt* m_DC = nullptr;
//     public:
//     DeclStmtFinder(clang::AnalysisDeclContext* ADC, clang::VarDecl* VD):
//     m_ADC(ADC), m_VD(VD){};
//
//     clang::DeclStmt* FindDeclStmt(){
//       for (auto* block : *m_ADC->getCFG()) {
//             for (const clang::CFGElement& Element : *block) {
//               if (Element.getKind() == clang::CFGElement::Statement) {
//                 const clang::Stmt* S =
//                     Element.castAs<clang::CFGStmt>().getStmt();
//                 TraverseStmt(const_cast<clang::Stmt*>(S));
//               }
//             }
//       }
//       return m_DC;
//     }
//
//     bool TraverseDeclStmt(clang::DeclStmt* DS){
//
//
//
//       return true;
//     }
//   };
// }

bool VariedAnalyzer::TraverseDeclRefExpr(DeclRefExpr* DRE) {
  auto* VD = dyn_cast<VarDecl>(DRE->getDecl());
  if (!VD)
    return true;

  if (VarData* data = getVarDataFromDecl(VD)) {
    if (findReq(*data)) {
      m_Varied = true;
    } else {
    }
  }

  if (m_Varied && m_Marking) {
    setVaried(DRE);
    m_DiffReq.addVariedDecl(VD);
    markExpr(DRE);
  } else if (m_Marking)
    setVaried(DRE, false);

  return true;
}
} // namespace clad
