#include "ActivityAnalyzer.h"
#include "AnalysisBase.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <cstddef>
#include <memory>
#include <set>

using namespace clang;

namespace clad {

void VariedAnalyzer::Analyze() {
  m_BlockData.resize(m_AnalysisDC->getCFG()->size());
  // Set current block ID to the ID of entry the block.

  CFGBlock& entry = m_AnalysisDC->getCFG()->getEntry();
  m_CurBlockID = entry.getBlockID();

  m_BlockData[m_CurBlockID] = std::make_unique<VarsData>();

  const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function);
  if (MD && !MD->isStatic()) {
    VarData& thisData = getCurBlockVarsData()[nullptr];
    thisData = VarData(MD->getThisType(), /*forceInit=*/true);
    if (!isa<CXXConstructorDecl>(m_DiffReq.Function))
      setIsRequired(&thisData);
  }

  for (const auto* i : m_DiffReq.getVariedDecls()) {
    addVar(i, /*forceInit=*/true);
    setIsRequired(getVarDataFromDecl(i));
  }

  if (const auto* CD = dyn_cast<CXXConstructorDecl>(m_DiffReq.Function))
    for (auto* CI : CD->inits())
      TraverseStmt(CI->getInit());

  auto paramsRef = m_DiffReq.Function->parameters();
  // If parameter was not marked as varied, add it's VarData and mark
  // non-varied.
  for (const auto& par : paramsRef) {
    if (m_DiffReq.getVariedDecls().find(par) ==
        m_DiffReq.getVariedDecls().end()) {
      addVar(par, /*forceInit=*/true);
      setIsRequired(getVarDataFromDecl(par), /*isReq=*/false);
    }
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

bool VariedAnalyzer::isNaNRiskCallee(const clang::FunctionDecl* FD) {
  if (!FD)
    return false;
  if (!FD->getDeclName().isIdentifier())
    return false;
  llvm::StringRef name = FD->getName();

  // List of domain-restricted math functions
  return name == "acos" || name == "asin" || name == "atan" ||
         name == "atan2" || name == "atanh" || name == "sqrt" ||
         name == "log" || name == "log10" || name == "log2" || name == "pow";
}
bool VariedAnalyzer::exprHasNaNRisk(const clang::Expr* E) {
  if (!E)
    return false;
  E = E->IgnoreParenCasts();

  // Check if we already marked this expression
  if (m_PotentialNaNExprs.count(E))
    return true;

  // Check if it's a call to a NaN-risk function
  if (const auto* CE = dyn_cast<CallExpr>(E)) {
    if (const FunctionDecl* FD = CE->getDirectCallee()) {
      if (isNaNRiskCallee(FD))
        return true;
    }
  }

  // Check if it references a NaN-risk variable
  if (const auto* DRE = dyn_cast<DeclRefExpr>(E)) {
    if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (m_PotentialNaNVars.count(VD))
        return true;
    }
  }

  // Recursively check children
  for (const Stmt* child : E->children()) {
    if (const Expr* childExpr = dyn_cast_or_null<Expr>(child)) {
      if (exprHasNaNRisk(childExpr))
        return true;
    }
  }

  return false;
}

void VariedAnalyzer::TraverseAllStmtInsideBlock(const CFGBlock& block) {
  for (const clang::CFGElement& Element : block) {
    if (Element.getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = Element.castAs<clang::CFGStmt>().getStmt();
      // The const_cast is inevitable, since there is no
      // ConstRecusiveASTVisitor.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }
}
void VariedAnalyzer::AnalyzeCFGBlock(const CFGBlock& block) {
  // Visit all the statements inside the block.
  TraverseAllStmtInsideBlock(block);

  for (clang::CFGBlock::AdjacentBlock succ : block.succs()) {
    if (!succ)
      continue;
    while (succ->empty() && succ->succ_size() == 1)
      succ = *succ->succ_begin();
    auto& succData = m_BlockData[succ->getBlockID()];

    // Create VarsData for the succ branch if it hasn't been done previously.
    // If the successor doesn't have a VarsData, assign it and attach the
    // current block as previous.
    if (!succData) {
      succData = std::make_unique<VarsData>();
      succData->m_Prev = m_BlockData[block.getBlockID()].get();
    }
    if (succData->m_Prev == m_BlockData[block.getBlockID()].get()) {
      m_CFGQueue.insert(succ->getBlockID());
    } else {
      if (const Stmt* TS = succ->getTerminatorStmt())
        if (isa<CXXForRangeStmt>(TS) || isa<ForStmt>(TS) ||
            isa<WhileStmt>(TS) || isa<DoStmt>(TS))
          TraverseAllStmtInsideBlock(*succ);
      if (merge(succData.get(), m_BlockData[block.getBlockID()].get()))
        m_CFGQueue.insert(succ->getBlockID());
    }
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
    TraverseStmt(L);
    TraverseStmt(R);
  }
  if (BinOp->isAssignmentOp() || BinOp->isCompoundAssignmentOp()) {
    Expr* R = BinOp->getRHS();
    Expr* L = BinOp->getLHS();

    if (exprHasNaNRisk(R)) {
      L = L->IgnoreParenCasts();

      // Mark the LHS variable as NaN-risk
      if (const auto* DRE = dyn_cast<DeclRefExpr>(L)) {
        if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->getDeclName().isIdentifier())
            m_PotentialNaNVars.insert(VD);
        }
      }
    }
  }
  return false;
}

bool VariedAnalyzer::TraverseCompoundAssignOperator(
    clang::CompoundAssignOperator* CAO) {
  VariedAnalyzer::TraverseBinaryOperator(CAO);
  return false;
}

bool VariedAnalyzer::TraverseConditionalOperator(ConditionalOperator* CO) {
  TraverseStmt(CO->getCond());
  TraverseStmt(CO->getTrueExpr());
  TraverseStmt(CO->getFalseExpr());
  return false;
}
bool VariedAnalyzer::TraverseCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr* CE) {
  FunctionDecl* FD = CE->getDirectCallee();
  const auto* MD = dyn_cast<CXXMethodDecl>(FD);
  bool isMethodOperatorCall = MD && isa<CXXOperatorCallExpr>(CE);
  Expr* baseOriginalE = CE->getArg(0);
  bool hasVariedArg = false;
  bool variedBefore = m_Varied;

  for (std::size_t i = static_cast<std::size_t>(isMethodOperatorCall),
                   e = CE->getNumArgs();
       i != e; ++i) {
    Expr* arg = CE->getArg(i);
    const auto* PVD =
        FD->getParamDecl(i - static_cast<unsigned long>(isMethodOperatorCall));

    m_Varied = false;
    TraverseStmt(arg);
    if (m_Varied) {
      hasVariedArg = true;
      markExpr(arg);
      m_DiffReq.addVariedDecl(PVD);
    }
  }

  if (hasVariedArg || variedBefore) {
    m_Varied = true;
    m_Marking = true;
  }

  TraverseStmt(baseOriginalE);

  m_Varied = false;
  m_Marking = false;

  m_Varied = hasVariedArg || variedBefore || m_DiffReq.isVaried(baseOriginalE);
  return false;
}

bool VariedAnalyzer::TraverseCallExpr(CallExpr* CE) {
  Expr* callee = CE->getCallee();
  if (isa<CXXPseudoDestructorExpr>(callee))
    return false;

  bool variedBefore = m_Varied;
  bool hasVariedArg = false;
  FunctionDecl* FD = CE->getDirectCallee();
  if (!FD)
    return RecursiveASTVisitor::TraverseCallExpr(CE);
  bool noHiddenParam = (CE->getNumArgs() == FD->getNumParams());
  if (noHiddenParam) {
    MutableArrayRef<ParmVarDecl*> FDparam = FD->parameters();
    for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
      clang::Expr* arg = CE->getArg(i);

      QualType parType = FDparam[i]->getType();
      QualType innerMostType = parType;

      while (innerMostType->isPointerType())
        innerMostType = innerMostType->getPointeeType();
      m_Varied = false;
      if ((utils::isArrayOrPointerType(parType) &&
           !innerMostType.isConstQualified()) ||
          (parType->isReferenceType() &&
           !parType.getNonReferenceType().isConstQualified())) {
        m_Marking = true;
        m_Varied = true;
      }

      TraverseStmt(arg);

      if (m_Varied) {
        markExpr(arg);
        hasVariedArg = true;
        m_DiffReq.addVariedDecl(FDparam[i]);
      }

      m_Marking = false;
      m_Varied = false;
    }
    m_Varied = hasVariedArg || variedBefore;
  }
  if (const FunctionDecl* FD = CE->getDirectCallee()) {
    if (isNaNRiskCallee(FD))
      m_PotentialNaNExprs.insert(CE);
  }
  return false;
}

bool VariedAnalyzer::TraverseDeclStmt(DeclStmt* DS) {
  for (Decl* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      addVar(VD);
      if (Expr* init = cast<VarDecl>(D)->getInit()) {
        m_Varied = false;
        m_Marking = false;
        TraverseStmt(init);

        if (m_Varied) {
          m_DiffReq.addVariedDecl(VD);
          setIsRequired(getVarDataFromDecl(VD));
        } else {
          setIsRequired(getVarDataFromDecl(VD), /*isReq=*/false);
        }

        auto* VDExpr = &getCurBlockVarsData()[VD];
        QualType VDType = VD->getType();
        // if the declared variable is ref type attach its VarData to the
        // VarData of the RHS variable.
        if (VDExpr->m_Type == VarData::REF_TYPE || VDType->isPointerType()) {
          init = init->IgnoreParenCasts();
          if (VDType->isPointerType())
            VDExpr->m_Type = VarData::REF_TYPE;
          new (&VDExpr->m_Val.m_RefData)
              std::unique_ptr<std::set<const VarDecl*>>(
                  std::make_unique<std::set<const VarDecl*>>());
          getDependencySet(init, *VDExpr->m_Val.m_RefData);
        }
      }

      const auto* init = dyn_cast<VarDecl>(D)->getInit();
      if (init && exprHasNaNRisk(init))
        if (VD->getDeclName().isIdentifier())
          if (exprHasNaNRisk(init))
            m_PotentialNaNVars.insert(VD);
    }
  }
  return false;
}

bool VariedAnalyzer::TraverseCXXConstructExpr(clang::CXXConstructExpr* CE) {
  CXXConstructorDecl* CD = CE->getConstructor();
  auto parCD = CD->parameters();
  bool variedBefore = m_Varied;
  bool hasVariedArg = false;
  for (unsigned i = 0; i < CE->getNumArgs(); ++i) {
    m_Varied = false;
    clang::Expr* argExpr = CE->getArg(i);

    QualType parType = parCD[i]->getType();
    m_Varied = false;
    if ((utils::isArrayOrPointerType(parType)) ||
        (parType->isLValueReferenceType())) {
      m_Marking = true;
      m_Varied = true;
    }

    TraverseStmt(argExpr);

    if (m_Varied) {
      hasVariedArg = true;
      markExpr(argExpr);
      m_DiffReq.addVariedDecl(parCD[i]);
    }
  }

  m_Varied = variedBefore || hasVariedArg;
  return false;
}

bool VariedAnalyzer::TraverseCXXThisExpr(clang::CXXThisExpr* TE) {
  markExpr(TE);
  setVaried(TE);
  return false;
}

bool VariedAnalyzer::TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* CE) {
  const CXXMethodDecl* Method = CE->getMethodDecl();
  auto params = Method->parameters();

  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    Expr* arg = CE->getArg(i);

    QualType parType = params[i]->getType();
    QualType innerMostType = parType;

    while (innerMostType->isPointerType())
      innerMostType = innerMostType->getPointeeType();
    m_Varied = false;
    if ((utils::isArrayOrPointerType(parType) &&
         !innerMostType.isConstQualified()) ||
        (parType->isReferenceType() &&
         !parType.getNonReferenceType().isConstQualified())) {
      m_Marking = true;
      m_Varied = true;
    }

    TraverseStmt(arg);

    if (m_Varied) {
      markExpr(arg);
      m_DiffReq.addVariedDecl(params[i]);
    }
    m_Varied = false;
    m_Marking = false;
  }

  // Since we now support restore tracker for member functions, the adjoint of
  // the object is needed either way.
  m_Varied = true;
  m_Marking = true;

  TraverseStmt(CE->getImplicitObjectArgument());

  m_Varied = false;
  m_Marking = false;

  m_Varied = true;
  return false;
}

bool VariedAnalyzer::TraverseMemberExpr(clang::MemberExpr* ME) {
  TraverseStmt(ME->getBase());
  if (m_Varied) {
    markExpr(ME);
    setVaried(ME);
  }
  return false;
}

bool VariedAnalyzer::TraverseInitListExpr(clang::InitListExpr* ILE) {
  bool hasVariedEntry = false;
  bool variedBefore = m_Varied;
  for (auto* init : ILE->inits()) {
    m_Varied = false;
    TraverseStmt(init);
    if (m_Varied) {
      hasVariedEntry = true;
      markExpr(ILE);
    }
  }
  m_Varied = hasVariedEntry || variedBefore;
  return false;
}

bool VariedAnalyzer::TraverseUnaryOperator(UnaryOperator* UnOp) {
  Expr* E = UnOp->getSubExpr();
  TraverseStmt(E);
  return false;
}

bool VariedAnalyzer::TraverseDeclRefExpr(DeclRefExpr* DRE) {
  auto* VD = cast<VarDecl>(DRE->getDecl());

  if (VD->hasGlobalStorage() && VD->isFileVarDecl())
    return false;
  if (m_Varied && m_Marking) {
    setVaried(DRE);
    m_DiffReq.addVariedDecl(VD);
    markExpr(DRE);
  } else if (m_Marking)
    setVaried(DRE, false);

  if (VarData* data = getVarDataFromDecl(VD))
    if (findReq(*data))
      m_Varied = true;
  return false;
}
} // namespace clad
