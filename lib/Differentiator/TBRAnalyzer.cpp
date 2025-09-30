#include "TBRAnalyzer.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <set>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/LLVM.h"

#include "AnalysisBase.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffPlanner.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "clad/Differentiator/CladUtils.h"
#undef DEBUG_TYPE
#define DEBUG_TYPE "clad-tbr"

using namespace clang;

namespace clad {

void TBRAnalyzer::markLocation(const clang::Stmt* S) { m_TBRLocs.insert(S); }

void TBRAnalyzer::setIsRequired(const clang::Expr* E, bool isReq) {
  llvm::SmallVector<ProfileID, 2> IDSequence;
  const VarDecl* VD = nullptr;
  bool sequenceFound = getIDSequence(E, VD, IDSequence);
  std::set<const clang::VarDecl*> vars;
  if (sequenceFound)
    vars.insert(VD);
  // If it wasn't possible to determine the exact VarData, we have to set all
  // dependencies to tbr.
  else if (isReq || m_ModifiedParams)
    getDependencySet(E, vars);
  // Make sure the current branch has a copy of VarDecl for VD
  auto& curBranch = getCurBlockVarsData();
  for (const VarDecl* iterVD : vars) {
    const auto* PVD = dyn_cast_or_null<ParmVarDecl>(iterVD);
    if (m_ModifiedParams && (!iterVD || PVD)) {
      if (!isReq)
        (*m_ModifiedParams)[m_Function].insert(PVD);
      else if (m_ModeStack.back() & Mode::kMarkingMode)
        (*m_UsedParams)[m_Function].insert(PVD);
    }
    if (isReq || sequenceFound) {
      if (curBranch.find(iterVD) == curBranch.end()) {
        if (VarData* data = getVarDataFromDecl(iterVD))
          curBranch[iterVD] = data->copy();
        else
          // If this variable was not found in predecessors, add it.
          addVar(iterVD);
      }

      if (!isReq ||
          (m_ModeStack.back() == (Mode::kMarkingMode | Mode::kNonLinearMode))) {
        VarData* data = getVarDataFromDecl(iterVD);
        AnalysisBase::setIsRequired(data, isReq, IDSequence);
      }
    }
  }
}

void TBRAnalyzer::Analyze(const DiffRequest& request) {
  m_BlockData.resize(request.m_AnalysisDC->getCFG()->size());
  m_BlockPassCounter.resize(request.m_AnalysisDC->getCFG()->size(), 0);

  // Set current block ID to the ID of entry the block.
  CFGBlock& entry = request.m_AnalysisDC->getCFG()->getEntry();
  m_CurBlockID = entry.getBlockID();
  m_BlockData[m_CurBlockID] = std::unique_ptr<VarsData>(new VarsData());

  const FunctionDecl* FD = request.Function;
  m_Function = FD;
  // FIXME: Perform TBR consistently and always pass this info.
  if (m_ModifiedParams)
    (*m_ModifiedParams)[FD];
  if (m_UsedParams)
    (*m_UsedParams)[FD];
  // If we are analysing a non-static method, add a VarData for 'this' pointer
  // (it is represented with nullptr).
  const auto* MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && !MD->isStatic()) {
    VarData& thisData = getCurBlockVarsData()[nullptr];
    thisData = VarData(MD->getThisType(), /*forceInit=*/true);
    // We have to set all pointer/reference parameters to tbr
    // since method pullbacks aren't supposed to change objects.
    // constructor pullbacks don't take `this` as a parameter
    if (!isa<CXXConstructorDecl>(FD))
      AnalysisBase::setIsRequired(&thisData);
  }
  auto paramsRef = FD->parameters();
  for (std::size_t i = 0; i < FD->getNumParams(); ++i)
    addVar(paramsRef[i], /*forceInit=*/true);
  // Add the entry block to the queue.
  m_CFGQueue.insert(m_CurBlockID);

  // Visit CFG blocks in the queue until it's empty.
  while (!m_CFGQueue.empty()) {
    auto IDIter = std::prev(m_CFGQueue.end());
    m_CurBlockID = *IDIter;
    m_CFGQueue.erase(IDIter);

    CFGBlock& nextBlock = *getCFGBlockByID(request.m_AnalysisDC, m_CurBlockID);
    VisitCFGBlock(nextBlock);
  }
#ifndef NDEBUG
  for (int id = m_CurBlockID; id >= 0; --id) {
    LLVM_DEBUG(llvm::dbgs() << "\n-----BLOCK" << id << "-----\n\n");
    for (auto succ : getCFGBlockByID(request.m_AnalysisDC, id)->succs())
      if (succ)
        LLVM_DEBUG(llvm::dbgs() << "successor: " << succ->getBlockID() << "\n");
  }

  clang::SourceManager& SM = m_AnalysisDC->getASTContext().getSourceManager();
  for (const Stmt* S : m_TBRLocs) {
    SourceLocation Loc = S->getBeginLoc();
    unsigned line = SM.getPresumedLoc(Loc).getLine();
    unsigned column = SM.getPresumedLoc(Loc).getColumn();
    LLVM_DEBUG(llvm::dbgs() << line << ":" << column << "\n");
  }
#endif // NDEBUG
}

void TBRAnalyzer::VisitCFGBlock(const CFGBlock& block) {
  LLVM_DEBUG(llvm::dbgs() << "Entering block " << block.getBlockID() << "\n");
  // Visiting loop blocks just once is not enough since the end of one loop
  // iteration may have an effect on the next one. However, two iterations is
  // always enough. Allow a third visit without going to successors to correctly
  // analyse loop conditions.
  bool notLastPass = ++m_BlockPassCounter[block.getBlockID()] <= 2;

  // Visit all the statements inside the block.
  for (const clang::CFGElement& Element : block) {
    if (Element.getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = Element.castAs<clang::CFGStmt>().getStmt();
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }

  // Traverse successor CFG blocks.
  for (const auto succ : block.succs()) {
    // Sometimes clang CFG does not create blocks for parts of code that
    // are never executed (e.g. 'if (0) {...'). Add this check for safety.
    if (!succ)
      continue;
    auto& varsData = m_BlockData[succ->getBlockID()];

    // Create VarsData for the succ branch if it hasn't been done previously.
    // If the successor doesn't have a VarsData, assign it and attach the
    // current block as previous.
    if (!varsData) {
      varsData = std::unique_ptr<VarsData>(new VarsData());
      varsData->m_Prev = m_BlockData[block.getBlockID()].get();
    }

    // If this is the third (last) pass of block, it means block represents a
    // loop condition and the loop body has already been visited 2 times.  This
    // means we should not visit the loop body anymore.
    if (notLastPass) {
      // Add the successor to the queue.
      m_CFGQueue.insert(succ->getBlockID());

      // This part is necessary for loops. For other cases, this is not supposed
      // to do anything.
      if (succ->getBlockID() < block.getBlockID()) {
        // If there is another loop condition present inside a loop,
        // We have to set it's loop pass counter to 0 (it might be 3
        // from the previous outer loop pass).
        m_BlockPassCounter[succ->getBlockID()] = 0;
        // Remove VarsData left after the previous pass.
        varsData->clear();
      }
    }

    // If the successor's previous block is not this one, perform a merge.
    if (varsData->m_Prev != m_BlockData[block.getBlockID()].get())
      merge(varsData.get(), m_BlockData[block.getBlockID()].get());
  }
  LLVM_DEBUG(llvm::dbgs() << "Leaving block " << block.getBlockID() << "\n");
}

bool TBRAnalyzer::TraverseDeclRefExpr(DeclRefExpr* DRE) {
  setIsRequired(DRE);
  return false;
}

bool TBRAnalyzer::TraverseDeclStmt(DeclStmt* DS) {
  for (auto* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      if (VarData* data = getVarDataFromDecl(VD)) {
        if (findReq(*data))
          markLocation(DS);
      }
      addVar(VD);
      if (clang::Expr* init = VD->getInit()) {

        setMode(Mode::kMarkingMode);
        TraverseStmt(init);
        resetMode();

        auto* VDExpr = &getCurBlockVarsData()[VD];
        QualType VDType = VD->getType();
        // if the declared variable is ref type attach its VarData to the
        // VarData of the RHS variable.
        if (VDExpr->m_Type == VarData::REF_TYPE || VDType->isPointerType()) {
          init = init->IgnoreParenCasts();
          if (VDType->isPointerType()) {
            if (isa<CXXNewExpr>(init)) {
              VDExpr->initializeAsArray(VDType);
              return false;
            }
            VDExpr->m_Type = VarData::REF_TYPE;
          }
          new (&VDExpr->m_Val.m_RefData)
              std::unique_ptr<std::set<const VarDecl*>>(
                  std::make_unique<std::set<const VarDecl*>>());
          getDependencySet(init, *VDExpr->m_Val.m_RefData);
        }
      }
    }
  }
  return false;
}

bool TBRAnalyzer::TraverseConditionalOperator(clang::ConditionalOperator* CO) {
  setMode(/*mode=*/0);
  TraverseStmt(CO->getCond());
  resetMode();

  auto elseBranch = std::move(m_BlockData[m_CurBlockID]);

  m_BlockData[m_CurBlockID] = std::unique_ptr<VarsData>(new VarsData());
  m_BlockData[m_CurBlockID]->m_Prev = elseBranch.get();
  TraverseStmt(CO->getTrueExpr());

  auto thenBranch = std::move(m_BlockData[m_CurBlockID]);
  m_BlockData[m_CurBlockID] = std::move(elseBranch);
  TraverseStmt(CO->getTrueExpr());

  merge(m_BlockData[m_CurBlockID].get(), thenBranch.get());
  return false;
}

bool TBRAnalyzer::TraverseBinaryOperator(BinaryOperator* BinOp) {
  const auto opCode = BinOp->getOpcode();
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();
  // Addition is not able to create any differential influence by itself so
  // markingMode should be left as it is. Similarly, addition does not affect
  // linearity so kNonLinearMode shouldn't be changed as well. The same applies
  // to subtraction.
  if (opCode == BO_Add || opCode == BO_Sub) {
    TraverseStmt(L);
    TraverseStmt(R);
  } else if (opCode == BO_Mul) {
    // Multiplication results in a linear expression if and only if one of the
    // factors is constant.
    Expr::EvalResult dummy;
    bool nonLinear = !clad_compat::Expr_EvaluateAsConstantExpr(
                         R, dummy, m_AnalysisDC->getASTContext()) &&
                     !clad_compat::Expr_EvaluateAsConstantExpr(
                         L, dummy, m_AnalysisDC->getASTContext());
    bool LHSIsStored =
        !utils::ShouldRecompute(L, m_AnalysisDC->getASTContext());
    bool RHSIsStored =
        !utils::ShouldRecompute(R, m_AnalysisDC->getASTContext());
    if (nonLinear)
      startNonLinearMode();

    if (LHSIsStored)
      setMode(/*mode=*/0);
    TraverseStmt(L);
    if (LHSIsStored)
      resetMode();

    if (RHSIsStored)
      setMode(/*mode=*/0);
    TraverseStmt(R);
    if (RHSIsStored)
      resetMode();

    if (nonLinear)
      resetMode();
  } else if (opCode == BO_Div) {
    // Division normally only results in a linear expression when the
    // denominator is constant.
    Expr::EvalResult dummy;
    bool nonLinear = !clad_compat::Expr_EvaluateAsConstantExpr(
        R, dummy, m_AnalysisDC->getASTContext());
    if (nonLinear)
      startNonLinearMode();
    bool LHSIsStored =
        !utils::ShouldRecompute(L, m_AnalysisDC->getASTContext());
    if (LHSIsStored)
      setMode(/*mode=*/0);
    TraverseStmt(L);
    if (LHSIsStored)
      resetMode();
    bool RHSIsStored = utils::UsefulToStore(R);
    if (RHSIsStored)
      setMode(/*mode=*/0);
    TraverseStmt(R);
    if (RHSIsStored)
      resetMode();
    if (nonLinear)
      resetMode();
  } else if (BinOp->isAssignmentOp()) {
    if (opCode == BO_Assign || opCode == BO_AddAssign ||
        opCode == BO_SubAssign) {
      bool isPointerOp = BinOp->getType()->isPointerType();
      // Since we only care about non-linear usages of variables, there is
      // no difference between operators =, -=, += in terms of TBR analysis.
      TraverseStmt(L);

      if (!isPointerOp)
        startMarkingMode();
      TraverseStmt(R);
      if (!isPointerOp)
        resetMode();
    } else if (opCode == BO_MulAssign || opCode == BO_DivAssign) {
      // *= (/=) normally only performs a linear operation if and only if
      // the RHS is constant. If RHS is not constant, 'x *= y' ('x /= y')
      // represents the same operation as 'x = x * y' ('x = x / y') and,
      // therefore, LHS has to be visited in kMarkingMode|kNonLinearMode.
      Expr::EvalResult dummy;
      bool RisNotConst = !clad_compat::Expr_EvaluateAsConstantExpr(
          R, dummy, m_AnalysisDC->getASTContext());
      if (RisNotConst)
        setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
      TraverseStmt(L);
      if (RisNotConst)
        resetMode();

      setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
      TraverseStmt(R);
      resetMode();
    }
    // If L should be recorded, mark its location.
    if (findReq(L))
      markLocation(L);
    // Set them to not required to store because the values were changed.
    // (if some value was not changed, this could only happen if it was
    // already not required to store).
    setIsRequired(L, /*isReq=*/false);
    // If we're in the non-linear marking mode, mark the LHS
    // (assignments act as references to the LHS).
    setIsRequired(L);
  } else if (opCode == BO_Comma) {
    setMode(/*mode=*/0);
    TraverseStmt(L);
    resetMode();

    TraverseStmt(R);
  }
  // else {
  // FIXME: add logic/bitwise/comparison operators
  // }
  return false;
}

bool TBRAnalyzer::TraverseCompoundAssignOperator(
    clang::CompoundAssignOperator* BinOp) {
  TBRAnalyzer::TraverseBinaryOperator(BinOp);
  return false;
}

bool TBRAnalyzer::TraverseUnaryOperator(clang::UnaryOperator* UnOp) {
  const auto opCode = UnOp->getOpcode();
  Expr* E = UnOp->getSubExpr();
  TraverseStmt(E);
  if (opCode == UO_PostInc || opCode == UO_PostDec || opCode == UO_PreInc ||
      opCode == UO_PreDec) {
    // If E should be recorded, mark its location.
    if (findReq(E))
      markLocation(E);
    if (m_ModifiedParams) {
      std::set<const clang::VarDecl*> vars;
      getDependencySet(E, vars);
      for (const VarDecl* VD : vars) {
        const auto* PVD = dyn_cast_or_null<ParmVarDecl>(VD);
        if (!VD || PVD)
          (*m_ModifiedParams)[m_Function].insert(PVD);
      }
    }
  }
  // FIXME: Ideally, `__real` and `__imag` operators should be treated as member
  // expressions. However, it is not clear where the FieldDecls of real and
  // imaginary parts should be deduced from (their names might be
  // compiler-specific).  So for now we visit the whole subexpression.
  return false;
}

bool TBRAnalyzer::TraverseCallExpr(clang::CallExpr* CE) {
  // FIXME: Currently TBR analysis just stops here and assumes that all the
  // variables passed by value/reference are used/used and changed. Analysis
  // could proceed to the function to analyse data flow inside it.
  FunctionDecl* FD = CE->getDirectCallee();
  // Use information about parameters assuming the analysis was performed.
  bool shouldAnalyzeParams = m_ModifiedParams && (m_ModifiedParams->find(FD) !=
                                                  m_ModifiedParams->end());
  bool hasHiddenParam = (CE->getNumArgs() != FD->getNumParams());
  std::size_t maxParamIdx = FD->getNumParams() - 1;
  setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
  bool nonDiff = utils::hasNonDifferentiableAttribute(CE);
  for (std::size_t i = hasHiddenParam, e = CE->getNumArgs(); i != e; ++i) {
    clang::Expr* arg = CE->getArg(i);
    const ParmVarDecl* par = nullptr;
    std::size_t paramIdx = std::min(i - hasHiddenParam, maxParamIdx);
    par = FD->getParamDecl(paramIdx);
    bool passByRef = false;
    if (par)
      passByRef = utils::isMemoryType(par->getType());
    bool paramUnused = false;
    if (shouldAnalyzeParams) {
      auto& usedParams = (*m_UsedParams)[FD];
      if (usedParams.find(par) == usedParams.end())
        paramUnused = true;
    }
    if (nonDiff)
      paramUnused = true;
    if (paramUnused)
      setMode(/*mode=*/0);
    TraverseStmt(arg);
    if (paramUnused)
      resetMode();
    if (passByRef) {
      bool paramModified = true;
      if (shouldAnalyzeParams) {
        auto& modifiedParams = (*m_ModifiedParams)[FD];
        if (modifiedParams.find(par) == modifiedParams.end())
          paramModified = false;
      }
      if (paramModified && findReq(arg)) {
        setIsRequired(arg, /*isReq=*/false);
        markLocation(arg);
      }
    }
  }

  auto* MD = dyn_cast<CXXMethodDecl>(FD);
  Expr* base = nullptr;
  if (MD && MD->isInstance() && !MD->isConst()) {
    if (auto* MCE = dyn_cast<CXXMemberCallExpr>(CE))
      base = MCE->getImplicitObjectArgument();
    else if (auto* OCE = dyn_cast<CXXOperatorCallExpr>(CE))
      base = OCE->getArg(0);
    base = base->IgnoreParenCasts();
  }

  if (base) {
    bool paramUnused = false;
    if (shouldAnalyzeParams) {
      auto& usedParams = (*m_UsedParams)[FD];
      if (usedParams.find(nullptr) == usedParams.end())
        paramUnused = true;
    }
    if (nonDiff)
      paramUnused = true;
    if (paramUnused)
      setMode(/*mode=*/0);
    TraverseStmt(base);
    if (paramUnused)
      resetMode();
    bool paramModified = true;
    if (shouldAnalyzeParams) {
      auto& modifiedParams = (*m_ModifiedParams)[FD];
      if (modifiedParams.find(nullptr) == modifiedParams.end())
        paramModified = false;
    }
    if (paramModified && findReq(base)) {
      markLocation(base);
      setIsRequired(base, /*isReq=*/false);
    }
  }

  resetMode();
  return false;
}

bool TBRAnalyzer::TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* CE) {
  TBRAnalyzer::TraverseCallExpr(CE);
  return false;
}

bool TBRAnalyzer::TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* CE) {
  TBRAnalyzer::TraverseCallExpr(CE);
  return false;
}

bool TBRAnalyzer::TraverseReturnStmt(clang::ReturnStmt* RS) {
  setMode(Mode::kMarkingMode);
  TraverseStmt(RS->getRetValue());
  resetMode();
  return false;
}

bool TBRAnalyzer::TraverseCXXConstructExpr(clang::CXXConstructExpr* CE) {
  // FIXME: Currently TBR analysis just stops here and assumes that all the
  // variables passed by value/reference are used/used and changed. Analysis
  // could proceed to the constructor to analyse data flow inside it.
  // FIXME: add support for default values
  FunctionDecl* FD = CE->getConstructor();
  setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    auto* arg = CE->getArg(i);
    bool passByRef = FD->getParamDecl(i)->getType()->isReferenceType();
    setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
    TraverseStmt(arg);
    resetMode();
    const auto* B = arg->IgnoreParenImpCasts();
    // FIXME: this supports only DeclRefExpr
    if (passByRef) {
      // Mark the args as required for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        m_TBRLocs.insert(arg);
        setIsRequired(arg, /*isReq=*/false);
      }
    }
  }
  resetMode();
  return false;
}

bool TBRAnalyzer::TraverseMemberExpr(clang::MemberExpr* ME) {
  setIsRequired(ME);
  return false;
}

bool TBRAnalyzer::TraverseCXXThisExpr(clang::CXXThisExpr* TE) {
  setIsRequired(TE);
  return false;
}

bool TBRAnalyzer::TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* ASE) {
  setMode(/*mode=*/0);
  TraverseStmt(ASE->getBase());
  resetMode();
  setIsRequired(ASE);
  setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
  TraverseStmt(ASE->getIdx());
  resetMode();
  return false;
}

bool TBRAnalyzer::TraverseInitListExpr(clang::InitListExpr* ILE) {
  setMode(Mode::kMarkingMode);
  for (auto* init : ILE->inits())
    TraverseStmt(init);
  resetMode();
  return false;
}
// NOLINTEND(cppcoreguidelines-pro-type-union-access)
} // end namespace clad
