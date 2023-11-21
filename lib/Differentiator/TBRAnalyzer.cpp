#include "clad/Differentiator/TBRAnalyzer.h"

using namespace clang;

namespace clad {

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
void TBRAnalyzer::setIsRequired(VarData& varData, bool isReq) {
  if (varData.type == VarData::FUND_TYPE)
    varData.val.m_FundData = isReq;
  else if (varData.type == VarData::OBJ_TYPE || varData.type == VarData::ARR_TYPE)
    for (auto& pair : *varData.val.m_ArrData)
      setIsRequired(pair.second, isReq);
  else if (varData.type == VarData::REF_TYPE && varData.val.m_RefData)
    if (auto* data = getExprVarData(varData.val.m_RefData)) {
      setIsRequired(*data, isReq);
    }
}

void TBRAnalyzer::merge(VarData& targetData, VarData& mergeData) {
  if (targetData.type == VarData::FUND_TYPE) {
    targetData.val.m_FundData =
        targetData.val.m_FundData || mergeData.val.m_FundData;
  } else if (targetData.type == VarData::OBJ_TYPE) {
    for (auto& pair : *targetData.val.m_ArrData)
      merge(pair.second, (*mergeData.val.m_ArrData)[pair.first]);
  } else if (targetData.type == VarData::ARR_TYPE) {
    /// FIXME: Currently non-constant indices are not supported in merging.
    for (auto& pair : *targetData.val.m_ArrData) {
      auto it = mergeData.val.m_ArrData->find(pair.first);
      if (it != mergeData.val.m_ArrData->end())
        merge(pair.second, it->second);
    }
    for (auto& pair : *mergeData.val.m_ArrData) {
      auto it = targetData.val.m_ArrData->find(pair.first);
      if (it == mergeData.val.m_ArrData->end())
        (*targetData.val.m_ArrData)[pair.first] = copy(pair.second);
    }
  }
  /// This might be useful in future if used to analyse pointers. However, for
  /// now it's only used for references for which merging doesn't make sense.
  // else if (this.type == VarData::REF_TYPE) {}
}

TBRAnalyzer::VarData TBRAnalyzer::copy(VarData& copyData) {
  VarData res;
  res.type = copyData.type;
  if (copyData.type == VarData::FUND_TYPE) {
    res.val.m_FundData = copyData.val.m_FundData;
  } else if (copyData.type == VarData::OBJ_TYPE || copyData.type == VarData::ARR_TYPE) {
    res.val.m_ArrData = std::unique_ptr<ArrMap>(new ArrMap());
    for (auto& pair : *copyData.val.m_ArrData)
      (*res.val.m_ArrData)[pair.first] = copy(pair.second);
  } else if (copyData.type == VarData::REF_TYPE && copyData.val.m_RefData) {
    res.val.m_RefData = copyData.val.m_RefData;
  }
  return res;
}

bool TBRAnalyzer::findReq(const VarData& varData) {
  if (varData.type == VarData::FUND_TYPE)
    return varData.val.m_FundData;
  if (varData.type == VarData::OBJ_TYPE || varData.type == VarData::ARR_TYPE) {
    for (auto& pair : *varData.val.m_ArrData)
      if (findReq(pair.second))
        return true;
  } else if (varData.type == VarData::REF_TYPE && varData.val.m_RefData) {
    if (auto* data = getExprVarData(varData.val.m_RefData)) {
      if (findReq(*data)) {
        return true;
      }
    }
  }
  return false;
}

void TBRAnalyzer::overlay(
    VarData& targetData,
    llvm::SmallVector<ProfileID, 2>& IDSequence, size_t i) {
  if (i == 0) {
    setIsRequired(targetData);
    return;
  }
  --i;
  ProfileID& curID = IDSequence[i];
  // non-constant indices are represented with default ID.
  ProfileID nonConstIdxID;
  if (curID == nonConstIdxID) {
    for (auto& pair : *targetData.val.m_ArrData)
      overlay(pair.second, IDSequence, i);
  } else {
    overlay((*targetData.val.m_ArrData)[curID], IDSequence, i);
  }
}

TBRAnalyzer::VarData* TBRAnalyzer::getMemberVarData(const clang::MemberExpr* ME,
                                                    bool addNonConstIdx) {
  if (const auto* FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
    const auto* base = ME->getBase();
    VarData* baseData = getExprVarData(base);

    if (!baseData)
      return nullptr;

    /// if non-const index was found and it is not supposed to be added just
    /// return the current VarData*.
    if (m_NonConstIndexFound && !addNonConstIdx)
      return baseData;

    return &(*baseData->val.m_ArrData)[getProfileID(FD)];
  }
  return nullptr;
}

TBRAnalyzer::VarData*
TBRAnalyzer::getArrSubVarData(const clang::ArraySubscriptExpr* ASE,
                              bool addNonConstIdx) {
  const auto* idxExpr = ASE->getIdx();
  ProfileID idxID;
  if (const auto* IL = dyn_cast<IntegerLiteral>(idxExpr)) {
    idxID = getProfileID(IL);
  } else {
    m_NonConstIndexFound = true;
    /// Non-const indices are represented with default FoldingSetNodeID.
  }

  const auto* base = ASE->getBase()->IgnoreImpCasts();
  VarData* baseData = getExprVarData(base);

  if (!baseData)
    return nullptr;

  /// if non-const index was found and it is not supposed to be added just
  /// return the current VarData*.
  if (m_NonConstIndexFound && !addNonConstIdx)
    return baseData;

  auto* baseArrMap = baseData->val.m_ArrData.get();
  auto it = baseArrMap->find(idxID);

  /// Add the current index if it was not added previously
  if (it == baseArrMap->end()) {
    auto& idxData = (*baseArrMap)[idxID];
    /// Since default ID represents non-const indices, whenever we add a new
    /// index we have to copy the VarData of default ID's element (if an element
    /// with undefined index was used this might be our current element).
    ProfileID nonConstIdxID;
    idxData = copy((*baseArrMap)[nonConstIdxID]);
    return &idxData;
  }

  return &it->second;
}

TBRAnalyzer::VarData* TBRAnalyzer::getExprVarData(const clang::Expr* E,
                                                  bool addNonConstIdx) {
  /// This line is necessary for pointer member expressions (in 'x->y'
  /// x would be implicitly casted with the * operator).
  E = E->IgnoreImpCasts();
  VarData* EData = nullptr;
  if (isa<clang::DeclRefExpr>(E) || isa<clang::CXXThisExpr>(E)) {
    const VarDecl* VD = nullptr;
    /// ``this`` does not have a declaration so it is represented with nullptr.
    if (const auto* DRE = dyn_cast<clang::DeclRefExpr>(E))
      VD = dyn_cast<clang::VarDecl>(DRE->getDecl());
    auto* branch = &getCurBlockVarsData();
    while (branch) {
      auto it = branch->find(VD);
      if (it != branch->end()) {
        EData = &it->second;
        break;
      }
      branch = branch->prev;
    }
  }
  if (const auto* ME = dyn_cast<clang::MemberExpr>(E))
    EData = getMemberVarData(ME, addNonConstIdx);
  if (const auto* ASE = dyn_cast<clang::ArraySubscriptExpr>(E))
    EData = getArrSubVarData(ASE, addNonConstIdx);

  if (EData && EData->type == VarData::REF_TYPE && EData->val.m_RefData)
    EData = getExprVarData(EData->val.m_RefData);

  return EData;
}

TBRAnalyzer::VarData::VarData(const QualType QT) {
  if (QT->isReferenceType()) {
    type = VarData::REF_TYPE;
    val.m_RefData = nullptr;
  } else if (utils::isArrayOrPointerType(QT)) {
    type = VarData::ARR_TYPE;
    val.m_ArrData = std::unique_ptr<ArrMap>(new ArrMap());
    const Type* elemType;
    if (const auto* const pointerType = llvm::dyn_cast<clang::PointerType>(QT))
      elemType = pointerType->getPointeeType().getTypePtrOrNull();
    else
      elemType = QT->getArrayElementTypeNoTypeQual();
    ProfileID nonConstIdxID;
    auto& idxData = (*val.m_ArrData)[nonConstIdxID];
    idxData = VarData (QualType::getFromOpaquePtr(elemType));
  } else if (QT->isBuiltinType()) {
    type = VarData::FUND_TYPE;
    val.m_FundData = false;
  } else if (QT->isRecordType()) {
    type = VarData::OBJ_TYPE;
    const auto* recordDecl = QT->getAs<RecordType>()->getDecl();
    auto& newArrMap = val.m_ArrData;
    newArrMap = std::unique_ptr<ArrMap>(new ArrMap());
    for (const auto* field : recordDecl->fields()) {
      const auto varType = field->getType();
      (*newArrMap)[getProfileID(field)] = VarData(varType);
    }
  }
}

void TBRAnalyzer::overlay(const clang::Expr* E) {
  m_NonConstIndexFound = false;
  llvm::SmallVector<ProfileID, 2> IDSequence;
  const clang::DeclRefExpr* innermostDRE;
  bool cond = true;
  /// Unwrap the given expression to a vector of indices and fields.
  while (cond) {
    E = E->IgnoreImplicit();
    if (const auto* ASE = dyn_cast<clang::ArraySubscriptExpr>(E)) {
      if (const auto* IL = dyn_cast<clang::IntegerLiteral>(ASE->getIdx()))
        IDSequence.push_back(getProfileID(IL));
      else
        IDSequence.push_back(ProfileID());
      E = ASE->getBase();
    } else if (const auto* ME = dyn_cast<clang::MemberExpr>(E)) {
      if (const auto* FD = dyn_cast<clang::FieldDecl>(ME->getMemberDecl()))
        IDSequence.push_back(getProfileID(FD));
      E = ME->getBase();
    } else if (isa<clang::DeclRefExpr>(E)) {
      innermostDRE = dyn_cast<clang::DeclRefExpr>(E);
      cond = false;
    } else
      return;
  }

  /// Overlay on all the VarData's recursively.
  if (const auto* VD = dyn_cast<clang::VarDecl>(innermostDRE->getDecl())) {
    overlay(getCurBlockVarsData()[VD], IDSequence, IDSequence.size());
  }
}
// NOLINTEND(cppcoreguidelines-pro-type-union-access)

void TBRAnalyzer::addVar(const clang::VarDecl* VD) {
  /// If a declaration is passed a second time (meaning it is inside a loop),
  /// treat it as an assigment operation.
  /// FIXME: this marks the SourceLocation of DeclStmt which doesn't work for
  /// declarations with multiple VarDecls.
  auto& curBranch = getCurBlockVarsData();

  auto* branch = curBranch.prev;
  while (branch) {
    auto it = branch->find(VD);
    if (it != branch->end()) {
      curBranch[VD] = copy(it->second);
      return;
    }
    branch = branch->prev;
  }

  QualType varType;
  if (const auto* arrayParam = dyn_cast<ParmVarDecl>(VD))
    varType = arrayParam->getOriginalType();
  else
    varType = VD->getType();
  /// If varType represents auto or auto*, get the type of init.
  if (utils::IsAutoOrAutoPtrType(varType))
    varType = VD->getInit()->getType();

  /// FIXME: If the pointer points to an object we represent it with a
  /// OBJ_TYPE VarData. This is done for '_d_this' pointer to be processed
  /// correctly in hessian mode. This should be removed once full support for
  /// pointers in analysis is introduced.
  if (const auto* const pointerType = dyn_cast<clang::PointerType>(varType)) {
    const auto* elemType = pointerType->getPointeeType().getTypePtrOrNull();
    if (elemType && elemType->isRecordType()) {
      curBranch[VD] = VarData(QualType::getFromOpaquePtr(elemType));
      return;
    }
  }
  curBranch[VD] = VarData(varType);
}

void TBRAnalyzer::markLocation(const clang::Expr* E) {
  VarData* data = getExprVarData(E);
  if (!data || findReq(*data)) {
    /// FIXME: If any of the data's child nodes are required to store then data
    /// itself is stored. We might add an option to store separate fields.
    /// FIXME: Sometimes one location might correspond to multiple stores.
    /// For example, in ``(x*=y)=u`` x's location will first be marked as
    /// required to be stored (when passing *= operator) but then marked as not
    /// required to be stored (when passing = operator). Current method of
    /// marking locations does not allow to differentiate between these two.
    m_TBRLocs.insert(E->getBeginLoc());
  }
}

void TBRAnalyzer::setIsRequired(const clang::Expr* E, bool isReq) {
  if (!isReq ||
      (m_ModeStack.back() == (Mode::kMarkingMode | Mode::kNonLinearMode))) {
    VarData* data = getExprVarData(E, /*addNonConstIdx=*/isReq);
    if (data && (isReq || !m_NonConstIndexFound))
      setIsRequired(*data, isReq);
    /// If an array element with a non-const element is set to required
    /// all the elements of that array should be set to required.
    if (isReq && m_NonConstIndexFound)
      overlay(E);
    m_NonConstIndexFound = false;
  }
}

void TBRAnalyzer::Analyze(const FunctionDecl* FD) {
  /// Build the CFG (control-flow graph) of FD.
  clang::CFG::BuildOptions Options;
  m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), &m_Context, Options);

  m_BlockData.resize(m_CFG->size());
  m_BlockPassCounter.resize(m_CFG->size(), 0);

  /// Set current block ID to the ID of entry the block.
  auto* entry = &m_CFG->getEntry();
  m_CurBlockID = entry->getBlockID();
  m_BlockData[m_CurBlockID] = std::unique_ptr<VarsData>(new VarsData());

  /// If we are analysing a non-static method, add a VarData for 'this' pointer
  /// (it is represented with nullptr).
  const auto* MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && !MD->isStatic()) {
    const Type* recordType =
        MD->getParent()->getTypeForDecl();
    getCurBlockVarsData()[nullptr] =
        VarData(QualType::getFromOpaquePtr(recordType));
  }
  auto paramsRef = FD->parameters();
  for (std::size_t i = 0; i < FD->getNumParams(); ++i)
    addVar(paramsRef[i]);
  /// Add the entry block to the queue.
  m_CFGQueue.insert(m_CurBlockID);

  /// Visit CFG blocks in the queue until it's empty.
  while (!m_CFGQueue.empty()) {
    auto IDIter = std::prev(m_CFGQueue.end());
    m_CurBlockID = *IDIter;
    m_CFGQueue.erase(IDIter);

    CFGBlock& nextBlock = *getCFGBlockByID(m_CurBlockID);
    VisitCFGBlock(nextBlock);
  }
  //    for (int id = m_CurBlockID; id >= 0; --id) {
  //        llvm::errs() << "\n-----BLOCK" << id << "-----\n\n";
  //        for (auto succ : getCFGBlockByID(id)->succs()) {
  //            if (succ)
  //            llvm::errs() << "successor: " << succ->getBlockID() << "\n";
  //        }
  //    }
}

void TBRAnalyzer::VisitCFGBlock(const CFGBlock& block) {
  //    llvm::errs() << "\n-----BLOCK" << block->getBlockID() << "-----\n";
  /// Visiting loop blocks just once is not enough since the end of one
  /// loop iteration may have an effect on the next one. However, two
  /// iterations is always enough. Allow a third visit without going to
  /// successors to correctly analyse loop conditions.
  bool notLastPass = ++m_BlockPassCounter[block.getBlockID()] <= 2;

  /// Visit all the statements inside the block.
  for (const clang::CFGElement& Element : block) {
    if (Element.getKind() == clang::CFGElement::Statement) {
      const clang::Stmt* S = Element.castAs<clang::CFGStmt>().getStmt();
      TraverseStmt(const_cast<clang::Stmt*>(S));
    }
  }

  /// Traverse successor CFG blocks.
  for (const auto succ : block.succs()) {
    /// Sometimes clang CFG does not create blocks for parts of code that
    /// are never executed (e.g. 'if (0) {...'). Add this check for safety.
    if (!succ)
      continue;
    auto& varsData = m_BlockData[succ->getBlockID()];

    /// Create VarsData for the succ branch if it hasn't been done previously.
    /// If the successor doesn't have a VarsData, assign it and attach the
    /// current block as previous.
    if (!varsData) {
      varsData = std::unique_ptr<VarsData>(new VarsData());
      varsData->prev = m_BlockData[block.getBlockID()].get();
    }

    /// If this is the third (last) pass of block, it means block represents
    /// a loop condition and the loop body has already been visited 2 times.
    /// This means we should not visit the loop body anymore.
    if (notLastPass) {
        /// Add the successor to the queue.
        m_CFGQueue.insert(succ->getBlockID());

        /// This part is necessary for loops. For other cases, this is not
        /// supposed to do anything.
        if (succ->getBlockID() < block.getBlockID()) {
            /// If there is another loop condition present inside a loop,
            /// We have to set it's loop pass counter to 0 (it might be 3
            /// from the previous outer loop pass).
            m_BlockPassCounter[succ->getBlockID()] = 0;
            /// Remove VarsData left after the previous pass.
            varsData->clear();
        }
    }

    /// If the successor's previous block is not this one,
    /// perform a merge.
    if (varsData->prev != m_BlockData[block.getBlockID()].get())
      merge(varsData.get(), m_BlockData[block.getBlockID()].get());
  }
  //    llvm::errs() << "----------------\n\n";
}

CFGBlock* TBRAnalyzer::getCFGBlockByID(unsigned ID) {
  return *(m_CFG->begin() + ID);
}

TBRAnalyzer::VarsData*
TBRAnalyzer::findLowestCommonAncestor(VarsData* varsData1,
                                      VarsData* varsData2) {
  VarsData* pred1 = varsData1;
  VarsData* pred2 = varsData2;
  while (true) {
    if (pred1 == pred2)
      return pred1;

    auto* branch = varsData1;
    while (branch != pred1) {
      if (branch == pred2)
        return branch;
      branch = branch->prev;
    }

    branch = varsData2;
    while (branch != pred2) {
      if (branch == pred1)
        return branch;
      branch = branch->prev;
    }

    if (pred1->prev) {
      pred1 = pred1->prev;
      /// This ensures we don't get an infinite loop because of VarsData being
      /// connected in a loop themselves.
      if (pred1 == varsData1)
        return nullptr;
    } else {
      /// pred1 not having a predecessor means it is corresponds to the entry
      /// block and, therefore it is the lowest common ancestor.
      return pred1;
    }

    if (pred2->prev) {
      pred2 = pred2->prev;
      /// This ensures we don't get an infinite loop because of VarsData being
      /// connected in a loop themselves.
      if (pred2 == varsData2)
        return nullptr;
    } else {
      /// pred2 not having a predecessor means it is corresponds to the entry
      /// block and, therefore it is the lowest common ancestor.
      return pred2;
    }
  }
}

std::unordered_map<const clang::VarDecl*, TBRAnalyzer::VarData*>
TBRAnalyzer::collectDataFromPredecessors(VarsData* varsData,
                                         TBRAnalyzer::VarsData* limit) {
  std::unordered_map<const clang::VarDecl*, VarData*> result;
  if (varsData != limit) {
    /// Copy data from every predecessor.
    for (auto* pred = varsData->prev; pred != limit; pred = pred->prev) {
      /// If a variable from 'pred' is not present
      /// in 'result', place it in there.
      for (auto& pair : *pred)
        if (result.find(pair.first) == result.end()) {
          result[pair.first] = &pair.second;
        }
    }
  }

  return result;
}

void TBRAnalyzer::merge(VarsData* targetData, VarsData* mergeData) {
  auto* LCA = findLowestCommonAncestor(targetData, mergeData);
  auto collectedMergeData =
      collectDataFromPredecessors(mergeData, /*limit=*/LCA);
  auto collectedTargetData = collectDataFromPredecessors(targetData, /*limit=*/LCA);

  /// For every variable in 'collectedMergeData', search it in targetData
  /// and all its predecessors (if found in a predecessor, make a copy to
  /// targetData).
  for (auto& pair : collectedMergeData) {
    VarData* found = nullptr;
    auto elemSearch = targetData->find(pair.first);
    if (elemSearch == targetData->end()) {
      auto* branch = targetData->prev;
      while (branch) {
        auto it = branch->find(pair.first);
        if (it != branch->end()) {
          (*targetData)[pair.first] = copy(it->second);
          found = &(*targetData)[pair.first];
          break;
        }
        branch = branch->prev;
      }
    } else {
      found = &elemSearch->second;
    }

    /// If the variable was found, perform a merge.
    /// Else, just copy it from collectedMergeData.
    if (found) {
      merge(*found, *pair.second);
    } else
      (*targetData)[pair.first] = copy(*pair.second);
  }

  /// For every variable in collectedTargetData, search it inside
  /// collectedMergeData. If it's not found, that means it
  /// was not used anywhere between LCA and mergeData.
  /// To correctly merge, we have to take it from LCA's
  /// predecessors and merge it to targetData.
  for (auto& pair : collectedTargetData) {
    auto elemSearch = collectedMergeData.find(pair.first);
    if (elemSearch == collectedMergeData.end()) {
      auto* branch = LCA;
      while (branch) {
        auto it = branch->find(pair.first);
        if (it != branch->end()) {
          merge(*pair.second, it->second);
          break;
        }
        branch = branch->prev;
      }
    }
  }
}

bool TBRAnalyzer::VisitDeclRefExpr(DeclRefExpr* DRE) {
  if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    auto& curBranch = getCurBlockVarsData();
    if (curBranch.find(VD) == curBranch.end())
      addVar(VD);
  }

  setIsRequired(DRE);

  return true;
}

bool TBRAnalyzer::VisitDeclStmt(DeclStmt* DS) {
  for (auto* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      addVar(VD);
      if (clang::Expr* init = VD->getInit()) {
        setMode(Mode::kMarkingMode);
        TraverseStmt(init);
        resetMode();
        auto& VDExpr = getCurBlockVarsData()[VD];
        /// if the declared variable is ref type attach its VarData to the
        /// VarData of the RHS variable.
        llvm::SmallVector<Expr*, 4> ExprsToStore;
        utils::GetInnermostReturnExpr(init, ExprsToStore);
        if (VDExpr.type == VarData::REF_TYPE && !ExprsToStore.empty())
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
          VDExpr.val.m_RefData = ExprsToStore[0];
      }
    }
  }
  return true;
}

bool TBRAnalyzer::VisitConditionalOperator(clang::ConditionalOperator* CO) {
  setMode(0);
  TraverseStmt(CO->getCond());
  resetMode();

  auto elseBranch = std::move(m_BlockData[m_CurBlockID]);

  m_BlockData[m_CurBlockID] = std::unique_ptr<VarsData>(new VarsData());
  m_BlockData[m_CurBlockID]->prev = elseBranch.get();
  TraverseStmt(CO->getTrueExpr());

  auto thenBranch = std::move(m_BlockData[m_CurBlockID]);
  m_BlockData[m_CurBlockID] = std::move(elseBranch);
  TraverseStmt(CO->getTrueExpr());

  merge(m_BlockData[m_CurBlockID].get(), thenBranch.get());
  return true;
}

bool TBRAnalyzer::VisitBinaryOperator(BinaryOperator* BinOp) {
  const auto opCode = BinOp->getOpcode();
  Expr* L = BinOp->getLHS();
  Expr* R = BinOp->getRHS();
  /// Addition is not able to create any differential influence by itself so
  /// markingMode should be left as it is. Similarly, addition does not affect
  /// linearity so kNonLinearMode shouldn't be changed as well. The same applies
  /// to subtraction.
  if (opCode == BO_Add || opCode == BO_Sub) {
    TraverseStmt(L);
    TraverseStmt(R);
  } else if (opCode == BO_Mul) {
    /// Multiplication results in a linear expression if and only if one of the
    /// factors is constant.
    Expr::EvalResult dummy;
    bool nonLinear =
        !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, m_Context) &&
        !clad_compat::Expr_EvaluateAsConstantExpr(L, dummy, m_Context);
    if (nonLinear)
      startNonLinearMode();

    TraverseStmt(L);
    TraverseStmt(R);

    if (nonLinear)
      resetMode();
  } else if (opCode == BO_Div) {
    /// Division normally only results in a linear expression when the
    /// denominator is constant.
    Expr::EvalResult dummy;
    bool nonLinear =
        !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, m_Context);
    if (nonLinear)
      startNonLinearMode();

    TraverseStmt(L);
    TraverseStmt(R);

    if (nonLinear)
      resetMode();
  } else if (BinOp->isAssignmentOp()) {
    if (opCode == BO_Assign || opCode == BO_AddAssign ||
        opCode == BO_SubAssign) {
      /// Since we only care about non-linear usages of variables, there is
      /// no difference between operators =, -=, += in terms of TBR analysis.
      TraverseStmt(L);

      startMarkingMode();
      TraverseStmt(R);
      resetMode();
    } else if (opCode == BO_MulAssign || opCode == BO_DivAssign) {
      /// *= (/=) normally only performs a linear operation if and only if
      /// the RHS is constant. If RHS is not constant, 'x *= y' ('x /= y')
      /// represents the same operation as 'x = x * y' ('x = x / y') and,
      /// therefore, LHS has to be visited in kMarkingMode|kNonLinearMode.
      Expr::EvalResult dummy;
      bool RisNotConst =
          !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, m_Context);
      if (RisNotConst)
        setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
      TraverseStmt(L);
      if (RisNotConst)
        resetMode();

      setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
      TraverseStmt(R);
      resetMode();
    }
    llvm::SmallVector<Expr*, 4> ExprsToStore;
    utils::GetInnermostReturnExpr(L, ExprsToStore);
    for (const auto* innerExpr : ExprsToStore) {
      /// Mark corresponding SourceLocation as required/not required to be
      /// stored for all expressions that could be used changed.
      markLocation(innerExpr);
      /// Set them to not required to store because the values were changed.
      /// (if some value was not changed, this could only happen if it was
      /// already not required to store).
      setIsRequired(innerExpr, /*isReq=*/false);
    }
  } else if (opCode == BO_Comma) {
    setMode(0);
    TraverseStmt(L);
    resetMode();

    TraverseStmt(R);
  }
  // else {
  // FIXME: add logic/bitwise/comparison operators
  // }
  return true;
}

bool TBRAnalyzer::VisitUnaryOperator(clang::UnaryOperator* UnOp) {
  const auto opCode = UnOp->getOpcode();
  Expr* E = UnOp->getSubExpr();
  TraverseStmt(E);
  if (opCode == UO_PostInc || opCode == UO_PostDec || opCode == UO_PreInc ||
      opCode == UO_PreDec) {
    // FIXME: this doesn't support all the possible references
    /// Mark corresponding SourceLocation as required/not required to be
    /// stored for all expressions that could be used in this operation.
    llvm::SmallVector<Expr*, 4> ExprsToStore;
    utils::GetInnermostReturnExpr(E, ExprsToStore);
    for (const auto* innerExpr : ExprsToStore) {
      /// Mark corresponding SourceLocation as required/not required to be
      /// stored for all expressions that could be changed.
      markLocation(innerExpr);
    }
  }
  return true;
}

bool TBRAnalyzer::VisitCallExpr(clang::CallExpr* CE) {
  /// FIXME: Currently TBR analysis just stops here and assumes that all the
  /// variables passed by value/reference are used/used and changed. Analysis
  /// could proceed to the function to analyse data flow inside it.
  FunctionDecl* FD = CE->getDirectCallee();
  bool noHiddenParam = (CE->getNumArgs() == FD->getNumParams());
  setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    clang::Expr* arg = CE->getArg(i);
    bool passByRef = false;
    if (noHiddenParam)
      passByRef = FD->getParamDecl(i)->getType()->isReferenceType();
    else if (i!=0)
      passByRef = FD->getParamDecl(i - 1)->getType()->isReferenceType();
    setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
    TraverseStmt(arg);
    resetMode();
    const auto* B = arg->IgnoreParenImpCasts();
    // FIXME: this supports only DeclRefExpr
    if (passByRef) {
      /// Mark SourceLocation as required to store for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        m_TBRLocs.insert(arg->getBeginLoc());
        setIsRequired(arg, /*isReq=*/false);
      }
    }
  }
  resetMode();
  return true;
}

bool TBRAnalyzer::VisitCXXConstructExpr(clang::CXXConstructExpr* CE) {
  /// FIXME: Currently TBR analysis just stops here and assumes that all the
  /// variables passed by value/reference are used/used and changed. Analysis
  /// could proceed to the constructor to analyse data flow inside it.
  /// FIXME: add support for default values
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
      /// Mark SourceLocation as required for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        m_TBRLocs.insert(arg->getBeginLoc());
        setIsRequired(arg, /*isReq=*/false);
      }
    }
  }
  resetMode();
  return true;
}

bool TBRAnalyzer::VisitMemberExpr(clang::MemberExpr* ME) {
  setIsRequired(ME);
  return true;
}

bool TBRAnalyzer::VisitArraySubscriptExpr(clang::ArraySubscriptExpr* ASE) {
  setMode(0);
  TraverseStmt(ASE->getBase());
  resetMode();
  setIsRequired(ASE);
  setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
  TraverseStmt(ASE->getIdx());
  resetMode();
  return true;
}

bool TBRAnalyzer::VisitInitListExpr(clang::InitListExpr* ILE) {
  setMode(Mode::kMarkingMode);
  for (auto* init : ILE->inits())
    TraverseStmt(init);
  resetMode();
  return true;
}

} // end namespace clad
