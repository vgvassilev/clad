#include "clad/Differentiator/TBRAnalyzer.h"

using namespace clang;

namespace clad {

void TBRAnalyzer::setIsRequired(VarData* varData, bool isReq) {
  if (varData->type == VarData::FUND_TYPE)
    varData->val.fundData = isReq;
  else if (varData->type == VarData::OBJ_TYPE)
    for (auto& pair : *varData->val.objData)
      setIsRequired(pair.second, isReq);
  else if (varData->type == VarData::ARR_TYPE)
    for (auto& pair : *varData->val.arrData)
      setIsRequired(pair.second, isReq);
  else if (varData->type == VarData::REF_TYPE && varData->val.refData)
    setIsRequired(getExprVarData(varData->val.refData), isReq);
}

void TBRAnalyzer::merge(VarData* targetData, VarData* mergeData) {
  if (targetData->type == VarData::FUND_TYPE) {
    targetData->val.fundData =
        targetData->val.fundData || mergeData->val.fundData;
  } else if (targetData->type == VarData::OBJ_TYPE) {
    for (auto& pair : *targetData->val.objData)
      merge(pair.second, (*mergeData->val.objData)[pair.first]);
  } else if (targetData->type == VarData::ARR_TYPE) {
    /// FIXME: Currently non-constant indices are not supported in merging.
    for (auto& pair : *targetData->val.arrData) {
      auto it = mergeData->val.arrData->find(pair.first);
      if (it != mergeData->val.arrData->end())
        merge(pair.second, it->second);
    }
    for (auto& pair : *mergeData->val.arrData) {
      auto it = targetData->val.arrData->find(pair.first);
      if (it == mergeData->val.arrData->end())
        (*targetData->val.arrData)[pair.first] = copy(pair.second);
    }
  }
  /// This might be useful in future if used to analyse pointers. However, for
  /// now it's only used for references for which merging doesn't make sense.
  // else if (this->type == VarData::REF_TYPE) {}
}

TBRAnalyzer::VarData* TBRAnalyzer::copy(VarData* copyData) {
  auto* res = new VarData();
  res->type = copyData->type;
  if (copyData->type == VarData::FUND_TYPE) {
    res->val.fundData = copyData->val.fundData;
  } else if (copyData->type == VarData::OBJ_TYPE) {
    res->val.objData = new ObjMap();
    for (auto& pair : *copyData->val.objData)
      (*res->val.objData)[pair.first] = copy(pair.second);
  } else if (copyData->type == VarData::ARR_TYPE) {
    res->val.arrData = new ArrMap();
    for (auto& pair : *copyData->val.arrData)
      (*res->val.arrData)[pair.first] = copy(pair.second);
  } else if (copyData->type == VarData::REF_TYPE && copyData->val.refData) {
    res->val.refData = copyData->val.refData;
  }
  return res;
}

bool TBRAnalyzer::findReq(const VarData* varData) {
  if (varData->type == VarData::FUND_TYPE)
    return varData->val.fundData;
  if (varData->type == VarData::OBJ_TYPE) {
    for (auto& pair : *varData->val.objData)
      if (findReq(pair.second))
        return true;
  } else if (varData->type == VarData::ARR_TYPE) {
    for (auto& pair : *varData->val.arrData)
      if (findReq(pair.second))
        return true;
  } else if (varData->type == VarData::REF_TYPE && varData->val.refData) {
    if (findReq(getExprVarData(varData->val.refData)))
      return true;
  }
  return false;
}

void TBRAnalyzer::overlay(
    VarData* targetData,
    llvm::SmallVector<IdxOrMember, 2>& IdxAndMemberSequence, size_t i) {
  if (i == 0) {
    setIsRequired(targetData);
    return;
  }
  --i;
  IdxOrMember& curIdxOrMember = IdxAndMemberSequence[i];
  if (curIdxOrMember.type == IdxOrMember::IdxOrMemberType::FIELD) {
    overlay((*targetData->val.objData)[curIdxOrMember.val.field],
            IdxAndMemberSequence, i);
  } else if (curIdxOrMember.type == IdxOrMember::IdxOrMemberType::INDEX) {
    auto idx = curIdxOrMember.val.index;
    if (eqAPInt(idx, llvm::APInt(2, -1, true)))
      for (auto& pair : *targetData->val.arrData)
        overlay(pair.second, IdxAndMemberSequence, i);
    else
      overlay((*targetData->val.arrData)[idx], IdxAndMemberSequence, i);
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
    if (nonConstIndexFound && !addNonConstIdx)
      return baseData;

    return (*baseData->val.objData)[FD];
  }
  return nullptr;
}

TBRAnalyzer::VarData*
TBRAnalyzer::getArrSubVarData(const clang::ArraySubscriptExpr* ASE,
                              bool addNonConstIdx) {
  const auto* idxExpr = ASE->getIdx();
  llvm::APInt idx;
  if (const auto* IL = dyn_cast<IntegerLiteral>(idxExpr)) {
    idx = IL->getValue();
  } else {
    nonConstIndexFound = true;
    /// Non-const indices are represented with -1.
    idx = llvm::APInt(2, -1, true);
  }

  const auto* base = ASE->getBase()->IgnoreImpCasts();
  VarData* baseData = getExprVarData(base);

  if (!baseData)
    return nullptr;

  /// if non-const index was found and it is not supposed to be added just
  /// return the current VarData*.
  if (nonConstIndexFound && !addNonConstIdx)
    return baseData;

  auto* baseArrMap = baseData->val.arrData;
  auto it = baseArrMap->find(idx);

  /// Add the current index if it was not added previously
  if (it == baseArrMap->end()) {
    auto*& idxData = (*baseArrMap)[idx];
    /// Since -1 represents non-const indices, whenever we add a new index we
    /// have to copy the VarData of -1's element (if an element with undefined
    /// index was used this might be our current element).
    idxData = copy((*baseArrMap)[llvm::APInt(2, -1, true)]);
    return idxData;
  }

  return it->second;
}

TBRAnalyzer::VarData* TBRAnalyzer::getExprVarData(const clang::Expr* E,
                                                  bool addNonConstIdx) {
  /// This line is necessary for pointer member expressions (in 'x->y'
  /// x would be implicitly casted with the * operator).
  E = E->IgnoreImpCasts();
  VarData* EData;
  if (isa<clang::DeclRefExpr>(E) || isa<clang::CXXThisExpr>(E)) {
    const VarDecl* VD = nullptr;
    /// ``this`` does not have a declaration so it is represented with nullptr.
    if (const auto* DRE = dyn_cast<clang::DeclRefExpr>(E))
      VD = dyn_cast<clang::VarDecl>(DRE->getDecl());
    auto* branch = &getCurBranch();
    while (branch) {
      auto it = branch->find(VD);
      if (it != branch->end()) {
        EData = it->second;
        break;
      }
      branch = branch->prev;
    }
  }
  if (const auto* ME = dyn_cast<clang::MemberExpr>(E))
    EData = getMemberVarData(ME, addNonConstIdx);
  if (const auto* ASE = dyn_cast<clang::ArraySubscriptExpr>(E))
    EData = getArrSubVarData(ASE, addNonConstIdx);

  if (EData && EData->type == VarData::REF_TYPE && EData->val.refData)
    EData = getExprVarData(EData->val.refData);

  return EData;
}

TBRAnalyzer::VarData::VarData(const QualType QT) {
  if (QT->isReferenceType()) {
    type = VarData::REF_TYPE;
    val.refData = nullptr;
  } else if (utils::isArrayOrPointerType(QT)) {
    type = VarData::ARR_TYPE;
    val.arrData = new ArrMap();
    const Type* elemType;
    if (const auto pointerType = llvm::dyn_cast<clang::PointerType>(QT))
      elemType = pointerType->getPointeeType().getTypePtrOrNull();
    else
      elemType = QT->getArrayElementTypeNoTypeQual();
    auto& idxData = (*val.arrData)[llvm::APInt(2, -1, true)];
    idxData = new VarData(QualType::getFromOpaquePtr(elemType));
  } else if (QT->isBuiltinType()) {
    type = VarData::FUND_TYPE;
    val.fundData = false;
  } else if (QT->isRecordType()) {
    type = VarData::OBJ_TYPE;
    const auto* recordDecl = QT->getAs<RecordType>()->getDecl();
    auto& newObjMap = val.objData;
    newObjMap = new ObjMap();
    for (const auto* field : recordDecl->fields()) {
      const auto varType = field->getType();
      (*newObjMap)[field] = new VarData(varType);
    }
  }
}

void TBRAnalyzer::overlay(const clang::Expr* E) {
  nonConstIndexFound = false;
  llvm::SmallVector<IdxOrMember, 2> IdxAndMemberSequence;
  const clang::DeclRefExpr* innermostDRE;
  bool cond = true;
  /// Unwrap the given expression to a vector of indices and fields.
  while (cond) {
    E = E->IgnoreImplicit();
    if (const auto* ASE = dyn_cast<clang::ArraySubscriptExpr>(E)) {
      if (const auto* IL = dyn_cast<clang::IntegerLiteral>(ASE->getIdx()))
        IdxAndMemberSequence.push_back(IdxOrMember(IL->getValue()));
      else
        IdxAndMemberSequence.push_back(IdxOrMember(llvm::APInt(2, -1, true)));
      E = ASE->getBase();
    } else if (const auto* ME = dyn_cast<clang::MemberExpr>(E)) {
      if (const auto* FD = dyn_cast<clang::FieldDecl>(ME->getMemberDecl()))
        IdxAndMemberSequence.push_back(IdxOrMember(FD));
      E = ME->getBase();
    } else if (isa<clang::DeclRefExpr>(E)) {
      innermostDRE = dyn_cast<clang::DeclRefExpr>(E);
      cond = false;
    } else
      return;
  }

  /// Overlay on all the VarData's recursively.
  if (const auto* VD = dyn_cast<clang::VarDecl>(innermostDRE->getDecl())) {
    overlay(getCurBranch()[VD], IdxAndMemberSequence,
            IdxAndMemberSequence.size());
  }
}

void TBRAnalyzer::addVar(const clang::VarDecl* VD) {
  /// If a declaration is passed a second time (meaning it is inside a loop),
  /// treat it as an assigment operation.
  /// FIXME: this marks the SourceLocation of DeclStmt which doesn't work for
  /// declarations with multiple VarDecls.
  auto& curBranch = getCurBranch();

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
  if (const ParmVarDecl* arrayParam = dyn_cast<ParmVarDecl>(VD))
    varType = arrayParam->getOriginalType();
  else
    varType = VD->getType();
  /// If varType represents auto or auto*, get the type of init.
  if (utils::IsAutoOrAutoPtrType(varType.getTypePtr()))
    varType = VD->getInit()->getType();

  /// FIXME: If the pointer points to an object we represent it with a
  /// OBJ_TYPE VarData*. This is done for '_d_this' pointer to be processed
  /// correctly in hessian mode. This should be removed once full support for
  /// pointers in analysis is introduced.
  if (const auto pointerType = dyn_cast<clang::PointerType>(varType)) {
    const auto* elemType = pointerType->getPointeeType().getTypePtrOrNull();
    if (elemType && elemType->isRecordType()) {
      curBranch[VD] = new VarData(QualType::getFromOpaquePtr(elemType));
      return;
    }
  }
  curBranch[VD] = new VarData(varType);
}

void TBRAnalyzer::markLocation(const clang::Expr* E) {
  VarData* data = getExprVarData(E);
  if (!data || findReq(data)) {
    /// FIXME: If any of the data's child nodes are required to store then data
    /// itself is stored. We might add an option to store separate fields.
    /// FIXME: Sometimes one location might correspond to multiple stores.
    /// For example, in ``(x*=y)=u`` x's location will first be marked as
    /// required to be stored (when passing *= operator) but then marked as not
    /// required to be stored (when passing = operator). Current method of
    /// marking locations does not allow to differentiate between these two.
    TBRLocs.insert(E->getBeginLoc());
  }
}

void TBRAnalyzer::setIsRequired(const clang::Expr* E, bool isReq) {
  if (!isReq ||
      (modeStack.back() == (Mode::markingMode | Mode::nonLinearMode))) {
    VarData* data = getExprVarData(E, /*addNonConstIdx=*/isReq);
    if (isReq || !nonConstIndexFound)
      setIsRequired(data, isReq);
    /// If an array element with a non-const element is set to required
    /// all the elements of that array should be set to required.
    if (isReq && nonConstIndexFound)
      overlay(E);
    nonConstIndexFound = false;
  }
}

void TBRAnalyzer::Analyze(const FunctionDecl* FD) {
  /// Build the CFG (control-flow graph) of FD.
  clang::CFG::BuildOptions Options;
  m_CFG = clang::CFG::buildCFG(FD, FD->getBody(), m_Context, Options);

  blockData.resize(m_CFG->size(), nullptr);
  blockPassCounter.resize(m_CFG->size(), 0);

  /// Set current block ID to the ID of entry the block.
  auto* entry = &m_CFG->getEntry();
  curBlockID = entry->getBlockID();
  blockData[curBlockID] = new VarsData();

  /// If we are analysing a method, add a VarData for 'this' pointer
  /// (it is represented with nullptr).
  if (isa<CXXMethodDecl>(FD)) {
    const Type* recordType =
        dyn_cast<CXXRecordDecl>(FD->getParent())->getTypeForDecl();
    getCurBranch()[nullptr] =
        new VarData(QualType::getFromOpaquePtr(recordType));
  }
  auto paramsRef = FD->parameters();
  for (std::size_t i = 0; i < FD->getNumParams(); ++i)
    addVar(paramsRef[i]);
  /// Add the entry block to the queue.
  CFGQueue.insert(curBlockID);

  /// Visit CFG blocks in the queue until it's empty.
  while (!CFGQueue.empty()) {
    auto IDIter = std::prev(CFGQueue.end());
    curBlockID = *IDIter;
    CFGQueue.erase(IDIter);

    auto* nextBlock = getCFGBlockByID(curBlockID);
    VisitCFGBlock(nextBlock);
  }
  //    for (int id = curBlockID; id >= 0; --id) {
  //        llvm::errs() << "\n-----BLOCK" << id << "-----\n\n";
  //        for (auto succ : getCFGBlockByID(id)->succs()) {
  //            if (succ)
  //            llvm::errs() << "successor: " << succ->getBlockID() << "\n";
  //        }
  //    }
}

void TBRAnalyzer::VisitCFGBlock(CFGBlock* block) {
  //    llvm::errs() << "\n-----BLOCK" << block->getBlockID() << "-----\n";
  /// Visiting loop blocks just once is not enough since the end of one
  /// loop iteration may have an effect on the next one. However, two
  /// iterations is always enough. Allow a third visit without going to
  /// successors to correctly analyse loop conditions.
  bool notLastPass = ++blockPassCounter[block->getBlockID()] <= 2;

  /// Visit all the statements inside the block.
  for (clang::CFGElement& Element : *block) {
    if (Element.getKind() == clang::CFGElement::Statement) {
      auto* Stmt = Element.castAs<clang::CFGStmt>().getStmt();
      Visit(Stmt);
    }
  }

  /// Traverse successor CFG blocks.
  for (auto succ : block->succs()) {
    /// Sometimes clang CFG does not create blocks for parts of code that
    /// are never executed (e.g. 'if (0) {...'). Add this check for safety.
    if (!succ)
      continue;
    auto*& varsData = blockData[succ->getBlockID()];

    /// Create VarsData for the succ branch if it hasn't been done previously.
    /// If the successor doesn't have a VarsData, assign it and attach the
    /// current block as previous.
    if (!varsData) {
      varsData = new VarsData();
      varsData->prev = blockData[block->getBlockID()];
    }

    /// If this is the third (last) pass of block, it means block represents
    /// a loop condition and the loop body has already been visited 2 times.
    /// This means we should not visit the loop body anymore.
    if (notLastPass) {
        /// Add the successor to the queue.
        CFGQueue.insert(succ->getBlockID());

        /// This part is necessary for loops. For other cases, this is not
        /// supposed to do anything.
        if (succ->getBlockID() < block->getBlockID()) {
            /// If there is another loop condition present inside a loop,
            /// We have to set it's loop pass counter to 0 (it might be 3
            /// from the previous outer loop pass).
            blockPassCounter[succ->getBlockID()] = 0;
            /// Remove VarsData left after the previous pass.
            varsData->clear();
        }
    }

    /// If the successor's previous block is not this one,
    /// perform a merge.
    if (varsData->prev != blockData[block->getBlockID()]) {
        merge(varsData, blockData[block->getBlockID()]);
    }
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

    auto branch = varsData1;
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
  /// This is not supposed to ever happen.
  return nullptr;
}

std::unique_ptr<TBRAnalyzer::VarsData>
TBRAnalyzer::collectDataFromPredecessors(VarsData* varsData,
                                         TBRAnalyzer::VarsData* limit) {
  auto result = std::unique_ptr<VarsData>(new VarsData(*varsData));

  if (varsData != limit) {
    /// Copy data from every predecessor.
    for (auto pred = varsData->prev; pred != limit; pred = pred->prev) {
      /// If a variable from 'pred' is not present
      /// in 'result', place it in there.
      for (auto pair : *pred)
        if (result->find(pair.first) == result->end())
          result->emplace(pair);
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
  for (auto& pair : *collectedMergeData) {
    VarData* found = nullptr;
    auto elemSearch = targetData->find(pair.first);
    if (elemSearch == targetData->end()) {
      auto* branch = targetData->prev;
      while (branch) {
        auto it = branch->find(pair.first);
        if (it != branch->end()) {
          found = copy(it->second);
          targetData->emplace(pair.first, found);
          break;
        }
        branch = branch->prev;
      }
    } else {
      found = elemSearch->second;
    }

    /// If the variable was found, perform a merge.
    /// Else, just copy it from collectedMergeData.
    if (found)
      merge(found, pair.second);
    else
      targetData->emplace(pair.first, copy(pair.second));
  }

  /// For every variable in collectedTargetData, search it inside
  /// collectedMergeData. If it's not found, that means it
  /// was not used anywhere between LCA and mergeData.
  /// To correctly merge, we have to take it from LCA's
  /// predecessors and merge it to targetData.
  for (auto& pair : *collectedTargetData) {
    auto elemSearch = collectedMergeData->find(pair.first);
    if (elemSearch == collectedMergeData->end()) {
      auto* branch = LCA;
      while (branch) {
        auto it = branch->find(pair.first);
        if (it != branch->end()) {
          merge(pair.second, it->second);
          break;
        }
        branch = branch->prev;
      }
    }
  }
}

void TBRAnalyzer::VisitCompoundStmt(const CompoundStmt* CS) {
  for (Stmt* S : CS->body())
    Visit(S);
}

void TBRAnalyzer::VisitDeclRefExpr(const DeclRefExpr* DRE) {
  if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    auto& curBranch = getCurBranch();
    if (curBranch.find(VD) == curBranch.end())
      addVar(VD);
  }

  if (const auto* E = dyn_cast<clang::Expr>(DRE))
    setIsRequired(E);
}

void TBRAnalyzer::VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE) {
  Visit(ICE->getSubExpr());
}

void TBRAnalyzer::VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE) {
  Visit(DE->getExpr());
}

void TBRAnalyzer::VisitParenExpr(const clang::ParenExpr* PE) {
  Visit(PE->getSubExpr());
}

void TBRAnalyzer::VisitReturnStmt(const clang::ReturnStmt* RS) {
  Visit(RS->getRetValue());
}

void TBRAnalyzer::VisitExprWithCleanups(const clang::ExprWithCleanups* EWC) {
  Visit(EWC->getSubExpr());
}

void TBRAnalyzer::VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* SCE) {
  Visit(SCE->getSubExpr());
}

void TBRAnalyzer::VisitDeclStmt(const DeclStmt* DS) {
  for (const auto* D : DS->decls()) {
    if (const auto* VD = dyn_cast<VarDecl>(D)) {
      addVar(VD);
      if (const clang::Expr* init = VD->getInit()) {
        setMode(Mode::markingMode);
        Visit(init);
        resetMode();
        auto& VDExpr = getCurBranch()[VD];
        /// if the declared variable is ref type attach its VarData* to the
        /// VarData* of the RHS variable.
        auto returnExprs = utils::GetInnermostReturnExpr(init);
        if (VDExpr->type == VarData::REF_TYPE && !returnExprs.empty())
          VDExpr->val.refData = returnExprs[0];
      }
    }
  }
}

void TBRAnalyzer::VisitConditionalOperator(
    const clang::ConditionalOperator* CO) {
  setMode(0);
  Visit(CO->getCond());
  resetMode();

  auto* elseBranch = blockData[curBlockID];
  auto* thenBranch = new VarsData();
  thenBranch->prev = elseBranch;

  blockData[curBlockID] = thenBranch;
  Visit(CO->getTrueExpr());

  blockData[curBlockID] = elseBranch;
  Visit(CO->getFalseExpr());

  merge(elseBranch, thenBranch);
  delete thenBranch;
}

void TBRAnalyzer::VisitBinaryOperator(const BinaryOperator* BinOp) {
  const auto opCode = BinOp->getOpcode();
  const auto* L = BinOp->getLHS();
  const auto* R = BinOp->getRHS();
  /// Addition is not able to create any differential influence by itself so
  /// markingMode should be left as it is. Similarly, addition does not affect
  /// linearity so nonLinearMode shouldn't be changed as well. The same applies
  /// to subtraction.
  if (opCode == BO_Add || opCode == BO_Sub) {
    Visit(L);
    Visit(R);
  } else if (opCode == BO_Mul) {
    /// Multiplication results in a linear expression if and only if one of the
    /// factors is constant.
    Expr::EvalResult dummy;
    bool nonLinear =
        !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, *m_Context) &&
        !clad_compat::Expr_EvaluateAsConstantExpr(L, dummy, *m_Context);
    if (nonLinear)
      startNonLinearMode();

    Visit(L);
    Visit(R);

    if (nonLinear)
      resetMode();
  } else if (opCode == BO_Div) {
    /// Division normally only results in a linear expression when the
    /// denominator is constant.
    Expr::EvalResult dummy;
    bool nonLinear =
        !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, *m_Context);
    if (nonLinear)
      startNonLinearMode();

    Visit(L);
    Visit(R);

    if (nonLinear)
      resetMode();
  } else if (BinOp->isAssignmentOp()) {
    if (opCode == BO_Assign || opCode == BO_AddAssign ||
        opCode == BO_SubAssign) {
      /// Since we only care about non-linear usages of variables, there is
      /// no difference between operators =, -=, += in terms of TBR analysis.
      Visit(L);

      startMarkingMode();
      Visit(R);
      resetMode();
    } else if (opCode == BO_MulAssign || opCode == BO_DivAssign) {
      /// *= (/=) normally only performs a linear operation if and only if
      /// the RHS is constant. If RHS is not constant, 'x *= y' ('x /= y')
      /// represents the same operation as 'x = x * y' ('x = x / y') and,
      /// therefore, LHS has to be visited in markingMode|nonLinearMode.
      Expr::EvalResult dummy;
      bool RisNotConst =
          !clad_compat::Expr_EvaluateAsConstantExpr(R, dummy, *m_Context);
      if (RisNotConst)
        setMode(Mode::markingMode | Mode::nonLinearMode);
      Visit(L);
      if (RisNotConst)
        resetMode();

      setMode(Mode::markingMode | Mode::nonLinearMode);
      Visit(R);
      resetMode();
    }
    const auto returnExprs = utils::GetInnermostReturnExpr(L);
    for (const auto* innerExpr : returnExprs) {
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
    Visit(L);
    resetMode();

    Visit(R);
  }
  // else {
  // FIXME: add logic/bitwise/comparison operators
  // }
}

void TBRAnalyzer::VisitUnaryOperator(const clang::UnaryOperator* UnOp) {
  const auto opCode = UnOp->getOpcode();
  const Expr* E = UnOp->getSubExpr();
  Visit(E);
  if (opCode == UO_PostInc || opCode == UO_PostDec || opCode == UO_PreInc ||
      opCode == UO_PreDec) {
    // FIXME: this doesn't support all the possible references
    /// Mark corresponding SourceLocation as required/not required to be
    /// stored for all expressions that could be used in this operation.
    const auto innerExprs = utils::GetInnermostReturnExpr(E);
    for (const auto* innerExpr : innerExprs) {
      /// Mark corresponding SourceLocation as required/not required to be
      /// stored for all expressions that could be changed.
      markLocation(innerExpr);
    }
  }
}

void TBRAnalyzer::VisitCallExpr(const clang::CallExpr* CE) {
  /// FIXME: Currently TBR analysis just stops here and assumes that all the
  /// variables passed by value/reference are used/used and changed. Analysis
  /// could proceed to the function to analyse data flow inside it.
  auto* FD = CE->getDirectCallee();
  bool noHiddenParam = (CE->getNumArgs() == FD->getNumParams());
  setMode(Mode::markingMode | Mode::nonLinearMode);
  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    const clang::Expr* arg = CE->getArg(i);
    bool passByRef = false;
    if (noHiddenParam)
      passByRef = FD->getParamDecl(i)->getType()->isReferenceType();
    else if (i)
      passByRef = FD->getParamDecl(i - 1)->getType()->isReferenceType();
    setMode(Mode::markingMode | Mode::nonLinearMode);
    Visit(arg);
    resetMode();
    const auto* B = arg->IgnoreParenImpCasts();
    // FIXME: this supports only DeclRefExpr
    const auto innerExpr = utils::GetInnermostReturnExpr(arg);
    if (passByRef) {
      /// Mark SourceLocation as required to store for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        TBRLocs.insert(arg->getBeginLoc());
        setIsRequired(arg, /*isReq=*/false);
      }
    }
  }
  resetMode();
}

void TBRAnalyzer::VisitCXXConstructExpr(const clang::CXXConstructExpr* CE) {
  /// FIXME: Currently TBR analysis just stops here and assumes that all the
  /// variables passed by value/reference are used/used and changed. Analysis
  /// could proceed to the constructor to analyse data flow inside it.
  /// FIXME: add support for default values
  auto* FD = CE->getConstructor();
  setMode(Mode::markingMode | Mode::nonLinearMode);
  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    const auto* arg = CE->getArg(i);
    bool passByRef = FD->getParamDecl(i)->getType()->isReferenceType();
    setMode(Mode::markingMode | Mode::nonLinearMode);
    Visit(arg);
    resetMode();
    const auto* B = arg->IgnoreParenImpCasts();
    // FIXME: this supports only DeclRefExpr
    if (passByRef) {
      /// Mark SourceLocation as required for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        TBRLocs.insert(arg->getBeginLoc());
        setIsRequired(arg, /*isReq=*/false);
      }
    }
  }
  resetMode();
}

void TBRAnalyzer::VisitMemberExpr(const clang::MemberExpr* ME) {
  setIsRequired(dyn_cast<clang::Expr>(ME));
}

void TBRAnalyzer::VisitArraySubscriptExpr(
    const clang::ArraySubscriptExpr* ASE) {
  setMode(0);
  Visit(ASE->getBase());
  resetMode();
  setIsRequired(dyn_cast<clang::Expr>(ASE));
  setMode(Mode::markingMode | Mode::nonLinearMode);
  Visit(ASE->getIdx());
  resetMode();
}

void TBRAnalyzer::VisitInitListExpr(const clang::InitListExpr* ILE) {
  setMode(Mode::markingMode);
  for (auto* init : ILE->inits()) {
      Visit(init);
  }
  resetMode();
}

} // end namespace clad
