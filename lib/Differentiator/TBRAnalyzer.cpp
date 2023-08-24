#include "clad/Differentiator/TBRAnalyzer.h"

using namespace clang;

namespace clad {

void TBRAnalyzer::VarData::setIsRequired(bool isReq) {
  if (type == FUND_TYPE) {
    val.fundData = isReq;
  } else if (type == OBJ_TYPE) {
    for (auto& pair : *val.objData) {
      pair.second->setIsRequired(isReq);
    }
  } else if (type == ARR_TYPE) {
    for (auto& pair : *val.arrData) {
      pair.second->setIsRequired(isReq);
    }
  } else if (type == REF_TYPE && val.refData) {
    val.refData->setIsRequired(isReq);
  }
}

void TBRAnalyzer::VarData::merge(VarData* mergeData) {
  if (this->type == FUND_TYPE) {
    this->val.fundData = this->val.fundData || mergeData->val.fundData;
  } else if (this->type == OBJ_TYPE) {
    for (auto& pair : *this->val.objData) {
      pair.second->merge((*mergeData->val.objData)[pair.first]);
    }
  } else if (this->type == ARR_TYPE) {
    /// FIXME: Currently non-constant indices are not supported in merging.
    for (auto& pair : *this->val.arrData) {
      auto it = mergeData->val.arrData->find(pair.first);
      if (it != mergeData->val.arrData->end()) {
        pair.second->merge(it->second);
      }
    }
    for (auto& pair : *mergeData->val.arrData) {
      auto it = this->val.arrData->find(pair.first);
      if (it == mergeData->val.arrData->end()) {
        std::unordered_map<VarData*, VarData*> refVars;
        (*this->val.arrData)[pair.first] = pair.second->copy(refVars);
      }
    }
  } else if (this->type == REF_TYPE && this->val.refData) {
    this->val.refData->merge(mergeData->val.refData);
  }
}

TBRAnalyzer::VarData* TBRAnalyzer::VarData::copy() {
  std::unordered_map<VarData*, VarData*> refVars;
  VarData* res = copy(refVars);
  res->restoreRefs(refVars);
  return res;
}

TBRAnalyzer::VarData*
TBRAnalyzer::VarData::copy(std::unordered_map<VarData*, VarData*>& refVars) {
  auto* res = new VarData();
  /// The child node of a reference node should be copied only once. Hence,
  /// we use refVars to match original referenced nodes to corresponding copies.
  if (isReferenced) {
    refVars[this] = res;
  }
  res->type = this->type;
  if (this->type == FUND_TYPE) {
    res->val.fundData = this->val.fundData;
  } else if (this->type == OBJ_TYPE) {
    res->val.objData = new ObjMap();
    for (auto& pair : *this->val.objData)
      (*res->val.objData)[pair.first] = pair.second->copy(refVars);
  } else if (this->type == ARR_TYPE) {
    res->val.arrData = new ArrMap();
    for (auto& pair : *this->val.arrData) {
      (*res->val.arrData)[pair.first] = pair.second->copy(refVars);
    }
  } else if (this->type == REF_TYPE && this->val.refData) {
    res->val.refData = this->val.refData;
  }
  return res;
}

void TBRAnalyzer::VarData::restoreRefs(
    std::unordered_map<VarData*, VarData*>& refVars) {
  if (this->type == OBJ_TYPE) {
    for (auto& pair : *val.objData)
      pair.second->restoreRefs(refVars);
  } else if (this->type == ARR_TYPE) {
    for (auto& pair : *this->val.arrData) {
      pair.second->restoreRefs(refVars);
    }
  } else if (this->type == REF_TYPE && this->val.refData) {
    this->val.refData = refVars[this->val.refData];
  }
}

bool TBRAnalyzer::VarData::findReq() const {
  if (type == FUND_TYPE) {
    return val.fundData;
  }
  if (type == OBJ_TYPE) {
    for (auto& pair : *val.objData) {
      if (pair.second->findReq()) {
        return true;
      }
    }
  } else if (type == ARR_TYPE) {
    for (auto& pair : *val.arrData) {
      if (pair.second->findReq()) {
        return true;
      }
    }
  } else if (type == REF_TYPE && val.refData) {
    if (val.refData->findReq()) {
      return true;
    }
  }
  return false;
}

void TBRAnalyzer::VarData::overlay(
    llvm::SmallVector<IdxOrMember, 2>& IdxAndMemberSequence, size_t i) {
  if (i == 0) {
    setIsRequired();
    return;
  }
  --i;
  IdxOrMember& curIdxOrMember = IdxAndMemberSequence[i];
  if (curIdxOrMember.type == IdxOrMember::IdxOrMemberType::FIELD) {
    (*val.objData)[curIdxOrMember.val.field]->overlay(IdxAndMemberSequence, i);
  } else if (curIdxOrMember.type == IdxOrMember::IdxOrMemberType::INDEX) {
    auto idx = curIdxOrMember.val.index;
    if (idx == llvm::APInt(2, -1, true)) {
      for (auto& pair : *val.arrData) {
        pair.second->overlay(IdxAndMemberSequence, i);
      }
    } else {
      (*val.arrData)[idx]->overlay(IdxAndMemberSequence, i);
    }
  }
}

TBRAnalyzer::VarData* TBRAnalyzer::getMemberVarData(const clang::MemberExpr* ME,
                                                    bool addNonConstIdx) {
  if (const auto* FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
    const auto* base = ME->getBase();
    VarData* baseData = getExprVarData(base);
    /// If the VarData is ref type just go to the VarData being referenced.
    if (baseData && baseData->type == VarData::VarDataType::REF_TYPE) {
      baseData = baseData->val.refData;
    }

    if (!baseData)
      return nullptr;
    /// FUND_TYPE might be set by default earlier.
    if (baseData->type == VarData::VarDataType::FUND_TYPE) {
      baseData->type = VarData::VarDataType::OBJ_TYPE;
      baseData->val.objData = new ObjMap();
    }
    /// if non-const index was found and it is not supposed to be added just
    /// return the current VarData*.
    if (nonConstIndexFound && !addNonConstIdx)
      return baseData;

    auto& baseObjData = baseData->val.objData;
    /// Add the current field if it was not added previously
    if (baseObjData->find(FD) == baseObjData->end()) {
      (*baseObjData)[FD] = new VarData();
      auto* FDData = (*baseObjData)[FD];
      FDData->type = VarData::VarDataType::UNDEFINED;
      return FDData;
    }

    return (*baseObjData)[FD];
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
  /// If the VarData is ref type just go to the VarData being referenced.
  if (baseData && baseData->type == VarData::VarDataType::REF_TYPE) {
    baseData = baseData->val.refData;
  }

  if (!baseData)
    return nullptr;
  /// FUND_TYPE might be set by default earlier.
  if (baseData->type == VarData::VarDataType::FUND_TYPE) {
    baseData->type = VarData::VarDataType::ARR_TYPE;
    baseData->val.arrData = new ArrMap();
  }

  /// if non-const index was found and it is not supposed to be added just
  /// return the current VarData*.
  if (nonConstIndexFound && !addNonConstIdx)
    return baseData;

  auto& baseArrData = baseData->val.arrData;
  auto itEnd = baseArrData->end();

  /// Add the current index if it was not added previously
  if (baseArrData->find(idx) == itEnd) {
    (*baseArrData)[idx] = new VarData();
    auto* idxData = (*baseArrData)[idx];
    /// Since -1 represents non-const indices, whenever we add a new index we
    /// have to copy the VarData of -1's element (if an element with undefined
    /// index was used this might be our current element).
    auto it = baseArrData->find(llvm::APInt(2, -1, true));
    if (it != itEnd) {
      std::unordered_map<VarData*, VarData*> dummy;
      idxData = it->second->copy(dummy);
    } else {
      idxData->type = VarData::VarDataType::UNDEFINED;
    }
    return idxData;
  }

  return (*baseArrData)[idx];
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
    /// The index i is shifted since otherwise the last value would be i=-1
    /// and size_t can only take positive values.
    for (size_t i = reqStack.size(); i > 0; --i) {
      auto& branch = reqStack[i - 1].back();
      const auto it = branch.find(VD);
      if (it != branch.end()) {
        EData = it->second;
        break;
      }
    }
  }
  if (const auto* ME = dyn_cast<clang::MemberExpr>(E)) {
    EData = getMemberVarData(ME, addNonConstIdx);
  }
  if (const auto* ASE = dyn_cast<clang::ArraySubscriptExpr>(E)) {
    EData = getArrSubVarData(ASE, addNonConstIdx);
  }

  /// If the type of this VarData was not defined previously set it to
  /// FUND_TYPE.
  /// FIXME: this assumes that we only assign fundamental values and not
  /// objects or pointers.
  if (EData && EData->type == VarData::VarDataType::UNDEFINED) {
    EData->type = VarData::VarDataType::FUND_TYPE;
    EData->val.fundData = false;
  }
  return EData;
}

void TBRAnalyzer::addField(ObjMap* objData, const FieldDecl* FD) {
  const auto varType = FD->getType();
  (*objData)[FD] = new VarData();
  VarData* data = (*objData)[FD];

  if (varType->isReferenceType()) {
    data->type = VarData::VarDataType::REF_TYPE;
    data->val.refData = nullptr;
  } else if (utils::isArrayOrPointerType(varType)) {
    data->type = VarData::VarDataType::ARR_TYPE;
    data->val.arrData = new ArrMap();
  } else if (varType->isBuiltinType()) {
    data->type = VarData::VarDataType::FUND_TYPE;
    data->val.fundData = false;
  } else if (varType->isRecordType()) {
    data->type = VarData::VarDataType::OBJ_TYPE;
    const auto* recordDecl = varType->getAs<RecordType>()->getDecl();
    auto& newObjData = data->val.objData;
    for (const auto* field : recordDecl->fields()) {
      addField(newObjData, field);
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
      if (const auto* IL = dyn_cast<clang::IntegerLiteral>(ASE->getIdx())) {
        IdxAndMemberSequence.push_back(IdxOrMember(IL->getValue()));
      } else {
        IdxAndMemberSequence.push_back(IdxOrMember(llvm::APInt(2, -1, true)));
      }
      E = ASE->getBase();
    } else if (const auto* ME = dyn_cast<clang::MemberExpr>(E)) {
      if (const auto* FD = dyn_cast<clang::FieldDecl>(ME->getMemberDecl())) {
        IdxAndMemberSequence.push_back(IdxOrMember(FD));
      }
      E = ME->getBase();
    } else if ((innermostDRE = dyn_cast<clang::DeclRefExpr>(E))) {
      cond = false;
    } else
      return;
  }

  /// Overlay on all the VarData's recursively.
  if (const auto* VD = dyn_cast<clang::VarDecl>(innermostDRE->getDecl())) {
    getCurBranch()[VD]->overlay(IdxAndMemberSequence,
                                IdxAndMemberSequence.size());
  }
}

void TBRAnalyzer::addVar(const clang::VarDecl* VD) {
  /// If a declaration is passed a second time (meaning it is inside a loop),
  /// treat it as an assigment operation.
  /// FIXME: this marks the SourceLocation of DeclStmt which doesn't work for
  /// declarations with multiple VarDecls.
  size_t len = reqStack.size();
  auto& curBranch = reqStack[len - 1].back();
  if (curBranch.find(VD) != curBranch.end()) {
    auto& VDData = curBranch[VD];
    if (VDData->type == VarData::VarDataType::FUND_TYPE) {
      TBRLocs[VD->getBeginLoc()] =
          (deleteCurBranch ? false : VDData->findReq());
    }
  }
  /// The index here is shifted by one since otherwise the loop would end with
  /// i=-1 and size_t is positive only.
  for (size_t i = len - 1; i > 0; --i) {
    auto& branch = reqStack[i - 1].back();
    if (branch.find(VD) != branch.end()) {
      curBranch[VD] = branch[VD]->copy();
      break;
    }
  }

  if (!localVarsStack.empty()) {
    localVarsStack.back().push_back(VD);
  }

  const auto varType = VD->getType();
  curBranch[VD] = new VarData();
  VarData* data = curBranch[VD];

  if (varType->isReferenceType()) {
    data->type = VarData::VarDataType::REF_TYPE;
    data->val.refData = nullptr;
  } else if (utils::isArrayOrPointerType(varType)) {
    if (const auto pointerType = llvm::dyn_cast<clang::PointerType>(varType)) {
      /// FIXME: If the pointer points to an object we represent it with a
      /// OBJ_TYPE VarData*.
      const auto pointeeType = pointerType->getPointeeType().getTypePtrOrNull();
      if (pointeeType && pointeeType->isRecordType()) {
        data->type = VarData::VarDataType::OBJ_TYPE;
        const auto* recordDecl = pointeeType->getAs<RecordType>()->getDecl();
        auto& objData = data->val.objData;
        objData = new ObjMap();
        for (const auto* field : recordDecl->fields()) {
          addField(objData, field);
        }
        return;
      }
    }
    data->type = VarData::VarDataType::ARR_TYPE;
    data->val.arrData = new ArrMap();
  } else if (varType->isBuiltinType()) {
    data->type = VarData::VarDataType::FUND_TYPE;
    data->val.fundData = false;
  } else if (varType->isRecordType()) {
    data->type = VarData::VarDataType::OBJ_TYPE;
    const auto* recordDecl = varType->getAs<RecordType>()->getDecl();
    auto& objData = data->val.objData;
    objData = new ObjMap();
    for (const auto* field : recordDecl->fields()) {
      addField(objData, field);
    }
  }
}

void TBRAnalyzer::markLocation(const clang::Expr* E) {
  VarData* data = getExprVarData(E);
  if (data) {
    /// FIXME: If any of the data's child nodes are required to store then data
    /// itselt is stored. We might add an option to store separate fields.
    bool& ToBeRec = TBRLocs[E->getBeginLoc()];
    /// FIXME: Sometimes one location might correspong to multiple stores.
    /// For example, in ``(x*=y)=u`` x's location will first be marked as
    /// required to be stored (when passing *= operator) but then marked as not
    /// required to be stored (when passing = operator). Current method of
    /// marking locations does not allow to differentiate between these two.
    ToBeRec = (deleteCurBranch ? false : ToBeRec || data->findReq());
  } else
    /// If the current branch is going to be deleted then there is not point in
    /// storing anything in it.
    TBRLocs[E->getBeginLoc()] = !deleteCurBranch;
}

void TBRAnalyzer::mergeLayer() {
  size_t len = reqStack.size();
  auto& removedLayer = reqStack[len - 1];
  auto& curBranch = reqStack[len - 2].back();

  for (auto& removedBranch : removedLayer) {
    for (auto& pair : curBranch) {
      auto it = removedBranch.find(pair.first);
      if (it != removedBranch.end()) {
        pair.second->merge(it->second);
      }
    }
    for (auto& pair : removedBranch) {
      auto it = curBranch.find(pair.first);
      if (it == curBranch.end()) {
        delete curBranch[pair.first];
        curBranch[pair.first] = pair.second;
      } else {
        delete pair.second;
      }
    }
  }
  reqStack.pop_back();
}

void TBRAnalyzer::mergeLayerOnTop() {
  size_t len = reqStack.size();
  auto& removedLayer = reqStack[len - 1];

  if (removedLayer.empty()) {
    reqStack.pop_back();
    return;
  }

  auto& curBranch = reqStack[len - 2].back();

  /// First, we merge every branch on the layer with the first one there.
  auto branchIter = removedLayer.begin();
  auto branchIterEnd = removedLayer.end();
  auto& firstBranch = *branchIter;
  while ((++branchIter) != branchIterEnd) {
    for (auto& pair : firstBranch) {
      auto elemIter = branchIter->find(pair.first);
      if (elemIter != branchIter->end()) {
        pair.second->merge(elemIter->second);
      }
    }
    for (auto& pair : *branchIter) {
      auto elemIter = firstBranch.find(pair.first);
      if (elemIter == firstBranch.end()) {
        delete firstBranch[pair.first];
        firstBranch[pair.first] = pair.second;
      } else {
        delete pair.second;
      }
    }
  }

  /// Second, we place it on top of the branch on the previous layer's last
  /// branch.
  for (auto& pair : firstBranch) {
    delete curBranch[pair.first];
    curBranch[pair.first] = pair.second;
  }

  reqStack.pop_back();
}

void TBRAnalyzer::mergeBranchTo(size_t sourceBranchNum,
                                VarsData& targetBranch) {
  for (auto& pair : targetBranch) {
    /// Index i is shifted by one since otherwise its last value could be -1
    /// and size_t is only positive.
    for (size_t i = sourceBranchNum + 1; i > 0; --i) {
      auto& sourceBranch = reqStack[i - 1].back();
      auto it = sourceBranch.find(pair.first);
      if (it != sourceBranch.end()) {
        pair.second->merge(it->second);
        break;
      }
    }
  }
}

void TBRAnalyzer::setIsRequired(const clang::Expr* E, bool isReq) {
  if (!isReq ||
      (modeStack.back() == (Mode::markingMode | Mode::nonLinearMode))) {
    VarData* data = getExprVarData(E, /*addNonConstIdx=*/isReq);
    if (isReq || !nonConstIndexFound) {
      data->setIsRequired(isReq);
    }
    /// If an array element with a non-const element is set to required
    /// all the elements of that array should be set to required.
    if (isReq && nonConstIndexFound) {
      overlay(E);
    }
    nonConstIndexFound = false;
  }
}

void TBRAnalyzer::Analyze(const FunctionDecl* FD) {
  /// If we are analysing a method add a VarData for 'this' pointer (it is
  /// represented with nullptr).

  if (isa<CXXMethodDecl>(FD)) {
    auto& thisData = getCurBranch()[nullptr];
    thisData = new VarData();
    thisData->type = VarData::VarDataType::OBJ_TYPE;
    auto recordDecl = dyn_cast<CXXRecordDecl>(FD->getParent());
    auto& objData = thisData->val.objData;
    objData = new ObjMap();
    for (const auto* field : recordDecl->fields()) {
      addField(objData, field);
    }
  }
  auto paramsRef = FD->parameters();

  for (std::size_t i = 0; i < FD->getNumParams(); ++i)
    addVar(paramsRef[i]);
  Visit(FD->getBody());
}

void TBRAnalyzer::VisitCompoundStmt(const CompoundStmt* CS) {
  for (Stmt* S : CS->body())
    Visit(S);
}

void TBRAnalyzer::VisitDeclRefExpr(const DeclRefExpr* DRE) {
  if (const auto* VD = dyn_cast<VarDecl>(DRE->getDecl())) {
    auto& curBranch = getCurBranch();
    // FIXME: this is only necessary to ensure global variables are added.
    // It doesn't make any sense to first add variables when visiting DeclStmt
    // and then checking if they were added while visiting DeclRefExpr.
    if (curBranch.find(VD) == curBranch.end())
      addVar(VD);
  }

  if (const auto* E = dyn_cast<clang::Expr>(DRE)) {
    setIsRequired(E);
  }
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
  deleteCurBranch = true;
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
        if (VDExpr->type == VarData::VarDataType::REF_TYPE) {
          auto* RHSExpr =
              getExprVarData(utils::GetInnermostReturnExpr(init)[0]);
          VDExpr->val.refData = RHSExpr;
          RHSExpr->isReferenced = true;
        }
      }
    }
  }
}

void TBRAnalyzer::VisitConditionalOperator(
    const clang::ConditionalOperator* CO) {
  setMode(0);
  Visit(CO->getCond());
  resetMode();

  addLayer();
  addBranch();
  Visit(CO->getTrueExpr());
  addBranch();
  Visit(CO->getFalseExpr());
  mergeLayerOnTop();
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
    const auto return_exprs = utils::GetInnermostReturnExpr(L);
    for (const auto* innerExpr : return_exprs) {
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
      /// Set them to not required to store because the values were changed.
      /// (if some value was not changed, this could only happen if it was
      /// already not required to store).
      setIsRequired(innerExpr, /*isReq=*/false);
    }
  }
}

void TBRAnalyzer::VisitIfStmt(const clang::IfStmt* If) {
  const auto* cond = If->getCond();
  const auto* condVarDecl = If->getConditionVariable();
  const auto* condInit = If->getInit();

  /// We have to separated analyse then-block and else-block and then merge
  /// them together. First, we make a copy of the current branch and analyse
  /// then-block on it. Then swap last two branches and analyse the else-block
  /// on the last branch. Finally, we merge them together. This diagram explains
  /// the transformations performed on the reqStack:
  /// ... - <original state>
  /// ... - <original state> - <original state (copy)>
  /// ... - <original state> - <state after then-block>
  /// ... - <state after then-block> - <original state>
  /// ... - <state after then-block> - <state after else-block>
  /// ... - <merged state after then-block/else-block>

  const auto* thenBranch = If->getThen();
  const auto* elseBranch = If->getElse();

  localVarsStack.emplace_back();
  addLayer();

  if (thenBranch) {
    addBranch();
    Visit(cond);
    if (condVarDecl)
      addVar(condVarDecl);
    if (condInit) {
      setMode(Mode::markingMode);
      Visit(condInit);
      resetMode();
    }
    Visit(thenBranch);
    if (deleteCurBranch) {
      /// This section is performed if this branch had break/continue/return
      /// and, therefore, shouldn't be merged.
      deleteBranch();
      deleteCurBranch = false;
    }
  }

  if (elseBranch) {
    addBranch();
    Visit(cond);
    if (condVarDecl)
      addVar(condVarDecl);
    if (condInit) {
      setMode(Mode::markingMode);
      Visit(condInit);
      resetMode();
    }
    Visit(elseBranch);
    if (deleteCurBranch) {
      /// This section is performed if this branch had break/continue/return
      /// and, therefore, shouldn't be merged.
      deleteBranch();
      deleteCurBranch = false;
    }
  }

  mergeLayerOnTop();
  removeLocalVars();
  localVarsStack.pop_back();
}

void TBRAnalyzer::VisitWhileStmt(const clang::WhileStmt* WS) {
  const auto* body = WS->getBody();
  const auto* cond = WS->getCond();
  size_t backupILB = innermostLoopLayer;
  bool backupFLP = firstLoopPass;
  bool backupDCB = deleteCurBranch;
  /// Let's assume we have a section of code structured like this
  /// (A, B, C represent blocks):
  /// ```
  /// A
  /// while (cond) B
  /// C
  /// ```
  /// Depending on cond, this could give us 3 types of scenarios: 'AC', 'ABC',
  /// 'AB...BC'. We must notice two things: 1) C comes either after B or A,
  /// 2) B comes either after A or B itself. So first, we have to merge original
  /// state with after-first-iteration state and analyse B a second time on top
  /// to get the state that represents arbitrary non-zero number of iterations.
  /// Finally, we have to merge it with the original state once again to account
  /// for the fact that the loop block may not be executed at all.
  /// This diagram explains the transformations performed on the reqStack:
  /// ... - <original state>
  /// ... - <original state> - <original state (copy)>
  /// ... - <original state> - <original state (copy)> - <original state (copy)>
  /// ... - <original state> - <original state (copy)> - <state after loop
  /// block>
  /// ... - <original state> - <merged original and after-loop-block states>
  /// ... - <original state> - <merged original and after-loop-block states +
  /// second loop pass>
  /// ... - <the state representing all possible scenarios merged>

  Visit(cond);
  addLayer();
  addBranch();
  addBranch();

  addLayer();
  addBranch();
  addBranch();
  /// First pass
  innermostLoopLayer = reqStack.size() - 1;
  firstLoopPass = true;
  if (body)
    Visit(body);
  if (deleteCurBranch) {
    deleteBranch();
    deleteCurBranch = backupDCB;
  } else {
    Visit(cond);
  }
  --innermostLoopLayer;
  mergeLayer();
  mergeBranchTo(innermostLoopLayer - 1, reqStack[innermostLoopLayer].back());

  /// Second pass
  firstLoopPass = false;
  if (body)
    Visit(body);
  if (deleteCurBranch)
    deleteBranch();
  else {
    Visit(cond);
  }
  mergeLayer();

  innermostLoopLayer = backupILB;
  firstLoopPass = backupFLP;
  deleteCurBranch = backupDCB;
}

void TBRAnalyzer::VisitForStmt(const clang::ForStmt* FS) {
  const auto* body = FS->getBody();
  const auto* condVar = FS->getConditionVariable();
  auto* init = FS->getInit();
  const auto* cond = FS->getCond();
  const auto* incr = FS->getInc();
  size_t backupILB = innermostLoopLayer;
  bool backupFLP = firstLoopPass;
  bool backupDCB = deleteCurBranch;
  /// The logic here is virtually the same as with while-loop. Take a look at
  /// TBRAnalyzer::VisitWhileStmt for more details.
  if (init) {
    setMode(Mode::markingMode);
    Visit(init);
    resetMode();
  }
  if (cond)
    Visit(cond);
  addLayer();
  addBranch();
  addBranch();
  if (condVar)
    addVar(condVar);
  addLayer();
  addBranch();
  addBranch();
  /// First pass
  innermostLoopLayer = reqStack.size() - 1;
  firstLoopPass = true;
  if (body)
    Visit(body);
  if (deleteCurBranch) {
    deleteBranch();
    deleteCurBranch = backupDCB;
  } else {
    if (incr)
      Visit(incr);
    if (cond)
      Visit(cond);
  }
  --innermostLoopLayer;
  mergeLayer();
  mergeBranchTo(innermostLoopLayer - 1, reqStack[innermostLoopLayer].back());

  /// Second pass
  firstLoopPass = false;
  if (body)
    Visit(body);
  if (incr)
    Visit(incr);
  if (deleteCurBranch)
    deleteBranch();
  else {
    if (cond)
      Visit(cond);
  }
  mergeLayer();

  innermostLoopLayer = backupILB;
  firstLoopPass = backupFLP;
  deleteCurBranch = backupDCB;
}

void TBRAnalyzer::VisitDoStmt(const clang::DoStmt* DS) {
  const auto* body = DS->getBody();
  const auto* cond = DS->getCond();
  size_t backupILB = innermostLoopLayer;
  bool backupFLP = firstLoopPass;
  bool backupDCB = deleteCurBranch;

  /// The logic used here is virtually the same as with while-loop. Take a look
  /// at TBRAnalyzer::VisitWhileStmt for more details.
  /// FIXME: do-while-block is performed at least once and so we don't have to
  /// account for the possibility of it not being performed at all. However,
  /// having two loop branches is necessary for handling continue statements
  /// so we can't just remove one of them.

  addLayer();
  addBranch();
  addBranch();
  addLayer();
  addBranch();
  addBranch();
  /// First pass
  innermostLoopLayer = reqStack.size() - 2;
  firstLoopPass = true;
  if (body)
    Visit(body);
  if (deleteCurBranch) {
    reqStack.pop_back();
    deleteCurBranch = backupDCB;
  } else {
    Visit(cond);
    mergeLayer();
  }

  /// Second pass
  --innermostLoopLayer;
  firstLoopPass = false;
  if (body)
    Visit(body);
  Visit(cond);
  if (deleteCurBranch) {
    reqStack.pop_back();
    mergeLayer();
  }

  innermostLoopLayer = backupILB;
  firstLoopPass = backupFLP;
  deleteCurBranch = backupDCB;

  addLayer();
  addBranch();
  addBranch();

  addLayer();
  addBranch();
  addBranch();
  /// First pass
  innermostLoopLayer = reqStack.size() - 1;
  firstLoopPass = true;
  if (body)
    Visit(body);
  if (deleteCurBranch) {
    deleteBranch();
    deleteCurBranch = backupDCB;
  } else {
    Visit(cond);
  }
  --innermostLoopLayer;
  mergeLayer();
  mergeBranchTo(innermostLoopLayer - 1, reqStack[innermostLoopLayer].back());

  /// Second pass
  firstLoopPass = false;
  if (body)
    Visit(body);
  if (deleteCurBranch)
    deleteBranch();
  else {
    Visit(cond);
  }
  mergeLayer();

  innermostLoopLayer = backupILB;
  firstLoopPass = backupFLP;
  deleteCurBranch = backupDCB;
}

void TBRAnalyzer::VisitContinueStmt(const clang::ContinueStmt* CS) {
  /// If this is the first loop pass, the reqStack will look like this:
  /// ... - <original state> - <original state (copy)> - <original state (copy)>
  /// And so continue might be the end of this loop as well as the the end of
  /// the first iteration. So we have to merge the current branch into first
  /// two branches on the diagram.
  /// If this is the second loop pass, the reqStack will look like this:
  /// ... - <original state> - <original state (copy)>
  /// And so this continue could be the end of this loop. So we have to merge
  /// the current branch into the first branch on the diagram.
  /// FIXME: If this is the second pass, this continue statement could still be
  /// followed by another iteration. We have to either add an additional branch
  /// or find a better solution. (However, this bug will matter only in really
  /// rare cases)

  auto& targetLayer1 = reqStack[innermostLoopLayer];
  auto& targetBranch1 = targetLayer1[targetLayer1.size() - 2];
  size_t sourceBranchNum = reqStack.size() - 1;
  mergeBranchTo(sourceBranchNum, targetBranch1);
  /// After the continue statement, this branch cannot be followed by any other
  /// code so we can delete it.
  if (firstLoopPass) {
    auto& targetLayer2 = reqStack[innermostLoopLayer - 1];
    auto& targetBranch2 = targetLayer2[targetLayer2.size() - 2];
    mergeBranchTo(sourceBranchNum, targetBranch2);
  }
  deleteCurBranch = true;
}

void TBRAnalyzer::VisitBreakStmt(const clang::BreakStmt* BS) {
  /// If this is the second loop pass, the reqStack will look like this:
  /// ... - <original state> - <original state (copy)>
  /// And so this break could be the end of this loop. So we have to merge
  /// the current branch into the first branch on the diagram.
  if (!firstLoopPass) {
    auto& targetLayer = reqStack[innermostLoopLayer];
    auto& targetBranch = targetLayer[targetLayer.size() - 2];
    mergeBranchTo(reqStack.size() - 1, targetBranch);
  }
  /// After the break statement, this branch cannot be followed by any other
  /// code so we can delete it.
  deleteCurBranch = true;
}

void TBRAnalyzer::VisitCallExpr(const clang::CallExpr* CE) {
  /// FIXME: Currently TBR analysis just stops here and assumes that all the
  /// variables passed by value/reference are used/used and changed. Analysis
  /// could proceed to the function to analyse data flow inside it.
  auto* FD = CE->getDirectCallee();
  setMode(Mode::markingMode | Mode::nonLinearMode);
  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    const clang::Expr* arg = CE->getArg(i);
    bool passByRef = FD->getParamDecl(i)->getType()->isReferenceType();
    setMode(Mode::markingMode | Mode::nonLinearMode);
    Visit(arg);
    resetMode();
    const auto* B = arg->IgnoreParenImpCasts();
    // FIXME: this supports only DeclRefExpr
    const auto innerExpr = utils::GetInnermostReturnExpr(arg);
    if (passByRef) {
      /// Mark SourceLocation as required for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        TBRLocs[arg->getBeginLoc()] = true;
        setIsRequired(arg, /*isReq=*/false);
      }
    } else {
      /// Mark SourceLocation as not required for non-ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B))
        TBRLocs[arg->getBeginLoc()] = false;
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
        TBRLocs[arg->getBeginLoc()] = true;
        setIsRequired(arg, /*isReq=*/false);
      }
    } else {
      /// Mark SourceLocation as not required for non-ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B))
        TBRLocs[arg->getBeginLoc()] = false;
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
  setMode(0);
  for (auto* init : ILE->inits()) {
    Visit(init);
  }
  resetMode();
}

void TBRAnalyzer::removeLocalVars() {
  auto& curBranch = getCurBranch();
  for (const auto* VD : localVarsStack.back())
    curBranch.erase(VD);
}

} // end namespace clad
