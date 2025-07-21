#include "TBRAnalyzer.h"

#include <algorithm>
#include <cassert>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/LLVM.h"

#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffPlanner.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#undef DEBUG_TYPE
#define DEBUG_TYPE "clad-tbr"

using namespace clang;

namespace clad {

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
void TBRAnalyzer::merge(VarData& targetData, VarData& mergeData) {
  if (targetData.m_Type == VarData::FUND_TYPE) {
    targetData.m_Val.m_FundData =
        targetData.m_Val.m_FundData || mergeData.m_Val.m_FundData;
  } else if (targetData.m_Type == VarData::OBJ_TYPE) {
    for (auto& pair : *targetData.m_Val.m_ArrData)
      merge(pair.second, (*mergeData.m_Val.m_ArrData)[pair.first]);
  } else if (targetData.m_Type == VarData::ARR_TYPE) {
    // FIXME: Currently non-constant indices are not supported in merging.
    for (auto& pair : *targetData.m_Val.m_ArrData) {
      auto it = mergeData.m_Val.m_ArrData->find(pair.first);
      if (it != mergeData.m_Val.m_ArrData->end())
        merge(pair.second, it->second);
    }
    for (auto& pair : *mergeData.m_Val.m_ArrData) {
      auto it = targetData.m_Val.m_ArrData->find(pair.first);
      if (it == mergeData.m_Val.m_ArrData->end())
        (*targetData.m_Val.m_ArrData)[pair.first] = pair.second.copy();
    }
  }
  // This might be useful in future if used to analyse pointers. However, for
  // now it's only used for references for which merging doesn't make sense.
  // else if (this.m_Type == VarData::REF_TYPE) {}
}

TBRAnalyzer::VarData TBRAnalyzer::VarData::copy() const {
  VarData res;
  res.m_Type = m_Type;
  if (m_Type == VarData::FUND_TYPE) {
    res.m_Val.m_FundData = m_Val.m_FundData;
  } else if (m_Type == VarData::OBJ_TYPE || m_Type == VarData::ARR_TYPE) {
    res.m_Val.m_ArrData = std::unique_ptr<ArrMap>(new ArrMap());
    for (auto& pair : *m_Val.m_ArrData)
      (*res.m_Val.m_ArrData)[pair.first] = pair.second.copy();
  } else if (m_Type == VarData::REF_TYPE) {
    res.m_Val.m_RefData = m_Val.m_RefData;
  }
  return res;
}

bool TBRAnalyzer::findReq(const VarData& varData) {
  assert(varData.m_Type != VarData::REF_TYPE &&
         "references should be removed on the getIDSequence stage");
  if (varData.m_Type == VarData::FUND_TYPE)
    return varData.m_Val.m_FundData;
  if (varData.m_Type == VarData::OBJ_TYPE ||
      varData.m_Type == VarData::ARR_TYPE) {
    for (auto& pair : *varData.m_Val.m_ArrData)
      if (findReq(pair.second))
        return true;
  }
  return false;
}

TBRAnalyzer::VarData*
TBRAnalyzer::VarData::operator[](const ProfileID& id) const {
  assert((m_Type == VarData::ARR_TYPE || m_Type == VarData::OBJ_TYPE) &&
         "attempted to get an element of a non-obj non-arr VarData");
  auto& baseArrMap = *m_Val.m_ArrData;
  auto foundElem = baseArrMap.find(id);
  if (foundElem != baseArrMap.end())
    return &foundElem->second;

  assert(m_Type != OBJ_TYPE &&
         "all fields of obj-type VarData should be initialized");
  // Non-const indices are represented with default FoldingSetNodeID.
  ProfileID nonConstIdxID;
  // Add the current index if it was not added previously
  auto& idxData = baseArrMap[id];
  // Since default ID represents non-const indices, whenever we add a
  // new index we have to copy the VarData of default ID's element.
  idxData = baseArrMap[nonConstIdxID].copy();
  return &idxData;
}

TBRAnalyzer::VarData* TBRAnalyzer::getVarDataFromExpr(const clang::Expr* E) {
  llvm::SmallVector<ProfileID, 2> IDSequence;
  const VarDecl* VD = getIDSequence(E, IDSequence);
  VarData* data = getVarDataFromDecl(VD);
  assert(data && "expression not found");
  for (ProfileID& id : IDSequence)
    data = (*data)[id];
  return data;
}

TBRAnalyzer::VarData::VarData(QualType QT, const ASTContext& C,
                              bool forceNonRefType) {
  QT = QT.getDesugaredType(C);
  if ((forceNonRefType && QT->isLValueReferenceType()) ||
      QT->isRValueReferenceType())
    QT = QT->getPointeeType();

  if (QT->isReferenceType()) {
    m_Type = VarData::REF_TYPE;
    m_Val.m_RefData = nullptr;
  } else if (utils::isArrayOrPointerType(QT)) {
    m_Type = VarData::ARR_TYPE;
    m_Val.m_ArrData = std::unique_ptr<ArrMap>(new ArrMap());
    const Type* elemType = nullptr;
    if (const auto* const pointerType = llvm::dyn_cast<clang::PointerType>(QT))
      elemType = pointerType->getPointeeType().getTypePtrOrNull();
    else
      elemType = QT->getArrayElementTypeNoTypeQual();
    ProfileID nonConstIdxID;
    auto& idxData = (*m_Val.m_ArrData)[nonConstIdxID];
    idxData = VarData(QualType::getFromOpaquePtr(elemType), C);
  } else if (QT->isBuiltinType()) {
    m_Type = VarData::FUND_TYPE;
    m_Val.m_FundData = false;
  } else if (const auto* recordType = QT->getAs<RecordType>()) {
    m_Type = VarData::OBJ_TYPE;
    const auto* recordDecl = recordType->getDecl();
    auto& newArrMap = m_Val.m_ArrData;
    newArrMap = std::unique_ptr<ArrMap>(new ArrMap());
    for (const auto* field : recordDecl->fields()) {
      const auto varType = field->getType();
      (*newArrMap)[getProfileID(field)] = VarData(varType, C);
    }
  }
}
const VarDecl*
TBRAnalyzer::getIDSequence(const clang::Expr* E,
                           llvm::SmallVectorImpl<ProfileID>& IDSequence) {
  const VarDecl* innerVD = nullptr;
  // Unwrap the given expression to a vector of indices and fields.
  while (true) {
    E = E->IgnoreCasts();
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
      if (E->getType()->isPointerType())
        IDSequence.push_back(ProfileID());
    } else if (const auto* DRE = dyn_cast<clang::DeclRefExpr>(E)) {
      const auto* VD = cast<VarDecl>(DRE->getDecl());
      if (VD->getType()->isLValueReferenceType()) {
        VarData* refData = getVarDataFromDecl(VD);
        if (refData->m_Type == VarData::REF_TYPE) {
          E = refData->m_Val.m_RefData;
          continue;
        }
      }
      innerVD = VD;
      break;
    } else if (isa<clang::CXXThisExpr>(E)) {
      innerVD = nullptr;
      break;
    } else if (const auto* UO = dyn_cast<UnaryOperator>(E)) {
      const auto opCode = UO->getOpcode();
      // FIXME: Dereference corresponds to the 0's element,
      // not an arbitrary element which is denoted be ProfileID().
      // This is done because it's unclear how to get a 0 literal
      // to generate the ProfileID.
      if (opCode == UO_Deref)
        IDSequence.push_back(ProfileID());
      E = UO->getSubExpr();
    } else {
      assert(0 && "unexpected expression");
      break;
    }
  }
  // All id's were added in the reverse order, e.g. `arr[0].k` -> `k`, `0`.
  // Reverse the sequence for easier handling.
  std::reverse(IDSequence.begin(), IDSequence.end());
  return innerVD;
}

void TBRAnalyzer::setIsRequired(VarData* data, bool isReq,
                                llvm::MutableArrayRef<ProfileID> IDSequence) {
  assert(data->m_Type != VarData::REF_TYPE &&
         "references should be removed on the getIDSequence stage");
  if (data->m_Type == VarData::UNDEFINED)
    return;
  if (data->m_Type == VarData::FUND_TYPE) {
    data->m_Val.m_FundData = isReq;
    return;
  }
  auto& baseArrMap = *data->m_Val.m_ArrData;
  if (IDSequence.empty()) {
    if (isReq)
      for (auto& pair : baseArrMap)
        setIsRequired(&pair.second, /*isReq=*/true, IDSequence);
    return;
  }
  const ProfileID& curID = IDSequence[0];
  if (data->m_Type == VarData::OBJ_TYPE) {
    setIsRequired((*data)[curID], isReq, IDSequence.drop_front());
    return;
  }

  // Arr type arrays are the only option left.
  assert(data->m_Type == VarData::ARR_TYPE);
  // All indices unknown in compile-time (like `arr[i]`) are represented with
  // default FoldingSetNodeID. Note: if we're unsure if an index is used, we
  // always assume it is for safety.
  ProfileID nonConstIdxID;
  if (curID == nonConstIdxID) {
    // The fact that one unknown index is set to unused
    // doesn't mean that all unknown indices should be.
    if (!isReq)
      return;
    // If we set an unknown index to true, we have to do the same to all
    // known indices for safety.
    for (auto& pair : baseArrMap)
      setIsRequired(&pair.second, /*isReq=*/true, IDSequence.drop_front());
  } else {
    setIsRequired((*data)[curID], isReq, IDSequence.drop_front());
    // If we set any index to true, we have to also do it
    // to the default index.
    if (isReq)
      setIsRequired((*data)[nonConstIdxID], /*isReq=*/true,
                    IDSequence.drop_front());
  }
}
// NOLINTEND(cppcoreguidelines-pro-type-union-access)

void TBRAnalyzer::addVar(const clang::VarDecl* VD, bool forceNonRefType) {
  auto& curBranch = getCurBlockVarsData();

  QualType varType;
  if (const auto* arrayParam = dyn_cast<ParmVarDecl>(VD))
    varType = arrayParam->getOriginalType();
  else
    varType = VD->getType();

  // If varType represents auto or auto*, get the type of init.
  if (utils::IsAutoOrAutoPtrType(varType))
    varType = VD->getInit()->getType();

  curBranch[VD] = VarData(varType, m_Context, forceNonRefType);
}

void TBRAnalyzer::markLocation(const clang::Expr* E) {
  m_TBRLocs.insert(E->getBeginLoc());
}

void TBRAnalyzer::setIsRequired(const clang::Expr* E, bool isReq) {
  llvm::SmallVector<ProfileID, 2> IDSequence;
  const VarDecl* VD = getIDSequence(E, IDSequence);
  // Make sure the current branch has a copy of VarDecl for VD
  auto& curBranch = getCurBlockVarsData();
  if (curBranch.find(VD) == curBranch.end()) {
    if (VarData* data = getVarDataFromDecl(VD))
      curBranch[VD] = data->copy();
    else
      // If this variable was not found in predecessors, add it.
      addVar(VD);
  }

  if (!isReq ||
      (m_ModeStack.back() == (Mode::kMarkingMode | Mode::kNonLinearMode))) {
    VarData* data = getVarDataFromDecl(VD);
    setIsRequired(data, isReq, IDSequence);
  }
}

TBRAnalyzer::VarData*
TBRAnalyzer::getVarDataFromDecl(const clang::VarDecl* VD) {
  auto* branch = &getCurBlockVarsData();
  while (branch) {
    auto it = branch->find(VD);
    if (it != branch->end())
      return &it->second;
    branch = branch->m_Prev;
  }
  return nullptr;
}

void TBRAnalyzer::Analyze(const DiffRequest& request) {
  m_BlockData.resize(request.m_AnalysisDC->getCFG()->size());
  m_BlockPassCounter.resize(request.m_AnalysisDC->getCFG()->size(), 0);

  // Set current block ID to the ID of entry the block.
  CFGBlock& entry = request.m_AnalysisDC->getCFG()->getEntry();
  m_CurBlockID = entry.getBlockID();
  m_BlockData[m_CurBlockID] = std::unique_ptr<VarsData>(new VarsData());

  const FunctionDecl* FD = request.Function;
  // If we are analysing a non-static method, add a VarData for 'this' pointer
  // (it is represented with nullptr).
  const auto* MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && !MD->isStatic()) {
    VarData& thisData = getCurBlockVarsData()[nullptr];
    thisData = VarData(MD->getThisType(), m_Context);
    // We have to set all pointer/reference parameters to tbr
    // since method pullbacks aren't supposed to change objects.
    // constructor pullbacks don't take `this` as a parameter
    if (!isa<CXXConstructorDecl>(FD))
      setIsRequired(&thisData);
  }
  auto paramsRef = FD->parameters();
  for (std::size_t i = 0; i < FD->getNumParams(); ++i)
    addVar(paramsRef[i], /*forceNonRefType=*/true);
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

  clang::SourceManager& SM = m_Context.getSourceManager();
  for (SourceLocation Loc : m_TBRLocs) {
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

CFGBlock* TBRAnalyzer::getCFGBlockByID(AnalysisDeclContext* ADC, unsigned ID) {
  return *(ADC->getCFG()->begin() + ID);
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
      branch = branch->m_Prev;
    }

    branch = varsData2;
    while (branch != pred2) {
      if (branch == pred1)
        return branch;
      branch = branch->m_Prev;
    }

    if (pred1->m_Prev) {
      pred1 = pred1->m_Prev;
      // This ensures we don't get an infinite loop because of VarsData being
      // connected in a loop themselves.
      if (pred1 == varsData1)
        return nullptr;
    } else {
      // pred1 not having a predecessor means it is corresponds to the entry
      // block and, therefore it is the lowest common ancestor.
      return pred1;
    }

    if (pred2->m_Prev) {
      pred2 = pred2->m_Prev;
      // This ensures we don't get an infinite loop because of VarsData being
      // connected in a loop themselves.
      if (pred2 == varsData2)
        return nullptr;
    } else {
      // pred2 not having a predecessor means it is corresponds to the entry
      // block and, therefore it is the lowest common ancestor.
      return pred2;
    }
  }
}

std::unordered_map<const clang::VarDecl*, TBRAnalyzer::VarData*>
TBRAnalyzer::collectDataFromPredecessors(VarsData* varsData,
                                         TBRAnalyzer::VarsData* limit) {
  std::unordered_map<const clang::VarDecl*, VarData*> result;
  if (varsData != limit) {
    // Copy data from every predecessor.
    for (auto* pred = varsData; pred != limit; pred = pred->m_Prev) {
      // If a variable from 'pred' is not present in 'result', place it there.
      for (auto& pair : *pred)
        if (result.find(pair.first) == result.end())
          result[pair.first] = &pair.second;
    }
  }

  return result;
}

void TBRAnalyzer::merge(VarsData* targetData, VarsData* mergeData) {
  auto* LCA = findLowestCommonAncestor(targetData, mergeData);
  auto collectedMergeData =
      collectDataFromPredecessors(mergeData, /*limit=*/LCA);

  // For every variable in 'collectedMergeData', search it in targetData and all
  // its predecessors (if found in a predecessor, make a copy to targetData).
  for (auto& pair : collectedMergeData) {
    VarData* found = nullptr;
    auto elemSearch = targetData->find(pair.first);
    if (elemSearch == targetData->end()) {
      auto* branch = targetData->m_Prev;
      while (branch) {
        auto it = branch->find(pair.first);
        if (it != branch->end()) {
          (*targetData)[pair.first] = it->second.copy();
          found = &(*targetData)[pair.first];
          break;
        }
        branch = branch->m_Prev;
      }
    } else {
      found = &elemSearch->second;
    }

    // If the variable was found, perform a merge.  Else, just copy it from
    // collectedMergeData.
    if (found)
      merge(*found, *pair.second);
    else
      (*targetData)[pair.first] = pair.second->copy();
  }

  // For every variable in collected targetData predecessors, search it inside
  // collectedMergeData. If it's not found, that means it was not used anywhere
  // between LCA and mergeData.  To correctly merge, we have to take it from
  // LCA's predecessors and merge it to targetData.  If targetData is LCA, LCA
  // will come after targetData->m_Prev and collectDataFromPredecessors will not
  // reach the limit.
  if (targetData != LCA) {
    for (auto& pair :
         collectDataFromPredecessors(targetData->m_Prev, /*limit=*/LCA)) {
      auto elemSearch = collectedMergeData.find(pair.first);
      if (elemSearch == collectedMergeData.end()) {
        auto* branch = LCA;
        while (branch) {
          auto it = branch->find(pair.first);
          if (it != branch->end()) {
            (*targetData)[pair.first] = pair.second->copy();
            merge((*targetData)[pair.first], it->second);
            break;
          }
          branch = branch->m_Prev;
        }
      }
    }
  }
  // For every variable in targetData, search it inside collectedMergeData. If
  // it's not found, that means it was not used anywhere between LCA and
  // mergeData.  To correctly merge, we have to take it from LCA's predecessors
  // and merge it to targetData.
  for (auto& pair : *targetData) {
    auto elemSearch = collectedMergeData.find(pair.first);
    if (elemSearch == collectedMergeData.end()) {
      auto* branch = LCA;
      while (branch) {
        auto it = branch->find(pair.first);
        if (it != branch->end()) {
          merge(pair.second, it->second);
          break;
        }
        branch = branch->m_Prev;
      }
    }
  }
}

bool TBRAnalyzer::TraverseDeclRefExpr(DeclRefExpr* DRE) {
  setIsRequired(DRE);
  return false;
}

bool TBRAnalyzer::TraverseDeclStmt(DeclStmt* DS) {
  for (auto* D : DS->decls()) {
    if (auto* VD = dyn_cast<VarDecl>(D)) {
      addVar(VD);
      if (clang::Expr* init = VD->getInit()) {

        setMode(Mode::kMarkingMode);
        TraverseStmt(init);
        resetMode();

        auto& VDExpr = getCurBlockVarsData()[VD];
        // if the declared variable is ref type attach its VarData to the
        // VarData of the RHS variable.
        llvm::SmallVector<Expr*, 4> ExprsToStore;
        utils::GetInnermostReturnExpr(init, ExprsToStore);
        if (VDExpr.m_Type == VarData::REF_TYPE) {
          // We only consider references that point to one
          // compile-time defined object.
          if (ExprsToStore.size() == 1)
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
            VDExpr.m_Val.m_RefData = ExprsToStore[0];
          else
            // If we don't know what this points to, mark undefined.
            // FIXME: we should mark all vars on the RHS undefined too.
            VDExpr.m_Type = VarData::UNDEFINED;
        }
      }
    }
  }
  return false;
}

bool TBRAnalyzer::TraverseConditionalOperator(clang::ConditionalOperator* CO) {
  setMode(0);
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
    // Division normally only results in a linear expression when the
    // denominator is constant.
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
      // Since we only care about non-linear usages of variables, there is
      // no difference between operators =, -=, += in terms of TBR analysis.
      TraverseStmt(L);

      startMarkingMode();
      TraverseStmt(R);
      resetMode();
    } else if (opCode == BO_MulAssign || opCode == BO_DivAssign) {
      // *= (/=) normally only performs a linear operation if and only if
      // the RHS is constant. If RHS is not constant, 'x *= y' ('x /= y')
      // represents the same operation as 'x = x * y' ('x = x / y') and,
      // therefore, LHS has to be visited in kMarkingMode|kNonLinearMode.
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
    bool hasToBeSetReq = false;
    for (const auto* innerExpr : ExprsToStore) {
      // If at least one of ExprsToStore has to be stored,
      // mark L as useful to store.
      if (VarData* data = getVarDataFromExpr(innerExpr))
        hasToBeSetReq = hasToBeSetReq || findReq(*data);
      // Set them to not required to store because the values were changed.
      // (if some value was not changed, this could only happen if it was
      // already not required to store).
      setIsRequired(innerExpr, /*isReq=*/false);
    }
    if (hasToBeSetReq)
      markLocation(L);
  } else if (opCode == BO_Comma) {
    setMode(0);
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
    // FIXME: this doesn't support all the possible references
    // Mark corresponding SourceLocation as required/not required to be
    // stored for all expressions that could be used in this operation.
    llvm::SmallVector<Expr*, 4> ExprsToStore;
    utils::GetInnermostReturnExpr(E, ExprsToStore);
    for (const auto* innerExpr : ExprsToStore) {
      // If at least one of ExprsToStore has to be stored,
      // mark L as useful to store.
      if (VarData* data = getVarDataFromExpr(innerExpr))
        if (findReq(*data)) {
          markLocation(E);
          break;
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
  bool noHiddenParam = (CE->getNumArgs() == FD->getNumParams());
  setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
  for (std::size_t i = 0, e = CE->getNumArgs(); i != e; ++i) {
    clang::Expr* arg = CE->getArg(i);
    QualType paramTy;
    if (noHiddenParam)
      paramTy = FD->getParamDecl(i)->getType();
    else if (i != 0)
      paramTy = FD->getParamDecl(i - 1)->getType();

    bool passByRef = false;
    if (!paramTy.isNull())
      passByRef = paramTy->isLValueReferenceType() &&
                  !paramTy.getNonReferenceType().isConstQualified();
    setMode(Mode::kMarkingMode | Mode::kNonLinearMode);
    TraverseStmt(arg);
    resetMode();
    const auto* B = arg->IgnoreParenImpCasts();
    // FIXME: this supports only DeclRefExpr
    if (passByRef) {
      // Mark SourceLocation as required to store for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        m_TBRLocs.insert(arg->getBeginLoc());
        setIsRequired(arg, /*isReq=*/false);
      }
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
      // Mark SourceLocation as required for ref-type arguments.
      if (isa<DeclRefExpr>(B) || isa<MemberExpr>(B)) {
        m_TBRLocs.insert(arg->getBeginLoc());
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

bool TBRAnalyzer::TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* ASE) {
  setMode(0);
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

} // end namespace clad
