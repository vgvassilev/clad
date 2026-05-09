#include "AnalysisBase.h"

#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <set>
#include <unordered_map>

using namespace clang;

namespace clad {

void AnalysisBase::addVar(const clang::VarDecl* VD, bool forceInit) {
  auto& curBranch = getCurBlockVarsData();
  QualType varType;
  if (const auto* arrayParam = dyn_cast<ParmVarDecl>(VD))
    varType = arrayParam->getOriginalType();
  else
    varType = VD->getType();

  // If varType represents auto or auto*, get the type of init.
  if (utils::IsAutoOrAutoPtrType(varType))
    varType = VD->getInit()->getType();

  curBranch[VD] = VarData(varType, forceInit);
}

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
VarData::VarData(QualType QT, bool forceInit) {
  QT = QT.getCanonicalType();
  if ((forceInit && QT->isLValueReferenceType()) || QT->isRValueReferenceType())
    QT = QT->getPointeeType();
  m_Type = VarData::FUND_TYPE;
  m_Val.m_FundData = false;
  if (QT->isReferenceType()) {
    m_Type = VarData::REF_TYPE;
    m_Val.m_RefData = std::make_unique<std::set<const clang::VarDecl*>>();
  } else if (QT->isArrayType() || (forceInit && QT->isPointerType())) {
    initializeAsArray(QT);
  } else if (QT->isBuiltinType()) {
    m_Type = VarData::FUND_TYPE;
    m_Val.m_FundData = false;
  } else if (const auto* recordType = QT->getAs<RecordType>()) {
    m_Type = VarData::OBJ_TYPE;
    const auto* recordDecl = recordType->getDecl();
    auto& newArrMap = m_Val.m_ArrData;
    newArrMap = std::make_unique<ArrMap>();
    llvm::SmallVector<const FieldDecl*, 4> Fields;
    utils::getRecordDeclFields(recordDecl, Fields);
    for (const auto* field : Fields) {
      const auto varType = field->getType();
      (*newArrMap)[getProfileID(field)] = VarData(varType);
    }
  }
}

void VarData::initializeAsArray(QualType QT) {
  assert(utils::isArrayOrPointerType(QT) &&
         "QT must represent an array or a pointer");
  m_Type = VarData::ARR_TYPE;
  new (&m_Val.m_RefData) std::unique_ptr<ArrMap>(std::make_unique<ArrMap>());
  QualType elemType = utils::GetValueType(QT);
  // Default ProfileID corresponds to non-const indices
  m_Val.m_ArrData->emplace(ProfileID{}, VarData(elemType));
}

VarData VarData::copy() const {
  VarData res;
  res.m_Type = m_Type;
  if (m_Type == VarData::FUND_TYPE) {
    res.m_Val.m_FundData = m_Val.m_FundData;
  } else if (m_Type == VarData::OBJ_TYPE || m_Type == VarData::ARR_TYPE) {
    res.m_Val.m_ArrData = std::make_unique<ArrMap>();
    for (auto& pair : *m_Val.m_ArrData)
      (*res.m_Val.m_ArrData)[pair.first] = pair.second.copy();
  } else if (m_Type == VarData::REF_TYPE) {
    new (&res.m_Val.m_RefData) std::unique_ptr<std::set<const VarDecl*>>(
        std::make_unique<std::set<const VarDecl*>>());
    *res.m_Val.m_RefData = *m_Val.m_RefData;
  }
  return res;
}

VarData* VarData::operator[](const ProfileID& id) const {
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

void AnalysisBase::getDependencySet(const clang::Expr* E,
                                    std::set<const clang::VarDecl*>& vars) {
  class DeclFinder : public RecursiveASTVisitor<DeclFinder> {
  public:
    std::set<const clang::VarDecl*>& vars;
    DeclFinder(std::set<const clang::VarDecl*>& pvars) : vars(pvars) {}
    bool TraverseDeclRefExpr(DeclRefExpr* DRE) {
      if (auto* VD = dyn_cast<VarDecl>(DRE->getDecl()))
        vars.insert(VD);
      return false;
    }
    bool TraverseArraySubscriptExpr(ArraySubscriptExpr* ASE) {
      TraverseStmt(ASE->getBase());
      return false;
    }
    bool TraverseCXXThisExpr(CXXThisExpr* TE) {
      vars.insert(nullptr);
      return false;
    }
  };
  DeclFinder finder(vars);
  finder.TraverseStmt(const_cast<Expr*>(E));
}

CFGBlock* AnalysisBase::getCFGBlockByID(AnalysisDeclContext* ADC, unsigned ID) {
  return *(ADC->getCFG()->begin() + ID);
}

bool AnalysisBase::getIDSequence(const clang::Expr* E, const VarDecl*& VD,
                                 llvm::SmallVectorImpl<ProfileID>& IDSequence) {
  // Unwrap the given expression to a vector of indices and fields.
  while (true) {
    E = E->IgnoreParenCasts();
    if (const auto* ASE = dyn_cast<clang::ArraySubscriptExpr>(E)) {
      if (const auto* IL = dyn_cast<clang::IntegerLiteral>(ASE->getIdx()))
        IDSequence.push_back(getProfileID(IL, m_AnalysisDC->getASTContext()));
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
      VD = dyn_cast<VarDecl>(DRE->getDecl());
      if (!VD)
        return false;
      QualType VDType = VD->getType();
      if (VDType->isLValueReferenceType() || VDType->isPointerType()) {
        VarData* refData = getVarDataFromDecl(VD);
        if (refData->m_Type == VarData::REF_TYPE) {
          std::set<const VarDecl*>& vars = *refData->m_Val.m_RefData;
          if (vars.size() == 1) {
            VD = *vars.begin();
            QualType VDType =
                VD ? VD->getType()
                   : cast<CXXMethodDecl>(m_Function)->getThisType();
            if (utils::isSameCanonicalType(VDType, E->getType()))
              break;
          }
          IDSequence.clear();
          VD = nullptr;
          return false;
        }
      }
      break;
    } else if (isa<clang::CXXThisExpr>(E)) {
      VD = nullptr;
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
      IDSequence.clear();
      VD = nullptr;
      return false;
    }
  }
  // All id's were added in the reverse order, e.g. `arr[0].k` -> `k`, `0`.
  // Reverse the sequence for easier handling.
  std::reverse(IDSequence.begin(), IDSequence.end());
  return true;
}

bool AnalysisBase::findReq(const VarData& varData) {
  if (varData.m_Type == VarData::FUND_TYPE)
    return varData.m_Val.m_FundData;
  if (varData.m_Type == VarData::OBJ_TYPE ||
      varData.m_Type == VarData::ARR_TYPE) {
    for (auto& pair : *varData.m_Val.m_ArrData)
      if (findReq(pair.second))
        return true;
  }
  if (varData.m_Type == VarData::REF_TYPE)
    for (const VarDecl* VD : *varData.m_Val.m_RefData)
      if (findReq(*getVarDataFromDecl(VD)))
        return true;
  return false;
}

bool AnalysisBase::findReq(const Expr* E) {
  llvm::SmallVector<ProfileID, 2> IDSequence;
  const VarDecl* VD = nullptr;
  if (getIDSequence(E, VD, IDSequence)) {
    VarData* data = getVarDataFromDecl(VD);
    for (ProfileID& id : IDSequence)
      data = (*data)[id];
    return findReq(*data);
  }

  std::set<const clang::VarDecl*> vars;
  getDependencySet(E, vars);
  for (const VarDecl* VD : vars)
    if (findReq(*getVarDataFromDecl(VD)))
      return true;
  return false;
}

bool AnalysisBase::merge(VarData& targetData, VarData& mergeData) {
  if (targetData.m_Type == VarData::FUND_TYPE) {
    bool oldTargetVal = targetData.m_Val.m_FundData;
    targetData.m_Val.m_FundData =
        targetData.m_Val.m_FundData || mergeData.m_Val.m_FundData;
    return oldTargetVal != targetData.m_Val.m_FundData;
  } else if (targetData.m_Type == VarData::OBJ_TYPE) {
    bool isMod = false;
    for (auto& pair : *targetData.m_Val.m_ArrData)
      isMod =
          merge(pair.second, (*mergeData.m_Val.m_ArrData)[pair.first]) || isMod;
    return isMod;
  } else if (targetData.m_Type == VarData::ARR_TYPE) {
    bool isMod = false;
    // FIXME: Currently non-constant indices are not supported in merging.
    for (auto& pair : *targetData.m_Val.m_ArrData) {
      auto it = mergeData.m_Val.m_ArrData->find(pair.first);
      if (it != mergeData.m_Val.m_ArrData->end())
        isMod = merge(pair.second, it->second) || isMod;
    }
    for (auto& pair : *mergeData.m_Val.m_ArrData) {
      auto it = targetData.m_Val.m_ArrData->find(pair.first);
      if (it == targetData.m_Val.m_ArrData->end())
        (*targetData.m_Val.m_ArrData)[pair.first] = pair.second.copy();
    }
    return isMod;
  }
  return false;
  // This might be useful in future if used to analyse pointers. However, for
  // now it's only used for references for which merging doesn't make sense.
  // else if (this.m_Type == VarData::REF_TYPE) {}
}

VarData* AnalysisBase::getVarDataFromDecl(const clang::VarDecl* VD) {
  auto* branch = &getCurBlockVarsData();
  while (branch) {
    auto it = branch->find(VD);
    if (it != branch->end())
      return &it->second;
    branch = branch->m_Prev;
  }
  return nullptr;
}

void AnalysisBase::setIsRequired(VarData* data, bool isReq,
                                 llvm::MutableArrayRef<ProfileID> IDSequence) {
  if (data->m_Type == VarData::UNDEFINED)
    return;
  if (data->m_Type == VarData::FUND_TYPE) {
    data->m_Val.m_FundData = isReq;
    return;
  }
  if (data->m_Type == VarData::REF_TYPE) {
    std::set<const VarDecl*>& vars = *data->m_Val.m_RefData;
    if (isReq || vars.size() == 1)
      for (const VarDecl* VD : vars)
        setIsRequired(getVarDataFromDecl(VD), isReq, IDSequence);
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

std::unordered_map<const clang::VarDecl*, VarData*>
AnalysisBase::collectDataFromPredecessors(VarsData* varsData, VarsData* limit) {
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

VarsData* AnalysisBase::findLowestCommonAncestor(VarsData* varsData1,
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

bool AnalysisBase::merge(VarsData* targetData, VarsData* mergeData) {
  bool isModified = false;
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
    if (found) {
      if (merge(*found, *pair.second))
        isModified = true;
    } else
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
  return isModified;
}
// NOLINTEND(cppcoreguidelines-pro-type-union-access)
} // namespace clad
