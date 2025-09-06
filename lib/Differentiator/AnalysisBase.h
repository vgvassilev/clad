#ifndef CLAD_DIFFERENTIATOR_ANALYSISBASE_H
#define CLAD_DIFFERENTIATOR_ANALYSISBASE_H

#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace clad {
/// ProfileID is the key type for ArrMap used to represent array indices
/// and object fields.
using ProfileID = clad_compat::FoldingSetNodeID;

inline ProfileID getProfileID(const clang::Expr* E,
                              clang::ASTContext& Context) {
  ProfileID profID;
  E->Profile(profID, Context, /*Canonical=*/true);
  return profID;
}

inline ProfileID getProfileID(const clang::FieldDecl* FD) {
  ProfileID profID;
  profID.AddPointer(FD);
  return profID;
}

struct ProfileIDHash {
  size_t operator()(const ProfileID& x) const { return x.ComputeHash(); }
};

struct VarData;
using ArrMap = std::unordered_map<const ProfileID, VarData, ProfileIDHash>;

// NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
/// Stores all the necessary information about one variable. Fundamental type
/// variables need only one bit. An object/array needs a separate VarData for
/// each field/element. Reference type variables store the clang::Expr* they
/// refer to. UNDEFINED is used whenever the type of a node cannot be
/// determined.
///
/// FIXME: Pointers to objects are considered OBJ_TYPE for simplicity. This
/// approach might cause problems when the support for pointers is added.
///
/// FIXME: Add support for references to call expression results.
/// 'double& x = f(b);' is not supported.
struct VarData {
  enum VarDataType : std::uint8_t {
    UNDEFINED,
    FUND_TYPE,
    OBJ_TYPE,
    ARR_TYPE,
    REF_TYPE
  };
  union VarDataValue {
    bool m_FundData;
    /// m_ArrData is stored as pointers for VarDataValue to take
    /// less space.
    /// Both arrays and and objects are modelled using m_ArrData;
    std::unique_ptr<ArrMap> m_ArrData;
    std::unique_ptr<std::set<const clang::VarDecl*>> m_RefData;
    VarDataValue() : m_ArrData(nullptr) {}
    /// `= default` cannot be used here since
    /// default destructor is implicitly deleted.
    // NOLINTNEXTLINE(modernize-use-equals-default)
    ~VarDataValue() {}

    VarDataValue(const VarDataValue&) = delete;
    VarDataValue& operator=(const VarDataValue&) = delete;
    VarDataValue(VarDataValue&&) = delete;
    VarDataValue& operator=(VarDataValue&&) = delete;
  };
  VarDataType m_Type = UNDEFINED;
  VarDataValue m_Val;

  VarData() = default;
  VarData(const VarData& other) = delete;
  VarData(VarData&& other) noexcept : m_Type(other.m_Type) {
    *this = std::move(other);
  }
  VarData& operator=(const VarData& other) = delete;
  VarData& operator=(VarData&& other) noexcept {
    m_Type = other.m_Type;
    if (m_Type == FUND_TYPE)
      m_Val.m_FundData = other.m_Val.m_FundData;
    else if (m_Type == OBJ_TYPE || m_Type == ARR_TYPE)
      m_Val.m_ArrData = std::move(other.m_Val.m_ArrData);
    else if (m_Type == REF_TYPE)
      m_Val.m_RefData = std::move(other.m_Val.m_RefData);
    other.m_Type = UNDEFINED;
    return *this;
  }

  /// Builds a VarData object (and its children) based on the provided type.
  /// If `forceInit` is true, the constructed VarData will never be of
  /// reference type (it will store TBR information itself without referring
  /// to other VarData's). This is necessary for reference-type parameters,
  /// when the referenced expressions are out of the function's scope.
  VarData(clang::QualType QT, bool forceInit = false);

  /// Erases all children VarData's of this VarData.
  ~VarData() {
    if (m_Type == OBJ_TYPE || m_Type == ARR_TYPE)
      m_Val.m_ArrData.reset();
    else if (m_Type == REF_TYPE)
      m_Val.m_RefData.reset();
  }

  /// Used to recursively copy VarData when separating into different branches
  /// (e.g. when entering an if-else statements). Look at the Control Flow
  /// section for more information.
  [[nodiscard]] VarData copy() const;
  void initializeAsArray(clang::QualType QT);
  /// Provides access to the nested VarData if the given VarData represents
  /// an array or a structure. Generates new array elements if necessary.
  VarData* operator[](const ProfileID& id) const;
};
/// Used to store all the necessary information about variables at a
/// particular moment.
/// Note: the VarsData of one CFG block only stores information specific
/// to that block and relies on the VarsData of it's predecessors
/// for old information. This is done to avoid excessive copying
/// memory and usage.
/// Note: 'this' pointer does not have a declaration so nullptr is used as
/// its key instead.
struct VarsData {
  std::unordered_map<const clang::VarDecl*, VarData> m_Data;
  VarsData* m_Prev = nullptr;

  VarsData() = default;
  VarsData(const VarsData& other) = default;
  ~VarsData() = default;
  VarsData(VarsData&& other) noexcept
      : m_Data(std::move(other.m_Data)), m_Prev(other.m_Prev) {}
  VarsData& operator=(const VarsData& other) = delete;
  VarsData& operator=(VarsData&& other) noexcept {
    if (&m_Data != &other.m_Data) {
      m_Data = std::move(other.m_Data);
      m_Prev = other.m_Prev;
    }
    return *this;
  }

  using iterator = std::unordered_map<const clang::VarDecl*, VarData>::iterator;
  iterator begin() { return m_Data.begin(); }
  iterator end() { return m_Data.end(); }
  VarData& operator[](const clang::VarDecl* VD) { return m_Data[VD]; }
  iterator find(const clang::VarDecl* VD) { return m_Data.find(VD); }
  void clear() { m_Data.clear(); }
};
// NOLINTEND(cppcoreguidelines-pro-type-union-access)
class AnalysisBase {
protected:
  clang::AnalysisDeclContext* m_AnalysisDC;
  /// Stores VarsData structures for CFG blocks (the indices in
  /// the vector correspond to CFG blocks' IDs)
  std::vector<std::unique_ptr<VarsData>> m_BlockData;
  /// The set of IDs of the CFG blocks that should be visited.
  std::set<unsigned> m_CFGQueue;
  /// ID of the CFG block being visited.
  unsigned m_CurBlockID{};
  const clang::FunctionDecl* m_Function = nullptr;

  static clang::CFGBlock* getCFGBlockByID(clang::AnalysisDeclContext* ADC,
                                          unsigned ID);
  AnalysisBase(clang::AnalysisDeclContext* AnalysisDC)
      : m_AnalysisDC(AnalysisDC) {}

  /// For a compound lvalue expr, generates a sequence of ProfileID's of it's
  /// indices/fields and returns the VarDecl of the base, e.g.
  /// ``arr[k].y`` --> returns `arr`, IDSequence = `{k, y}`.
  bool getIDSequence(const clang::Expr* E, const clang::VarDecl*& VD,
                     llvm::SmallVectorImpl<ProfileID>& IDSequence);

  /// Returns true if there is at least one required to store node among
  /// child nodes.
  bool findReq(const VarData& varData);
  /// Returns true if there is at least one required to store sub-expr.
  bool findReq(const clang::Expr* E);
  /// Used to merge together VarData for one variable from two branches
  /// (e.g. after an if-else statements). Look at the Control Flow section for
  /// more information.
  bool merge(VarData& targetData, VarData& mergeData);
  /// Creates VarData for a new VarDecl*.
  void addVar(const clang::VarDecl* VD, bool forceInit = false);
  /// Finds VD in the most recent block.
  VarData* getVarDataFromDecl(const clang::VarDecl* VD);
  // A helper function that recursively sets all nodes to the requested value of
  // isReq.
  void setIsRequired(VarData* targetData, bool isReq = true,
                     llvm::MutableArrayRef<ProfileID> IDSequence = {});
  /// Sets E's corresponding VarData (or all its child nodes) to
  /// required/not required. For isReq==true, checks if the current mode is
  /// markingMode and nonLinearMode. E could be DeclRefExpr*,
  /// ArraySubscriptExpr* or MemberExpr*.
  void setIsRequired(const clang::Expr* E, bool isReq = true);
  /// Collects the data from 'varsData' and its predecessors until
  /// 'limit' into one map ('limit' VarsData is not included).
  /// If 'limit' is 'nullptr', data is collected starting with
  /// the entry CFG block.
  /// Note: the returned VarsData contains original data from
  /// the predecessors (NOT copies). It should not be modified.
  std::unordered_map<
      const clang::VarDecl*,
      VarData*> static collectDataFromPredecessors(VarsData* varsData,
                                                   VarsData* limit = nullptr);
  /// Finds the lowest common ancestor of two VarsData
  /// (based on the m_Prev field in VarsData).

  static VarsData* findLowestCommonAncestor(VarsData* varsData1,
                                            VarsData* varsData2);
  /// Merges mergeData into targetData. Should be called
  /// after mergeData is passed and the corresponding CFG
  /// block is one of the predecessors of targetData's CFG block
  /// (e.g. instance when merging if- and else- blocks).
  /// Note: The first predecessor (targetData->m_Prev) does NOT have
  /// to be merged to targetData.
  bool merge(VarsData* targetData, VarsData* mergeData);
  /// Returns the VarsData of the CFG block being visited.

  VarsData& getCurBlockVarsData() { return *m_BlockData[m_CurBlockID]; }
  /// Determines the set of all variables that the expression E depends on.
  static void getDependencySet(const clang::Expr* E,
                               std::set<const clang::VarDecl*>& vars);
};

} // namespace clad
#endif // CLAD_DIFFERENTIATOR_ANALYSISBASE_H
