#ifndef CLAD_DIFFERENTIATOR_TBRANALYZER_H
#define CLAD_DIFFERENTIATOR_TBRANALYZER_H

#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/CFG.h"

#include "llvm/ADT/ArrayRef.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <map>
#include <unordered_map>

using namespace clang;

namespace clad {

/// Gradient computation requres reversal of the control flow of the original
/// program becomes necessary. To guarantee correctness, certain values that are
/// computed and overwritten in the original program must be made available in
/// the adjoint program. They can be determined by performing a static data flow
/// analysis, the so-called To-Be-Recorded (TBR) analysis. Overestimation of
/// this set must be kept minimal to get efficient adjoint codes.
///
/// This class implements this to-be-recorded analysis.
class TBRAnalyzer : public clang::RecursiveASTVisitor<TBRAnalyzer> {
  /// ProfileID is the key type for ArrMap used to represent array indices
  /// and object fields.
  using ProfileID = clad_compat::FoldingSetNodeID;

  ProfileID getProfileID(const Expr* E) const {
    ProfileID profID;
    E->Profile(profID, m_Context, /* Canonical */ true);
    return profID;
  }

  static ProfileID getProfileID(const FieldDecl* FD) {
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
    enum VarDataType { UNDEFINED, FUND_TYPE, OBJ_TYPE, ARR_TYPE, REF_TYPE };
    union VarDataValue {
      bool m_FundData;
      /// m_ArrData is stored as pointers for VarDataValue to take
      /// less space.
      /// Both arrays and and objects are modelled using m_ArrData;
      std::unique_ptr<ArrMap> m_ArrData;
      Expr* m_RefData;
      VarDataValue() : m_ArrData(nullptr) {}
      /// `= default` cannot be used here since
      /// default destructor is implicitly deleted.
      // NOLINTNEXTLINE(modernize-use-equals-default)
      ~VarDataValue() {}
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
      if (m_Type == FUND_TYPE) {
        m_Val.m_FundData = other.m_Val.m_FundData;
      } else if (m_Type == OBJ_TYPE || m_Type == ARR_TYPE) {
        m_Val.m_ArrData = std::move(other.m_Val.m_ArrData);
      } else if (m_Type == REF_TYPE) {
        m_Val.m_RefData = other.m_Val.m_RefData;
      }
      other.m_Type = UNDEFINED;
      return *this;
    }

    /// Builds a VarData object (and its children) based on the provided type.
    /// If `forceNonRefType` is true, the constructed VarData will not be of
    /// reference type (it will store TBR information itself without referring
    /// to other VarData's). This is necessary for reference-type parameters,
    /// when the referenced expressions are out of the function's scope.
    VarData(QualType QT, const ASTContext& C, bool forceNonRefType = false);

    /// Erases all children VarData's of this VarData.
    ~VarData() {
      if (m_Type == OBJ_TYPE || m_Type == ARR_TYPE)
        m_Val.m_ArrData.reset();
    }

    /// Used to recursively copy VarData when separating into different branches
    /// (e.g. when entering an if-else statements). Look at the Control Flow
    /// section for more information.
    [[nodiscard]] VarData copy() const;
    /// Provides access to the nested VarData if the given VarData represents
    /// an array or a structure. Generates new array elements if necessary.
    VarData* operator[](const ProfileID& id) const;
  };
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)

  /// For a compound lvalue expr, generates a sequence of ProfileID's of it's
  /// indices/fields and returns the VarDecl of the base, e.g.
  /// ``arr[k].y`` --> returns `arr`, IDSequence = `{k, y}`.
  const clang::VarDecl*
  getIDSequence(const clang::Expr* E,
                llvm::SmallVectorImpl<ProfileID>& IDSequence);
  /// Returns true if there is at least one required to store node among
  /// child nodes.
  bool findReq(const VarData& varData);
  /// Used to merge together VarData for one variable from two branches
  /// (e.g. after an if-else statements). Look at the Control Flow section for
  /// more information.
  void merge(VarData& targetData, VarData& mergeData);

  clang::CFGBlock* getCFGBlockByID(unsigned ID);

  /// Given an Expr* returns its corresponding VarData. If the given element of
  /// an array does not have a VarData yet it will be added automatically.
  VarData* getVarDataFromExpr(const clang::Expr* E);
  /// Finds VD in the most recent block.
  VarData* getVarDataFromDecl(const clang::VarDecl* VD);
  // A helper function that recursively sets all nodes to the requested value of
  // isReq.
  void setIsRequired(VarData* targetData, bool isReq = true,
                     llvm::MutableArrayRef<ProfileID> IDSequence = {});
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
      if (&m_Data == &other.m_Data) {
        m_Data = std::move(other.m_Data);
        m_Prev = other.m_Prev;
      }
      return *this;
    }

    using iterator =
        std::unordered_map<const clang::VarDecl*, VarData>::iterator;
    iterator begin() { return m_Data.begin(); }
    iterator end() { return m_Data.end(); }
    VarData& operator[](const clang::VarDecl* VD) { return m_Data[VD]; }
    iterator find(const clang::VarDecl* VD) { return m_Data.find(VD); }
    void clear() { m_Data.clear(); }
  };

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
  void merge(VarsData* targetData, VarsData* mergeData);

  /// Used to find DeclRefExpr's that will be used in the backwards pass.
  /// In order to be marked as required, a variables has to appear in a place
  /// where it would have a differential influence and will appear non-linearly
  /// (e.g. for 'x = 2 * y;', y will not appear in the backwards pass). Hence,
  /// markingMode and nonLinearMode.
  enum Mode { kMarkingMode = 1, kNonLinearMode = 2 };
  /// Tells if the variable at a given location is required to store. Basically,
  /// is the result of analysis.
  std::set<clang::SourceLocation>& m_TBRLocs;

  /// Stores modes in a stack (used to retrieve the old mode after entering
  /// a new one).
  std::vector<int> m_ModeStack;

  ASTContext& m_Context;

  /// clang::CFG of the function being analysed.
  std::unique_ptr<clang::CFG> m_CFG;

  /// Stores VarsData structures for CFG blocks (the indices in
  /// the vector correspond to CFG blocks' IDs)
  std::vector<std::unique_ptr<VarsData>> m_BlockData;

  /// Stores the number of performed passes for a given CFG block index.
  std::vector<short> m_BlockPassCounter;

  /// ID of the CFG block being visited.
  unsigned m_CurBlockID{};

  /// The set of IDs of the CFG blocks that should be visited.
  std::set<unsigned> m_CFGQueue;

  //// Setters
  /// Creates VarData for a new VarDecl*.
  void addVar(const clang::VarDecl* VD, bool forceNonRefType = false);
  /// Marks the SourceLocation of E if it is required to store.
  /// E could be DeclRefExpr*, ArraySubscriptExpr* or MemberExpr*.
  void markLocation(const clang::Expr* E);
  /// Sets E's corresponding VarData (or all its child nodes) to
  /// required/not required. For isReq==true, checks if the current mode is
  /// markingMode and nonLinearMode. E could be DeclRefExpr*,
  /// ArraySubscriptExpr* or MemberExpr*.
  void setIsRequired(const clang::Expr* E, bool isReq = true);

  /// Returns the VarsData of the CFG block being visited.
  VarsData& getCurBlockVarsData() { return *m_BlockData[m_CurBlockID]; }

  //// Modes Setters
  /// Sets the mode manually
  void setMode(int mode) { m_ModeStack.push_back(mode); }
  /// Sets nonLinearMode but leaves markingMode just as it was.
  void startNonLinearMode() {
    m_ModeStack.push_back(m_ModeStack.back() | Mode::kNonLinearMode);
  }
  /// Sets markingMode but leaves nonLinearMode just as it was.
  void startMarkingMode() {
    m_ModeStack.push_back(Mode::kMarkingMode | m_ModeStack.back());
  }
  /// Removes the last mode in the stack (retrieves the previous one).
  void resetMode() { m_ModeStack.pop_back(); }

public:
  /// Constructor
  TBRAnalyzer(ASTContext& Context, std::set<clang::SourceLocation>& Locs)
      : m_TBRLocs(Locs), m_Context(Context) {
    m_ModeStack.push_back(0);
  }

  /// Destructor
  ~TBRAnalyzer() = default;

  /// Delete copy/move operators and constructors.
  TBRAnalyzer(const TBRAnalyzer&) = delete;
  TBRAnalyzer& operator=(const TBRAnalyzer&) = delete;
  TBRAnalyzer(const TBRAnalyzer&&) = delete;
  TBRAnalyzer& operator=(const TBRAnalyzer&&) = delete;

  /// Visitors
  void Analyze(const clang::FunctionDecl* FD);

  void VisitCFGBlock(const clang::CFGBlock& block);

  bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* ASE);
  bool TraverseBinaryOperator(clang::BinaryOperator* BinOp);
  bool TraverseCallExpr(clang::CallExpr* CE);
  bool TraverseConditionalOperator(clang::ConditionalOperator* CO);
  bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* BinOp);
  bool TraverseCXXConstructExpr(clang::CXXConstructExpr* CE);
  bool TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* CE);
  bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* CE);
  bool TraverseDeclRefExpr(clang::DeclRefExpr* DRE);
  bool TraverseDeclStmt(clang::DeclStmt* DS);
  bool TraverseInitListExpr(clang::InitListExpr* ILE);
  bool TraverseMemberExpr(clang::MemberExpr* ME);
  bool TraverseUnaryOperator(clang::UnaryOperator* UnOp);
};

} // end namespace clad
#endif // CLAD_DIFFERENTIATOR_TBRANALYZER_H
