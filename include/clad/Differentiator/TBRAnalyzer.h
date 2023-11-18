#ifndef CLAD_DIFFERENTIATOR_TBRANALYZER_H
#define CLAD_DIFFERENTIATOR_TBRANALYZER_H

#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/CFG.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <map>
#include <unordered_map>

using namespace clang;

namespace clad {

class TBRAnalyzer : public clang::ConstStmtVisitor<TBRAnalyzer> {
private:

  /// ProfileID is the key type for ArrMap used to represent array indices
  /// and object fields.
  using ProfileID = clad_compat::FoldingSetNodeID;

  ProfileID getProfileID(const Expr* E) const{
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
    size_t operator()(const ProfileID& x) const {
      return x.ComputeHash();
    }
  };

  /// Stores all the necessary information about one variable. Fundamental type
  /// variables need only one bit. An object/array needs a separate VarData for
  /// each field/element. Reference type variables store the clang::Expr* they
  /// refer to. UNDEFINED is used whenever the type of a node cannot be determined.

  /// FIXME: Pointers to objects are considered OBJ_TYPE for simplicity. This
  /// approach might cause problems when the support for pointers is added.

  /// FIXME: Add support for references to call expression results.
  /// 'double& x = f(b);' is not supported.

  struct VarData;
  using ArrMap =
      std::unordered_map<const ProfileID, VarData, ProfileIDHash>;

  // NOLINTBEGIN(cppcoreguidelines-pro-type-union-access)
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
      ~VarDataValue() {}
    };
    VarDataType type = UNDEFINED;
    VarDataValue val;

    VarData() = default;
    VarData(const VarData& other) = delete;
    VarData(VarData&& other) noexcept: type(other.type) {
      *this = std::move(other);
    }
    VarData& operator=(const VarData& other) = delete;
    VarData& operator=(VarData&& other) noexcept {
      if (this!=&other) {
        type = other.type;
        if (type == FUND_TYPE) {
          val.m_FundData = other.val.m_FundData;
        } else if (type == OBJ_TYPE || type == ARR_TYPE) {
          val.m_ArrData = std::move(other.val.m_ArrData);
          other.val.m_ArrData = nullptr;
        } else if (type == REF_TYPE) {
          val.m_RefData = other.val.m_RefData;
        }
        other.type = UNDEFINED;
      }
      return *this;
    }

    /// Builds a VarData object (and its children) based on the provided type.
    VarData(QualType QT);

    /// Erases all children VarData's of this VarData.
    ~VarData() {
      if (type == OBJ_TYPE || type == ARR_TYPE) {
        val.m_ArrData.reset();
      }
    }
  };
  // NOLINTEND(cppcoreguidelines-pro-type-union-access)

  /// Recursively sets all the leaves' bools to isReq.
  void setIsRequired(VarData& varData, bool isReq = true);
  /// Whenever an array element with a non-constant index is set to required
  /// this function is used to set to required all the array elements that
  /// could match that element (e.g. set 'a[1].y' and 'a[6].y' to required
  /// when 'a[k].y' is set to required). Takes unwrapped sequence of
  /// indices/members of the expression being overlaid and the index of of the
  /// current index/member.
  void overlay(VarData& targetData,
               llvm::SmallVector<ProfileID, 2>& IDSequence,
               size_t i);
  /// Returns true if there is at least one required to store node among
  /// child nodes.
  bool findReq(const VarData& varData);
  /// Used to merge together VarData for one variable from two branches
  /// (e.g. after an if-else statements). Look at the Control Flow section for
  /// more information.
  void merge(VarData& targetData, VarData& mergeData);
  /// Used to recursively copy VarData when separating into different branches
  /// (e.g. when entering an if-else statements). Look at the Control Flow
  /// section for more information.
  VarData copy(VarData& copyData);

  clang::CFGBlock* getCFGBlockByID(unsigned ID);

  /// Given a MemberExpr*/ArraySubscriptExpr* return a pointer to its
  /// corresponding VarData. If the given element of an array does not have a
  /// VarData yet it will be added automatically. If addNonConstIdx==false this
  /// will return the last VarData before the non-constant index
  /// (e.g. for 'x.arr[k+1].y' the return value will be the VarData of x.arr).
  /// Otherwise, non-const indices will be represented as index -1.
  VarData* getMemberVarData(const clang::MemberExpr* ME,
                            bool addNonConstIdx = false);
  VarData* getArrSubVarData(const clang::ArraySubscriptExpr* ASE,
                            bool addNonConstIdx = false);
  /// Given an Expr* returns its corresponding VarData.
  VarData* getExprVarData(const clang::Expr* E, bool addNonConstIdx = false);

  /// Whenever an array element with a non-constant index is set to required
  /// this function is used to set to required all the array elements that
  /// could match that element (e.g. set 'a[1].y' and 'a[6].y' to required when
  /// 'a[k].y' is set to required). Unwraps the a given expression into a
  /// sequence of indices/members of the expression being overlaid and calls
  /// VarData::overlay() recursively.
  void overlay(const clang::Expr* E);



  /// Used to store all the necessary information about variables at a
  /// particular moment.
  /// Note: the VarsData of one CFG block only stores information specific
  /// to that block and relies on the VarsData of it's predecessors
  /// for old information. This is done to avoid excessive copying
  /// memory and usage.
  /// Note: 'this' pointer does not have a declaration so nullptr is used as
  /// its key instead.
  struct VarsData {
    std::unordered_map<const clang::VarDecl*, VarData> data;
    VarsData* prev = nullptr;

    VarsData() = default;
    VarsData(const VarsData& other) = default;
    VarsData(VarsData&& other) noexcept : data(std::move(other.data)), prev(other.prev) {}
    VarsData& operator=(const VarsData& other) = delete;
    VarsData& operator=(VarsData&& other) noexcept {
      if (&data == &other.data) {
        data = std::move(other.data);
        prev = other.prev;
      }
      return *this;
    }

    using iterator =
        std::unordered_map<const clang::VarDecl*, VarData>::iterator;
    iterator begin() { return data.begin(); }
    iterator end() { return data.end(); }
    VarData& operator[](const clang::VarDecl* VD) { return data[VD]; }
    iterator find(const clang::VarDecl* VD) { return data.find(VD); }
    void clear() {
        data.clear();
    }
  };


  /// Collects the data from 'varsData' and its predecessors until
  /// 'limit' into one map ('limit' VarsData is not included).
  /// If 'limit' is 'nullptr', data is collected starting with
  /// the entry CFG block.
  /// Note: the returned VarsData contains original data from
  /// the predecessors (NOT copies). It should not be modified.
  std::unordered_map<const clang::VarDecl*, VarData*>
  static collectDataFromPredecessors(VarsData* varsData, VarsData* limit = nullptr);

  /// Finds the lowest common ancestor of two VarsData
  /// (based on the prev field in VarsData).
  static VarsData* findLowestCommonAncestor(VarsData* varsData1, VarsData* varsData2);

  /// Merges mergeData into targetData. Should be called
  /// after mergeData is passed and the corresponding CFG
  /// block is one of the predecessors of targetData's CFG block
  /// (e.g. instance when merging if- and else- blocks).
  /// Note: The first predecessor (targetData->prev) does NOT have
  /// to be merged to targetData.
  void merge(VarsData* targetData, VarsData* mergeData);

  /// Used to find DeclRefExpr's that will be used in the backwards pass.
  /// In order to be marked as required, a variables has to appear in a place
  /// where it would have a differential influence and will appear non-linearly
  /// (e.g. for 'x = 2 * y;', y will not appear in the backwards pass). Hence,
  /// markingMode and nonLinearMode.
  enum Mode { markingMode = 1, nonLinearMode = 2 };
  /// Tells if the variable at a given location is required to store. Basically,
  /// is the result of analysis.
  std::set<clang::SourceLocation> TBRLocs;

  /// Stores modes in a stack (used to retrieve the old mode after entering
  /// a new one).
  std::vector<int> modeStack;

  ASTContext& m_Context;

  /// clang::CFG of the function being analysed.
  std::unique_ptr<clang::CFG> m_CFG;

  /// Stores VarsData structures for CFG blocks (the indices in
  /// the vector correspond to CFG blocks' IDs)
  std::vector<std::unique_ptr<VarsData>> blockData;

  /// Stores the number of performed passes for a given CFG block index.
  std::vector<short> blockPassCounter;

  /// ID of the CFG block being visited.
  unsigned curBlockID{};

  /// The set of IDs of the CFG blocks that should be visited.
  std::set<unsigned> CFGQueue;

  /// Set to true when a non-const index is found while analysing an
  /// array subscript expression.
  bool nonConstIndexFound = false;

  //// Setters
  /// Creates VarData for a new VarDecl*.
  void addVar(const clang::VarDecl* VD);
  /// Marks the SourceLocation of E if it is required to store.
  /// E could be DeclRefExpr*, ArraySubscriptExpr* or MemberExpr*.
  void markLocation(const clang::Expr* E);
  /// Sets E's corresponding VarData (or all its child nodes) to
  /// required/not required. For isReq==true, checks if the current mode is
  /// markingMode and nonLinearMode. E could be DeclRefExpr*,
  /// ArraySubscriptExpr* or MemberExpr*.
  void setIsRequired(const clang::Expr* E, bool isReq = true);

  /// Returns the VarsData of the CFG block being visited.
  VarsData& getCurBlockVarsData() { return *blockData[curBlockID]; }

  //// Modes Setters
  /// Sets the mode manually
  void setMode(int mode) { modeStack.push_back(mode); }
  /// Sets nonLinearMode but leaves markingMode just as it was.
  void startNonLinearMode() {
    modeStack.push_back(modeStack.back() | Mode::nonLinearMode);
  }
  /// Sets markingMode but leaves nonLinearMode just as it was.
  void startMarkingMode() {
    modeStack.push_back(Mode::markingMode | modeStack.back());
  }
  /// Removes the last mode in the stack (retrieves the previous one).
  void resetMode() { modeStack.pop_back(); }

public:
  /// Constructor
  TBRAnalyzer(ASTContext& m_Context) : m_Context(m_Context) {
    modeStack.push_back(0);
  }

  /// Destructor
  ~TBRAnalyzer() = default;

  /// Delete copy/move operators and constructors.
  TBRAnalyzer(const TBRAnalyzer&) = delete;
  TBRAnalyzer& operator=(const TBRAnalyzer&) = delete;
  TBRAnalyzer(const TBRAnalyzer&&) = delete;
  TBRAnalyzer& operator=(const TBRAnalyzer&&) = delete;

  /// Returns the result of the whole analysis
  std::set<clang::SourceLocation> getResult() { return TBRLocs; }

  /// Visitors
  void Analyze(const clang::FunctionDecl* FD);

  void VisitCFGBlock(const clang::CFGBlock* block);

  void Visit(const clang::Stmt* stmt) {
    clang::ConstStmtVisitor<TBRAnalyzer, void>::Visit(stmt);
  }

  void VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
  void VisitBinaryOperator(const clang::BinaryOperator* BinOp);
  void VisitCallExpr(const clang::CallExpr* CE);
  void VisitCompoundStmt(const clang::CompoundStmt* CS);
  void VisitConditionalOperator(const clang::ConditionalOperator* CO);
  void VisitCXXConstructExpr(const clang::CXXConstructExpr* CE);
  void VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
  void VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* SCE);
  void VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
  void VisitDeclStmt(const clang::DeclStmt* DS);
  void VisitExprWithCleanups(const clang::ExprWithCleanups* EWC);
  void VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
  void VisitInitListExpr(const clang::InitListExpr* ILE);
  void VisitMemberExpr(const clang::MemberExpr* ME);
  void VisitParenExpr(const clang::ParenExpr* PE);
  void VisitReturnStmt(const clang::ReturnStmt* RS);
  void VisitUnaryOperator(const clang::UnaryOperator* UnOp);

  /// FIXME: Make sure these are not necessary
  /// Unused Visitors:
  // void VisitCXXBoolLiteralExpr(const clang::CXXBoolLiteralExpr* BL);
  // void VisitCXXThisExpr(const clang::CXXThisExpr* TE);
  // void VisitFloatingLiteral(const clang::FloatingLiteral* FL);
  // void VisitIntegerLiteral(const clang::IntegerLiteral* IL);
  // void VisitStmt(const clang::Stmt* S);
};

} // end namespace clad
#endif // CLAD_DIFFERENTIATOR_TBRANALYZER_H
