#ifndef CLAD_DIFFERENTIATOR_TBRANALYZER_H
#define CLAD_DIFFERENTIATOR_TBRANALYZER_H

#include "clang/AST/StmtVisitor.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include <map>
#include <unordered_map>

using namespace clang;

namespace clad {

class TBRAnalyzer : public clang::ConstStmtVisitor<TBRAnalyzer> {
private:
  /// Used to provide a hash function for an unordered_map with llvm::APInt
  /// type keys.
  struct APIntHash {
    size_t operator()(const llvm::APInt& apint) const {
      return llvm::hash_value(apint);
    }
  };

  /// Just a helper struct serving as a wrapper for IdxOrMemberValue union.
  /// Used to unwrap expressions like a[6].x.t[3]. Only used in
  /// TBRAnalyzer::overlay().
  struct IdxOrMember {
    enum IdxOrMemberType { FIELD, INDEX };
    union IdxOrMemberValue {
      const clang::FieldDecl* field;
      llvm::APInt index;
      IdxOrMemberValue() : field(nullptr) {}
      ~IdxOrMemberValue() {}
      IdxOrMemberValue(const IdxOrMemberValue&) = delete;
      IdxOrMemberValue& operator=(const IdxOrMemberValue&) = delete;
      IdxOrMemberValue(const IdxOrMemberValue&&) = delete;
      IdxOrMemberValue& operator=(const IdxOrMemberValue&&) = delete;
    };
    IdxOrMemberType type;
    IdxOrMemberValue val;
    IdxOrMember(const clang::FieldDecl* field) : type(IdxOrMemberType::FIELD) {
      val.field = field;
    }
    IdxOrMember(llvm::APInt&& index) : type(IdxOrMemberType::INDEX) {
      new (&val.index) llvm::APInt(index);
    }
    IdxOrMember(const IdxOrMember& other) {
      new (&val.index) llvm::APInt();
      *this = other;
    }
    IdxOrMember(const IdxOrMember&& other) noexcept {
      new (&val.index) llvm::APInt();
      *this = other;
    }
    IdxOrMember& operator=(const IdxOrMember& other) {
      type = other.type;
      if (type == IdxOrMemberType::FIELD)
        val.field = other.val.field;
      else
        val.index = other.val.index;
      return *this;
    }
    IdxOrMember& operator=(const IdxOrMember&& other) noexcept {
      return *this = other;
    }
    ~IdxOrMember() = default;
  };

  /// Stores all the necessary information about one variable. Fundamental type
  /// variables need only one bool. An object/array needs a separate VarData for
  /// every its field/element. Reference type variables have their own type for
  /// convenience reasons and just point to the corresponding VarData.
  /// UNDEFINED is used whenever the type of a node cannot be determined.

  /// FIXME: Pointers to objects are considered OBJ_TYPE for simplicity. This
  /// approach might cause problems when the support for pointers is added.

  /// FIXME: Only References to concrete variables are supported. Using
  /// 'double& x = (cond ? a : b);' or 'double& x = arr[n*k]' (non-const index)
  /// will lead to unpredictable behavior.

  /// FIXME: Different array elements are considered different variables
  /// which makes the analysis way more complicated (both in terms of
  /// readability and performance) when non-constant indices are used. Moreover,
  /// in such cases we start assuming 'arr[k]' could be any element of arr.
  /// The only scenario where analysing elements separately would make sense is
  /// when an element with the same constant index is changed multiple times in
  /// a row, which seems uncommon. It's worth considering analysing arrays as
  /// whole structures instead (just one VarData for the whole array).

  struct VarData;
  using ObjMap = std::unordered_map<const clang::FieldDecl*, VarData*>;
  using ArrMap = std::unordered_map<const llvm::APInt, VarData*, APIntHash>;

  struct VarData {
    enum VarDataType { UNDEFINED, FUND_TYPE, OBJ_TYPE, ARR_TYPE, REF_TYPE };
    union VarDataValue {
      bool fundData;
      /// objData, arrData are stored as pointers for VarDataValue to take
      /// less space.
      ObjMap* objData;
      ArrMap* arrData;
      VarData* refData;
      VarDataValue() : fundData(false) {}
    };
    VarDataType type;
    VarDataValue val;
    bool isReferenced = false;

    VarData() = default;
    VarData(const VarData&) = delete;
    VarData& operator=(const VarData&) = delete;
    VarData(const VarData&&) = delete;
    VarData& operator=(const VarData&&) = delete;

    ~VarData() {
      if (type == OBJ_TYPE) {
        for (auto& pair : *val.objData) {
          delete pair.second;
        }
      } else if (type == ARR_TYPE) {
        for (auto& pair : *val.arrData) {
          delete pair.second;
        }
      }
    }
    /// Recursively sets all the leaves' bools to isReq.
    void setIsRequired(bool isReq = true);
    /// Returns true if there is at least one required to store node among
    /// child nodes.
    bool findReq() const;
    /// Whenever an array element with a non-constant index is set to required
    /// this function is used to set to required all the array elements that
    /// could match that element (e.g. set 'a[1].y' and 'a[6].y' to required
    /// when 'a[k].y' is set to required). Takes unwrapped sequence of
    /// indices/members of the expression being overlaid and the index of of the
    /// current index/member.
    void overlay(llvm::SmallVector<IdxOrMember, 2>& IdxAndMemberSequence,
                 size_t i);
    /// Used to merge together VarData for one variable from two branches
    /// (e.g. after an if-else statements). Look at the Control Flow section for
    /// more information.
    void merge(VarData* mergeData);
    /// Used to recursively copy VarData when separating into different branches
    /// (e.g. when entering an if-else statements). Look at the Control Flow
    /// section for more information. refVars stores copied nodes for
    /// corresponding original nodes in case those are referenced (a referenced
    /// node is a child to multiple nodes, therefore, we need to make sure we
    /// don't make multiple copies of it).
    VarData* copy();
    VarData* copy(std::unordered_map<VarData*, VarData*>& refVars);
    void restoreRefs(std::unordered_map<VarData*, VarData*>& refVars);
  };

  /// Given a MemberExpr*/ArraySubscriptExpr* return a pointer to its
  /// corresponding VarData. If the given element of an array does not have a
  /// VarData* yet it will be added automatically. If addNonConstIdx==false this
  /// will return the last VarData* before the non-constant index
  /// (e.g. for 'x.arr[k+1].y' the return value will be the VarData* of x.arr).
  /// Otherwise, non-const indices will be represented as index -1.
  VarData* getMemberVarData(const clang::MemberExpr* ME,
                            bool addNonConstIdx = false);
  VarData* getArrSubVarData(const clang::ArraySubscriptExpr* ASE,
                            bool addNonConstIdx = false);
  /// Given an Expr* returns its corresponding VarData.
  VarData* getExprVarData(const clang::Expr* E, bool addNonConstIdx = false);
  /// Adds the field FD to objData.
  void addField(std::unordered_map<const clang::FieldDecl*, VarData*>* objData,
                const FieldDecl* FD);
  /// Whenever an array element with a non-constant index is set to required
  /// this function is used to set to required all the array elements that
  /// could match that element (e.g. set 'a[1].y' and 'a[6].y' to required when
  /// 'a[k].y' is set to required). Unwraps the a given expression into a
  /// sequence of indices/members of the expression being overlaid and calls
  /// VarData::overlay() recursively.
  void overlay(const clang::Expr* E);

  /// Used to store all the necessary information about variables at a
  /// particular moment.
  /// Note: 'this' pointer does not have a declaration so nullptr is used as
  /// its key instead.
  using VarsData = std::unordered_map<const clang::VarDecl*, VarData*>;
  /// Used to find DeclRefExpr's that will be used in the backwards pass.
  /// In order to be marked as required, a variables has to appear in a place
  /// where it would have a differential influence and will appear non-linearly
  /// (e.g. for 'x = 2 * y;', y will not appear in the backwards pass). Hence,
  /// markingMode and nonLinearMode.
  enum Mode { markingMode = 1, nonLinearMode = 2 };
  /// Tells if the variable at a given location is required to store. Basically,
  /// is the result of analysis.
  std::map<clang::SourceLocation, bool> TBRLocs;
  /// Stores VarsData for every branch in control flow (e.g. if-else statements,
  /// loops).
  std::vector<std::vector<VarsData>> reqStack;
  /// Stores modes in a stack (used to retrieve the old mode after entering
  /// a new one).
  std::vector<int> modeStack;
  /// Stores local variables to delete them after exiting the corresponding
  /// scope.
  /// Note: This is not used every time a new scope is entered. This is only
  /// used when merging an if-else statement to get rid of local variables in
  /// the then-branch.
  std::vector<std::vector<const VarDecl*>> localVarsStack;

  ASTContext* m_Context;

  /// The index of the innermost branch corresponding to a loop (used to handle
  /// break/continue statements).
  size_t innermostLoopLayer = 0;
  /// Tells if the current branch should be deleted instead of merged with
  /// others. This happens when the branch has a break/continue statement or a
  /// return expression in it.
  bool deleteCurBranch = false;
  /// Loop bodies have to be passed twice. This tells us what pass is currently
  /// happening.
  bool firstLoopPass = false;
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

  //// Control Flow
  /// Returns the current branch.
  VarsData& getCurBranch() { return reqStack.back().back(); }
  /// Adds a new layer.
  void addLayer() { reqStack.emplace_back(); }
  /// Creates a new empty branch.
  void addBranch() { reqStack.back().emplace_back(); }
  /// Deletes the last branch.
  void deleteBranch() {
    for (auto& pair : getCurBranch())
      delete pair.second;
    reqStack.back().pop_back();
  }
  /// Merges the last layer into the one last branch on the previous layer
  /// right and deletes the last layer.
  void mergeLayer();
  /// Merges the last layer but, unlike the previous method, basically replaces
  /// the last branch on the previous layer with the result of merging. After
  /// that, removes the last layer.
  void mergeLayerOnTop();
  /// Merges the branch with index targetBranch into a sourceBranchNum.
  /// No branches are deleted.
  void mergeBranchTo(size_t sourceBranchNum, VarsData& targetBranch);
  /// Removes local variables from the current branch (uses localVarsStack).
  /// This is necessary when merging if-else branches.
  void removeLocalVars();

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
  TBRAnalyzer(ASTContext* m_Context) : m_Context(m_Context) {
    modeStack.push_back(0);
    addLayer();
    addBranch();
  }

  /// Destructor
  ~TBRAnalyzer() {
    for (auto& layer : reqStack)
      for (auto& branch : layer)
        for (auto& pair : branch)
          delete pair.second;
  }

  /// Delete copy/move operators and constructors.
  TBRAnalyzer(const TBRAnalyzer&) = delete;
  TBRAnalyzer& operator=(const TBRAnalyzer&) = delete;
  TBRAnalyzer(const TBRAnalyzer&&) = delete;
  TBRAnalyzer& operator=(const TBRAnalyzer&&) = delete;

  /// Returns the result of the whole analysis
  std::map<clang::SourceLocation, bool> getResult() { return TBRLocs; }

  /// Visitors
  void Analyze(const clang::FunctionDecl* FD);

  void Visit(const clang::Stmt* stmt) {
    clang::ConstStmtVisitor<TBRAnalyzer, void>::Visit(stmt);
  }

  void VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE);
  void VisitBinaryOperator(const clang::BinaryOperator* BinOp);
  void VisitBreakStmt(const clang::BreakStmt* BS);
  void VisitCallExpr(const clang::CallExpr* CE);
  void VisitCompoundStmt(const clang::CompoundStmt* CS);
  void VisitConditionalOperator(const clang::ConditionalOperator* CO);
  void VisitContinueStmt(const clang::ContinueStmt* CS);
  void VisitCXXConstructExpr(const clang::CXXConstructExpr* CE);
  void VisitCXXDefaultArgExpr(const clang::CXXDefaultArgExpr* DE);
  void VisitCXXStaticCastExpr(const clang::CXXStaticCastExpr* SCE);
  void VisitDeclRefExpr(const clang::DeclRefExpr* DRE);
  void VisitDeclStmt(const clang::DeclStmt* DS);
  void VisitDoStmt(const clang::DoStmt* DS);
  void VisitExprWithCleanups(const clang::ExprWithCleanups* EWC);
  void VisitForStmt(const clang::ForStmt* FS);
  void VisitIfStmt(const clang::IfStmt* If);
  void VisitImplicitCastExpr(const clang::ImplicitCastExpr* ICE);
  void VisitInitListExpr(const clang::InitListExpr* ILE);
  void VisitMemberExpr(const clang::MemberExpr* ME);
  void VisitParenExpr(const clang::ParenExpr* PE);
  void VisitReturnStmt(const clang::ReturnStmt* RS);
  void VisitUnaryOperator(const clang::UnaryOperator* UnOp);
  void VisitWhileStmt(const clang::WhileStmt* WS);

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
