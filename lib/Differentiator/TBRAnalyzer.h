#ifndef CLAD_DIFFERENTIATOR_TBRANALYZER_H
#define CLAD_DIFFERENTIATOR_TBRANALYZER_H

#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"

#include "llvm/ADT/ArrayRef.h"

#include "AnalysisBase.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffPlanner.h"

#include <map>
#include <set>
#include <unordered_map>

namespace clad {

/// Gradient computation requres reversal of the control flow of the original
/// program becomes necessary. To guarantee correctness, certain values that are
/// computed and overwritten in the original program must be made available in
/// the adjoint program. They can be determined by performing a static data flow
/// analysis, the so-called To-Be-Recorded (TBR) analysis. Overestimation of
/// this set must be kept minimal to get efficient adjoint codes.
///
/// This class implements this to-be-recorded analysis.
class TBRAnalyzer : public clang::RecursiveASTVisitor<TBRAnalyzer>,
                    public AnalysisBase {
  /// Used to find DeclRefExpr's that will be used in the backwards pass.
  /// In order to be marked as required, a variables has to appear in a place
  /// where it would have a differential influence and will appear non-linearly
  /// (e.g. for 'x = 2 * y;', y will not appear in the backwards pass). Hence,
  /// markingMode and nonLinearMode.
  enum Mode { kMarkingMode = 1, kNonLinearMode = 2 };
  /// Tells if the variable at a given location is required to store. Basically,
  /// is the result of analysis.
  std::set<const clang::Stmt*>& m_TBRLocs;
  ParamInfo* m_ModifiedParams;
  ParamInfo* m_UsedParams;

  /// Stores modes in a stack (used to retrieve the old mode after entering
  /// a new one).
  std::vector<int> m_ModeStack;

  /// Stores the number of performed passes for a given CFG block index.
  std::vector<short> m_BlockPassCounter;

  //// Setters
  /// Marks S if it is required to store.
  /// E could be DeclRefExpr, ArraySubscriptExpr, MemberExpr, or DeclStmt.
  void markLocation(const clang::Stmt* S);
  /// Sets E's corresponding VarData (or all its child nodes) to
  /// required/not required. For isReq==true, checks if the current mode is
  /// markingMode and nonLinearMode. E could be DeclRefExpr,
  /// ArraySubscriptExpr or MemberExpr.
  void setIsRequired(const clang::Expr* E, bool isReq = true);

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
  TBRAnalyzer(clang::AnalysisDeclContext* AnalysisDC,
              std::set<const clang::Stmt*>& Locs,
              ParamInfo* ModifiedParams = nullptr,
              ParamInfo* UsedParams = nullptr)
      : AnalysisBase(AnalysisDC), m_TBRLocs(Locs),
        m_ModifiedParams(ModifiedParams), m_UsedParams(UsedParams) {
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
  void Analyze(const DiffRequest& request);

  void VisitCFGBlock(const clang::CFGBlock& block);

  bool TraverseArraySubscriptExpr(clang::ArraySubscriptExpr* ASE);
  bool TraverseBinaryOperator(clang::BinaryOperator* BinOp);
  bool TraverseCallExpr(clang::CallExpr* CE);
  bool TraverseConditionalOperator(clang::ConditionalOperator* CO);
  bool TraverseCompoundAssignOperator(clang::CompoundAssignOperator* BinOp);
  bool TraverseCXXConstructExpr(clang::CXXConstructExpr* CE);
  bool TraverseCXXThisExpr(clang::CXXThisExpr* TE);
  bool TraverseCXXMemberCallExpr(clang::CXXMemberCallExpr* CE);
  bool TraverseCXXOperatorCallExpr(clang::CXXOperatorCallExpr* CE);
  bool TraverseDeclRefExpr(clang::DeclRefExpr* DRE);
  bool TraverseDeclStmt(clang::DeclStmt* DS);
  bool TraverseInitListExpr(clang::InitListExpr* ILE);
  bool TraverseMemberExpr(clang::MemberExpr* ME);
  bool TraverseReturnStmt(clang::ReturnStmt* RS);
  bool TraverseUnaryOperator(clang::UnaryOperator* UnOp);
};

} // end namespace clad
#endif // CLAD_DIFFERENTIATOR_TBRANALYZER_H
