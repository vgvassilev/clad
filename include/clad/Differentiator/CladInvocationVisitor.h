#ifndef CLAD_DIFFERENTIATOR_CLADINVOCATIONVISITOR_H
#define CLAD_DIFFERENTIATOR_CLADINVOCATIONVISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
class CallExpr;
class CUDAKernelCallExpr;
class Decl;
class DeclGroupRef;
class FunctionDecl;
class Sema;
} // namespace clang

namespace clad {
class DiffScheduler;
struct DiffRequest;
struct Options;

/// Reports a clad entry-point call in the initialiser of a variable the
/// compiler has to work out before clad is handed the declaration, and says
/// what to write instead. clad cannot fix such a call: by the time the
/// declaration reaches this plugin the compiler has already diagnosed the
/// initialiser. See #2188.
void DiagnoseConstantInitRequests(clang::Sema& S,
                                  const DiffScheduler& Scheduler,
                                  clang::DeclGroupRef DGR);

/// Recognizes ordinary Clad entry points and submits root requests.
/// Traverses declarations regardless of pragma state. The scheduler checks
/// each root's location before invoking this visitor's request preparation.
/// \ingroup pipeline
class CladInvocationVisitor
    : public clang::RecursiveASTVisitor<CladInvocationVisitor> {
  clang::Sema& m_Sema;
  DiffScheduler& m_Scheduler;
  const Options& m_Options;
  const clang::FunctionDecl* m_EnclosingFD = nullptr;
  bool m_Traversing = false;

public:
  CladInvocationVisitor(clang::Sema& S, DiffScheduler& Scheduler,
                        const Options& Opts)
      : m_Sema(S), m_Scheduler(Scheduler), m_Options(Opts) {}

  void Walk(clang::DeclGroupRef DGR);
  [[nodiscard]] bool isTraversing() const { return m_Traversing; }
  bool TraverseDecl(clang::Decl* D);
  bool VisitCallExpr(clang::CallExpr* E);
  bool TraverseCUDAKernelCallExpr(clang::CUDAKernelCallExpr* KCE);
  [[nodiscard]] bool shouldVisitImplicitCode() const { return true; }

private:
  bool PrepareRequest(DiffRequest& Request, clang::CallExpr* E,
                      const clang::FunctionDecl* Invocation);
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_CLADINVOCATIONVISITOR_H
