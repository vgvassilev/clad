//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_CLANG_PLUGIN
#define CLAD_CLANG_PLUGIN

#include "clad/Differentiator/Version.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/OwningPtr.h"

namespace clang {
  class ASTContext;
  class CallExpr;
  class CompilerInstance;
  class DeclGroupRef;
  class Expr;
  class FunctionDecl;
  class ParmVarDecl;
  class Sema;
}

namespace clad {
  class DerivativeBuilder;

  ///\brief A pair function, independent variable.
  ///
  class FunctionDeclInfo {
  private:
    clang::FunctionDecl* m_FD;
    clang::ParmVarDecl* m_PVD;
  public:
    FunctionDeclInfo(clang::FunctionDecl* FD, clang::ParmVarDecl* PVD);
    clang::FunctionDecl* getFD() const { return m_FD; }
    clang::ParmVarDecl* getPVD() const { return m_PVD; }
    bool isValid() const { return m_FD && m_PVD; }
    LLVM_DUMP_METHOD void dump() const;
  };

  ///\brief The list of the dependent functions which also need differentiation
  /// because they are called by the function we are asked to differentitate.
  ///
  class DiffPlan {
  private:
    typedef llvm::SmallVector<FunctionDeclInfo, 16> Functions;
    Functions m_Functions;
    clang::CallExpr* m_CallToUpdate;
  public:
    DiffPlan() : m_CallToUpdate(0) { }
    typedef Functions::iterator iterator;
    typedef Functions::const_iterator const_iterator;
    void push_back(FunctionDeclInfo FDI) { m_Functions.push_back(FDI); }
    iterator begin() { return m_Functions.begin(); }
    iterator end() { return m_Functions.end(); }
    const_iterator begin() const { return m_Functions.begin(); }
    const_iterator end() const { return m_Functions.end(); }
    size_t size() const { return m_Functions.size(); }
    void setCallToUpdate(clang::CallExpr* CE) { m_CallToUpdate = CE; }
    void updateCall(clang::FunctionDecl* FD, clang::Sema& SemaRef);
    LLVM_DUMP_METHOD void dump();
  };

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
  private:
    ///\brief The diff step-by-step plan for differentiation.
    ///
    DiffPlan& m_DiffPlan;

    ///\brief If set it means that we need to find the called functions and
    /// add them for implicit diff.
    ///
    FunctionDeclInfo* m_TopMostFDI;

    clang::Sema& m_Sema;

    ///\brief Tries to find the independent variable of explicitly diffed
    /// functions.
    ///
    clang::ParmVarDecl* getIndependentArg(clang::Expr* argExpr,
                                          clang::FunctionDecl* FD);
  public:
    DiffCollector(clang::DeclGroupRef DGR, DiffPlan& plan, clang::Sema& S);
    bool VisitCallExpr(clang::CallExpr* E);
  };
}

namespace clad {
  namespace plugin {
    struct DifferentiationOptions {
      DifferentiationOptions()
        : DumpSourceFn(false), DumpSourceFnAST(false), DumpDerivedFn(false),
          DumpDerivedAST(false), GenerateSourceFile(false) { }

      bool DumpSourceFn : 1;
      bool DumpSourceFnAST : 1;
      bool DumpDerivedFn : 1;
      bool DumpDerivedAST : 1;
      bool GenerateSourceFile : 1;
    };

    class CladPlugin : public clang::ASTConsumer {
    private:
      clang::CompilerInstance& m_CI;
      DifferentiationOptions m_DO;
      llvm::OwningPtr<DerivativeBuilder> m_DerivativeBuilder;
    public:
      CladPlugin(clang::CompilerInstance& CI, DifferentiationOptions& DO);
      ~CladPlugin();

      virtual void Initialize(clang::ASTContext& Context);
      virtual bool HandleTopLevelDecl(clang::DeclGroupRef DGR);
    };

    template<typename ConsumerType>
    class Action : public clang::PluginASTAction {
    private:
      DifferentiationOptions m_DO;
    protected:
      clang::ASTConsumer *CreateASTConsumer(clang::CompilerInstance& CI,
                                            llvm::StringRef InFile) {
        return new ConsumerType(CI, m_DO);
      }

      bool ParseArgs(const clang::CompilerInstance &CI,
                     const std::vector<std::string>& args) {
        if (clang::getClangRevision() != clad::getClangCompatRevision()) {
          // TODO: Print nice looking diagnostics through the DiagEngine.
          llvm::errs() << "Clang is not compatible with clad."
                       << " (" << clang::getClangRevision() << " != "
                       << clad::getClangCompatRevision() << " )\n";
          return false;
        }
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          if (args[i] == "-fdump-source-fn") {
            m_DO.DumpSourceFn = true;
          }
          else if (args[i] == "-fdump-source-fn-ast") {
            m_DO.DumpSourceFnAST = true;
          }
          else if (args[i] == "-fdump-derived-fn") {
            m_DO.DumpDerivedFn = true;
          }
          else if (args[i] == "-fdump-derived-fn-ast") {
            m_DO.DumpDerivedAST = true;
          }
          else if (args[i] == "-fgenerate-source-file") {
            m_DO.GenerateSourceFile = true;
          }
          else if (args[i] == "-help") {
            // Print some help info.
            llvm::errs() <<
              "Option set for the clang-based automatic differentiator - clad:\n\n" <<
              "-fdump-source-fn - Prints out the source code of the function.\n" <<
              "-fdump-source-fn-ast - Prints out the AST of the function.\n" <<
              "-fdump-derived-fn - Prints out the source code of the derivative.\n" <<
              "-fdump-derived-fn-ast - Prints out the AST of the derivative.\n" <<
              "-fgenerate-source-file - Produces a file containing the derivatives.\n";

            llvm::errs() << "-help - Prints out this screen.\n\n";
          }
          else {
            llvm::errs() << "clad: Error: invalid option "
                         << args[i] << "\n";
            return false; // Tells clang not to create the plugin.
          }
        }
        return true;
      }
    };
  } // end namespace plugin
} // end namespace clad

#endif // CLAD_CLANG_PLUGIN
