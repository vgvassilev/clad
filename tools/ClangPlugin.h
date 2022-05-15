//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_CLANG_PLUGIN
#define CLAD_CLANG_PLUGIN

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/Version.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

namespace clang {
  class ASTContext;
  class CallExpr;
  class CompilerInstance;
  class DeclGroupRef;
  class Expr;
  class FunctionDecl;
  class ParmVarDecl;
  class Sema;
} // namespace clang

namespace clad {
  namespace plugin {
    struct DifferentiationOptions {
      DifferentiationOptions()
          : DumpSourceFn(false), DumpSourceFnAST(false), DumpDerivedFn(false),
            DumpDerivedAST(false), GenerateSourceFile(false),
            ValidateClangVersion(false), CustomEstimationModel(false),
            PrintNumDiffErrorInfo(false), CustomModelName("") {}

      bool DumpSourceFn : 1;
      bool DumpSourceFnAST : 1;
      bool DumpDerivedFn : 1;
      bool DumpDerivedAST : 1;
      bool GenerateSourceFile : 1;
      bool ValidateClangVersion : 1;
      bool CustomEstimationModel : 1;
      bool PrintNumDiffErrorInfo : 1;
      std::string CustomModelName;
    };

    class CladPlugin : public clang::ASTConsumer {
      clang::CompilerInstance& m_CI;
      DifferentiationOptions m_DO;
      std::unique_ptr<DerivativeBuilder> m_DerivativeBuilder;
      bool m_HasRuntime = false;
      bool m_PendingInstantiationsInFlight = false;
      bool m_HandleTopLevelDeclInternal = false;
    public:
      CladPlugin(clang::CompilerInstance& CI, DifferentiationOptions& DO);
      ~CladPlugin();
      bool HandleTopLevelDecl(clang::DeclGroupRef DGR) override;
      clang::FunctionDecl* ProcessDiffRequest(DiffRequest& request);
      void DumpRequestedInfo(const clang::FunctionDecl* sourceFn,
                             const clang::FunctionDecl* derivedFn);
      void ProcessTopLevelDecl(clang::Decl* D);
    private:
      bool CheckBuiltins();
      void InitializeDerivativeBuilder();
    };

    clang::FunctionDecl* ProcessDiffRequest(CladPlugin& P,
                                            DiffRequest& request) {
      return P.ProcessDiffRequest(request);
    }

    template <typename ConsumerType>
    class Action : public clang::PluginASTAction {
    private:
      DifferentiationOptions m_DO;

    protected:
      std::unique_ptr<clang::ASTConsumer>
      CreateASTConsumer(clang::CompilerInstance& CI, llvm::StringRef InFile) {
        return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI, m_DO));
      }

      static bool IsRunningOnExpectedClangVersion() {
        // FIXME: The check does more damage than good. We need to make it much
        // more sophisticated to work as expected. For example, clang can be
        // checked out from svn or git; the compatible revision can be a range;
        // clang itself can have local patches on top of the compatible version.
        if (clang::getClangRevision() != "" &&
            clang::getClangRevision() != clad::getClangCompatRevision()) {
          // TODO: Print nice looking diagnostics through the DiagEngine.
          llvm::errs() << "Clang is not compatible with clad."
                       << " (" << clang::getClangRevision()
                       << " != " << clad::getClangCompatRevision() << " )\n";
          return false;
        }
        return true;
      }

      bool ParseArgs(const clang::CompilerInstance& CI,
                     const std::vector<std::string>& args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          if (args[i] == "-fdump-source-fn") {
            m_DO.DumpSourceFn = true;
          } else if (args[i] == "-fdump-source-fn-ast") {
            m_DO.DumpSourceFnAST = true;
          } else if (args[i] == "-fdump-derived-fn") {
            m_DO.DumpDerivedFn = true;
          } else if (args[i] == "-fdump-derived-fn-ast") {
            m_DO.DumpDerivedAST = true;
          } else if (args[i] == "-fgenerate-source-file") {
            m_DO.GenerateSourceFile = true;
          } else if (args[i] == "-fvalidate-clang-version") {
            m_DO.ValidateClangVersion = true;
            if (!IsRunningOnExpectedClangVersion())
              return false; // Tells clang not to create the plugin.
          } else if (args[i] == "-fcustom-estimation-model") {
            m_DO.CustomEstimationModel = true;
            if (++i == e) {
              llvm::errs() << "No shared object was specified.";
              return false;
            }
            m_DO.CustomModelName = args[i];
          } else if (args[i] == "-fprint-num-diff-errors") {
            m_DO.PrintNumDiffErrorInfo = true;
          } else if (args[i] == "-help") {
            // Print some help info.
            llvm::errs()
                << "Option set for the clang-based automatic differentiator - "
                   "clad:\n\n"
                << "-fdump-source-fn - Prints out the source code of the "
                   "function.\n"
                << "-fdump-source-fn-ast - Prints out the AST of the "
                   "function.\n"
                << "-fdump-derived-fn - Prints out the source code of the "
                   "derivative.\n"
                << "-fdump-derived-fn-ast - Prints out the AST of the "
                   "derivative.\n"
                << "-fgenerate-source-file - Produces a file containing the "
                   "derivatives.\n"
                << "-fcustom-estimation-model - allows user to send in a "
                   "shared object to use as the custom estimation model.\n"
                << "-fprint-num-diff-errors - allows users to print the "
                   "calculated numerical diff errors, this flag is overriden "
                   "by -DCLAD_NO_NUM_DIFF.\n";

            llvm::errs() << "-help - Prints out this screen.\n\n";
          } else {
            llvm::errs() << "clad: Error: invalid option " << args[i] << "\n";
            return false; // Tells clang not to create the plugin.
          }
        }
        return true;
      }

      PluginASTAction::ActionType getActionType() override {
        return AddBeforeMainAction;
      }
    };
  } // end namespace plugin
} // end namespace clad

#endif // CLAD_CLANG_PLUGIN
