//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/DerivativeBuilder.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Sema/Sema.h"

using namespace clang;

namespace {
  
  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
    llvm::SmallVector<CallExpr*, 4> m_DiffCalls;
  public:
    llvm::SmallVector<CallExpr*, 4>& getDiffCalls() {
      return m_DiffCalls;
    }
    
    void collectFunctionsToDiff(DeclGroupRef DGR) {
      if (DGR.isSingleDecl())
        TraverseDecl(DGR.getSingleDecl());
    }
    
    bool VisitCallExpr(CallExpr* E) {
      if (const FunctionDecl *FD = E->getDirectCallee()) {
        // Note here that best would be to annotate with, eg:
        //  __attribute__((annotate("This is our diff that must differentiate"))) {
        // However, GCC doesn't support the annotate attribute on a function
        // definition and clang's version on MacOS chokes up (with clang's trunk
        // everything seems ok 03.06.2013)
        
        //        if (const AnnotateAttr* A = FD->getAttr<AnnotateAttr>())
        if (FD->getNameAsString() == "diff")
          // We know that is *our* diff function.
          m_DiffCalls.push_back(E);
      }
      return true;     // return false to abort visiting.
    }
  };
} // end namespace

namespace clad {
  namespace plugin {
    
    bool fPrintSourceFn = false,  fPrintSourceAst = false,
    fPrintDerivedFn = false, fPrintDerivedAst = false;
    // index of current function to derive in functionsToDerive
    size_t lastIndex = 0;

    class CladPlugin : public ASTConsumer {
    private:
      DiffCollector m_Collector;
      clang::CompilerInstance& m_CI;
      llvm::OwningPtr<DerivativeBuilder> m_DerivativeBuilder;
      clang::FunctionDecl* m_CurDerivative;
    public:
      CladPlugin(CompilerInstance& CI) : m_CI(CI), m_CurDerivative(0) { }

      virtual void HandleCXXImplicitFunctionInstantiation (FunctionDecl *D) {
        
      }

      virtual bool HandleTopLevelDecl(DeclGroupRef DGR) {
        if (!m_DerivativeBuilder)
          m_DerivativeBuilder.reset(new DerivativeBuilder(m_CI.getSema()));

        // Check if this is called recursively and if it is abort, because we
        // don't want CodeGen to see that decl twice.
        if (m_CurDerivative == *DGR.begin())
          return false;

        m_Collector.collectFunctionsToDiff(DGR);
        llvm::SmallVector<CallExpr*, 4>& diffCallExprs
          = m_Collector.getDiffCalls();
        
        //set up printing policy
        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);
        
        for (size_t i = lastIndex, e = diffCallExprs.size(); i < e; ++i) {
          FunctionDecl* Derivative = 0;
          if (ImplicitCastExpr* ICE
              = dyn_cast<ImplicitCastExpr>(diffCallExprs[i]->getArg(0))) {
            if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
              assert(isa<FunctionDecl>(DRE->getDecl()) && "Must not happen.");
              FunctionDecl* functionToDerive
                = cast<FunctionDecl>(DRE->getDecl());

              // if enabled, print source code of the original functions
              if (fPrintSourceFn) {
                functionToDerive->print(llvm::outs(), Policy);
              }
              // if enabled, print ASTs of the original functions
              if (fPrintSourceAst) {
                functionToDerive->dumpColor();
              }

              ValueDecl* argVar = 0;
              //const MaterializeTemporaryExpr* MTE;
              bool isIndexOutOfRange = false;
              llvm::APSInt result;
              ASTContext& C = m_CI.getASTContext();
              if (diffCallExprs[i]->getArg(1)->EvaluateAsInt(result, C)) {
                const int64_t argIndex = result.getSExtValue();
                const int64_t argNum = functionToDerive->getNumParams();
                //TODO: Implement the argument checks in the DerivativeBuilder
                if (argNum == 0) {
                  DiagnosticsEngine& Diags = m_CI.getSema().Diags;
                  unsigned DiagID
                    = Diags.getCustomDiagID(DiagnosticsEngine::Warning,
                                             "Trying to differentiate function %0 taking no arguments");
                  Diags.Report(diffCallExprs[i]->getLocStart(), DiagID)
                    << functionToDerive->getNameAsString();
                }
                //if arg is int but do not exists print error
                else if (argIndex > argNum || argIndex < 1) {
                  isIndexOutOfRange = true;
                  DiagnosticsEngine& Diags = m_CI.getSema().Diags;
                  unsigned DiagID
                    = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                            "Invalid argument index %0 among %1 argument(s)");
                  Diags.Report(diffCallExprs[i]->getArg(1)->getLocStart(), DiagID)
                    << (int)argIndex
                    << (int)argNum;
                }
                else
                  argVar = functionToDerive->getParamDecl(argIndex - 1);
              }
              else if (!isIndexOutOfRange){
                DiagnosticsEngine& Diags = m_CI.getSema().Diags;
                unsigned DiagID
                  = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                          "Must be an integral value");
                Diags.Report(diffCallExprs[i]->getArg(1)->getLocStart(), DiagID);
              }

              if (argVar) {
                // derive the collected functions
                Derivative
                  = m_DerivativeBuilder->Derive(functionToDerive, argVar);

                // if enabled, print source code of the derived functions
                if (fPrintDerivedFn) {
                  Derivative->print(llvm::outs(), Policy);
                }
                // if enabled, print ASTs of the derived functions
                if (fPrintDerivedAst) {
                  Derivative->dumpColor();
                }
              }
            }
            lastIndex = i + 1;
            if (Derivative) {
              m_CurDerivative = Derivative;
              m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(Derivative));
              m_CurDerivative = 0;
            }
          }
        }
        return true; // Happiness
      }

      virtual void HandleTranslationUnit(clang::ASTContext& C) {
      }
    };

    template<typename ConsumerType>
    class Action : public PluginASTAction {
    protected:
      ASTConsumer *CreateASTConsumer(CompilerInstance& CI,
                                     llvm::StringRef InFile) {
        return new ConsumerType(CI);
      }
      
      bool ParseArgs(const CompilerInstance &CI,
                     const std::vector<std::string>& args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          
          if (args[i] == "-fprint-source-fn") {
            fPrintSourceFn = true;
          }
          else if (args[i] == "-fprint-source-fn-ast") {
            fPrintSourceAst = true;
          }
          else if (args[i] == "-fprint-derived-fn") {
            fPrintDerivedFn = true;
          }
          else if (args[i] == "-fprint-derived-fn-ast") {
            fPrintDerivedAst = true;
          }
          else {
            llvm::outs() << "clad: Error: invalid option "
            << args[i] << "\n";
          }
        }
        
        return true;
      }
    };
  } // end namespace plugin
} // end namespace clad

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad","Produces derivatives or arbitrary functions");
