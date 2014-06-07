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

  ///\brief A pair function, independent variable.
  ///
  class FunctionDeclInfo {
  private:
    FunctionDecl* m_FD;
    ParmVarDecl* m_PVD;
  public:
    FunctionDeclInfo(FunctionDecl* FD, ParmVarDecl* PVD)
      : m_FD(FD), m_PVD(PVD) {
#ifndef NDEBUG
      if (!PVD) // Can happen if there was error deducing the argument
        return;
      bool inArgs = false;
      for (unsigned i = 0; i < FD->getNumParams(); ++i)
        if (PVD == FD->getParamDecl(i)) {
          inArgs = true;
          break;
        }
      assert(inArgs && "Must pass in a param of the FD.");
#endif
    }
    FunctionDecl* getFD() const { return m_FD; }
    ParmVarDecl* getPVD() const { return m_PVD; }
    bool isValid() const { return m_FD && m_PVD; }
    void dump() const {
      if (!isValid())
        llvm::errs() << "<invalid> FD :"<< m_FD << " , PVD:" << m_PVD << "\n";
      else {
        m_FD->dump();
        m_PVD->dump();
      }
    }
  };

  ///\brief The list of the dependent functions which also need differentiation
  /// because they are called by the function we are asked to differentitate.
  ///
  class DiffPlan {
  private:
    typedef llvm::SmallVector<FunctionDeclInfo, 16> Functions;
    Functions m_Functions;
  public:
    typedef Functions::iterator iterator;
    typedef Functions::const_iterator const_iterator;
    void push_back(FunctionDeclInfo FDI) { m_Functions.push_back(FDI); }
    iterator begin() { return m_Functions.begin(); }
    iterator end() { return m_Functions.end(); }
    const_iterator begin() const { return m_Functions.begin(); }
    const_iterator end() const { return m_Functions.end(); }
    size_t size() const { return m_Functions.size(); }
    void dump() {
      for (const_iterator I = begin(), E = end(); I != E; ++I) {
        I->dump();
        llvm::errs() << "\n";
      }
    }
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

    Sema& m_Sema;

    ///\brief Tries to find the independent variable of explicitly diffed
    /// functions.
    ///
    ParmVarDecl* getIndependentArg(Expr* argExpr, FunctionDecl* FD) {
      assert(!m_TopMostFDI && "Must be not in implicit fn collecting mode.");
      bool isIndexOutOfRange = false;
      llvm::APSInt result;
      ASTContext& C = m_Sema.getASTContext();
      DiagnosticsEngine& Diags = m_Sema.Diags;
      if (argExpr->EvaluateAsInt(result, C)) {
        const int64_t argIndex = result.getSExtValue();
        const int64_t argNum = FD->getNumParams();
        //TODO: Implement the argument checks in the DerivativeBuilder
        if (argNum == 0) {
          unsigned DiagID
            = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                    "Trying to differentiate function '%0' taking no arguments");
          Diags.Report(argExpr->getLocStart(), DiagID)
            << FD->getNameAsString();
          return 0;
        }
        //if arg is int but do not exists print error
        else if (argIndex > argNum || argIndex < 1) {
          isIndexOutOfRange = true;
          unsigned DiagID
            = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                    "Invalid argument index %0 among %1 argument(s)");
          Diags.Report(argExpr->getLocStart(), DiagID)
            << (int)argIndex
            << (int)argNum;
          return 0;
        }
        else
          return FD->getParamDecl(argIndex - 1);
      }
      else if (!isIndexOutOfRange){
        unsigned DiagID
          = Diags.getCustomDiagID(DiagnosticsEngine::Error,
                                  "Must be an integral value");
        Diags.Report(argExpr->getLocStart(), DiagID);
        return 0;
      }
      return 0;
    }
  public:
    DiffCollector(DeclGroupRef DGR, DiffPlan& plan, Sema& S)
      : m_DiffPlan(plan), m_TopMostFDI(0), m_Sema(S) {
      if (DGR.isSingleDecl())
        TraverseDecl(DGR.getSingleDecl());
    }

    bool VisitCallExpr(CallExpr* E) {
      if (FunctionDecl *FD = E->getDirectCallee()) {
        if (m_TopMostFDI) {
          unsigned index;
          for (index = 0; index < m_TopMostFDI->getFD()->getNumParams();++index)
            if (FD->getParamDecl(index) == m_TopMostFDI->getPVD())
              break;

            FunctionDeclInfo FDI(FD, FD->getParamDecl(index));
            m_DiffPlan.push_back(FDI);
        }
        // We need to find our 'special' diff annotated such:
        // diff(...) __attribute__((annotate("D")))
        else if (const AnnotateAttr* A = FD->getAttr<AnnotateAttr>()) {
          if (A->getAnnotation().equals("D"))
            if (ImplicitCastExpr* ICE
                = dyn_cast<ImplicitCastExpr>(E->getArg(0))) {
              if (DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
                assert(isa<FunctionDecl>(DRE->getDecl()) && "Must not happen.");

                // We know that is *our* diff function.
                FunctionDecl* cand = cast<FunctionDecl>(DRE->getDecl());
                ParmVarDecl* candPVD = getIndependentArg(E->getArg(1), cand);
                FunctionDeclInfo FDI(cand, candPVD);
                m_TopMostFDI = &FDI;
                TraverseDecl(FD);
                m_TopMostFDI = 0;
                m_DiffPlan.push_back(FDI);
              }
            }
        }
      }
      return true;     // return false to abort visiting.
    }
  };
} // end namespace

namespace clad {
  namespace plugin {
    struct DifferentiationOptions {
      DifferentiationOptions()
        : PrintSourceFn(false), PrintSourceFnAST(false), PrintDerivedFn(false),
          PrintDerivedAST(false) { }

      bool PrintSourceFn : 1;
      bool PrintSourceFnAST : 1;
      bool PrintDerivedFn : 1;
      bool PrintDerivedAST : 1;
    };

    class CladPlugin : public ASTConsumer {
    private:
      clang::CompilerInstance& m_CI;
      DifferentiationOptions m_DO;
      llvm::OwningPtr<DerivativeBuilder> m_DerivativeBuilder;
    public:
      CladPlugin(CompilerInstance& CI, DifferentiationOptions& DO)
        : m_CI(CI), m_DO(DO) { }

      virtual void HandleCXXImplicitFunctionInstantiation (FunctionDecl *D) {

      }

      virtual bool HandleTopLevelDecl(DeclGroupRef DGR) {
        if (!m_DerivativeBuilder)
          m_DerivativeBuilder.reset(new DerivativeBuilder(m_CI.getSema()));

        DiffPlan plan;
        DiffCollector m_Collector(DGR, plan, m_CI.getSema());

        //set up printing policy
        clang::LangOptions LangOpts;
        LangOpts.CPlusPlus = true;
        clang::PrintingPolicy Policy(LangOpts);

        for (DiffPlan::iterator I = plan.begin(), E = plan.end(); I != E; ++I) {
          if (!I->isValid())
            continue;
          // if enabled, print source code of the original functions
          if (m_DO.PrintSourceFn) {
            I->getFD()->print(llvm::outs(), Policy);
          }
          // if enabled, print ASTs of the original functions
          if (m_DO.PrintSourceFnAST) {
            I->getFD()->dumpColor();
          }

          // derive the collected functions
          FunctionDecl* Derivative = Derivative
            = m_DerivativeBuilder->Derive(I->getFD(), I->getPVD());

            // if enabled, print source code of the derived functions
            if (m_DO.PrintDerivedFn) {
              Derivative->print(llvm::outs(), Policy);
            }
            // if enabled, print ASTs of the derived functions
            if (m_DO.PrintDerivedAST) {
              Derivative->dumpColor();
            }
            if (Derivative) {
              m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(Derivative));
              Derivative->getDeclContext()->addDecl(Derivative);
            }

        }
        return true; // Happiness
      }

      virtual void HandleTranslationUnit(clang::ASTContext& C) {
      }
    };

    template<typename ConsumerType>
    class Action : public PluginASTAction {
    private:
      DifferentiationOptions m_DO;
    protected:
      ASTConsumer *CreateASTConsumer(CompilerInstance& CI,
                                     llvm::StringRef InFile) {
        return new ConsumerType(CI, m_DO);
      }

      bool ParseArgs(const CompilerInstance &CI,
                     const std::vector<std::string>& args) {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          if (args[i] == "-fprint-source-fn") {
            m_DO.PrintSourceFn = true;
          }
          else if (args[i] == "-fprint-source-fn-ast") {
            m_DO.PrintSourceFnAST = true;
          }
          else if (args[i] == "-fprint-derived-fn") {
            m_DO.PrintDerivedFn = true;
          }
          else if (args[i] == "-fprint-derived-fn-ast") {
            m_DO.PrintDerivedAST = true;
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
