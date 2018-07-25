//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "ClangPlugin.h"

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"

#include "clad/Differentiator/Version.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Lookup.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"

using namespace clang;

namespace {
  class SimpleTimer {
    bool WantTiming;
    llvm::TimeRecord Start;
    std::string Output;

  public:
    explicit SimpleTimer(bool WantTiming) : WantTiming(WantTiming) {
      if (WantTiming)
        Start = llvm::TimeRecord::getCurrentTime();
    }

    void setOutput(const Twine &Output) {
      if (WantTiming)
        this->Output = Output.str();
    }

    ~SimpleTimer() {
      if (WantTiming) {
        llvm::TimeRecord Elapsed = llvm::TimeRecord::getCurrentTime();
        Elapsed -= Start;
        llvm::errs() << Output << ": user | system | process | all :";
        Elapsed.print(Elapsed, llvm::errs());
        llvm::errs() << '\n';
      }
    }
  };
}


namespace clad {
  namespace plugin {
    CladPlugin::CladPlugin(CompilerInstance& CI, DifferentiationOptions& DO)
      : m_CI(CI), m_DO(DO), m_HasRuntime(false) { }
    CladPlugin::~CladPlugin() {}

    static bool HasRuntime(Sema& SemaR) {
      ASTContext& C = SemaR.getASTContext();
      DeclarationName Name = &C.Idents.get("custom_derivatives");
      LookupResult R(SemaR, Name, SourceLocation(), Sema::LookupNamespaceName,
                     Sema::ForRedeclaration);
      SemaR.LookupQualifiedName(R, C.getTranslationUnitDecl(),
                                /*allowBuiltinCreation*/ false);
      return !R.empty();
    }

    bool CladPlugin::HandleTopLevelDecl(DeclGroupRef DGR) {
      m_HasRuntime |= HasRuntime(m_CI.getSema());
      // If we have not included "clad/Differentiator/Differentiator.h" exit.
      if (!m_HasRuntime)
        return true;

      if (!m_DerivativeBuilder)
        m_DerivativeBuilder.reset(new DerivativeBuilder(m_CI.getSema()));

      // Instantiate all pending for instantiations templates, because we will
      // need the full bodies to produce derivatives.
      m_CI.getSema().PerformPendingInstantiations();

      DiffPlans plans;
      DiffCollector collector(DGR, plans, m_CI.getSema());

      //set up printing policy
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      for (DiffPlans::iterator plan = plans.begin(), planE = plans.end();
           plan != planE; ++plan)
         for (DiffPlan::iterator I = plan->begin(); I != plan->end(); ++I) {
            if (!I->isValidInMode(plan->getMode()))
                // Some error happened, ignore this plan.
                continue;
            // if enabled, print source code of the original functions
            if (m_DO.DumpSourceFn) {
               I->getFD()->print(llvm::outs(), Policy);
            }
            // if enabled, print ASTs of the original functions
            if (m_DO.DumpSourceFnAST) {
               I->getFD()->dumpColor();
            }

            FunctionDecl* Result = nullptr;
            {
              // FIXME: Move the timing inside the DerivativeBuilder. This would
              // require to pass in the DifferentiationOptions in the DiffPlan.
              // derive the collected functions
              bool WantTiming = getenv("LIBCLAD_TIMING");
              SimpleTimer Timer(WantTiming);
              Timer.setOutput("Generation time for "
                              + plan->begin()->getFD()->getNameAsString());

              Result = m_DerivativeBuilder->Derive(*I, *plan);
            }
            collector.UpdatePlan(Result, &*plan);
            if (I + 1 == plan->end()) // The last element
               plan->updateCall(Result, m_CI.getSema());

            // if enabled, print source code of the derived functions
            if (m_DO.DumpDerivedFn) {
               Result->print(llvm::outs(), Policy);
            }
            // if enabled, print ASTs of the derived functions
            if (m_DO.DumpDerivedAST) {
               Result->dumpColor();
            }
            // if enabled, print the derivatives in a file.
            if (m_DO.GenerateSourceFile) {
               std::error_code err;
               llvm::raw_fd_ostream f("Derivatives.cpp", err,
                                      llvm::sys::fs::F_Append);
               Result->print(f, Policy);
               f.flush();
            }
            if (Result) {
              // Call CodeGen only if the produced decl is a top-most decl.
              if (Result->getDeclContext()
                  == m_CI.getASTContext().getTranslationUnitDecl())
                m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(Result));
            }
      }
      return true; // Happiness
    }
  } // end namespace plugin
} // end namespace clad

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad","Produces derivatives or arbitrary functions");
