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
#include "clang/Lex/LexDiagnostic.h"
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

    bool CladPlugin::HandleTopLevelDecl(DeclGroupRef DGR) {
      if (!ShouldProcessDecl(DGR))
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

            FunctionDecl* DerivativeDecl = nullptr;
            Decl* DerivativeDeclContext = nullptr;
            {
              // FIXME: Move the timing inside the DerivativeBuilder. This would
              // require to pass in the DifferentiationOptions in the DiffPlan.
              // derive the collected functions
              bool WantTiming = getenv("LIBCLAD_TIMING");
              SimpleTimer Timer(WantTiming);
              Timer.setOutput("Generation time for "
                              + plan->begin()->getFD()->getNameAsString());

              std::tie(DerivativeDecl, DerivativeDeclContext) =
                m_DerivativeBuilder->Derive(*I, *plan);
            }

            if (DerivativeDecl) {
              collector.UpdatePlan(DerivativeDecl, &*plan);
              if (I + 1 == plan->end()) // The last element
                plan->updateCall(DerivativeDecl, m_CI.getSema());

              // if enabled, print source code of the derived functions
              if (m_DO.DumpDerivedFn) {
                 DerivativeDecl->print(llvm::outs(), Policy);
              }
              // if enabled, print ASTs of the derived functions
              if (m_DO.DumpDerivedAST) {
                 DerivativeDecl->dumpColor();
              }
              // if enabled, print the derivatives in a file.
              if (m_DO.GenerateSourceFile) {
                 std::error_code err;
                 llvm::raw_fd_ostream f("Derivatives.cpp", err,
                                        llvm::sys::fs::F_Append);
                 DerivativeDecl->print(f, Policy);
                 f.flush();
              }
              // Call CodeGen only if the produced decl is a top-most decl.
              Decl* DerivativeDeclOrEnclosingContext = DerivativeDeclContext ?
                DerivativeDeclContext : DerivativeDecl;
              bool isTU =
                DerivativeDeclOrEnclosingContext->getDeclContext()->isTranslationUnit();
              if (isTU) {
                m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(
                  DerivativeDeclOrEnclosingContext));
              }
            }
      }
      return true; // Happiness
    }

    /// Keeps track if we encountered #pragma clad on/off.
    // FIXME: Figure out how to make it a member of CladPlugin.
    SourceLocation CladPragmaEnabledLoc = SourceLocation();

    bool CladPlugin::ShouldProcessDecl(DeclGroupRef DGR) {
      if (CladPragmaEnabledLoc.isValid()) {
        SourceLocation DGREndLoc = (*DGR.begin())->getLocEnd();
        SourceManager& SM = m_CI.getSourceManager();
        return SM.isBeforeInTranslationUnit(CladPragmaEnabledLoc, DGREndLoc);
      }
      // If we have included "clad/Differentiator/Differentiator.h" return here.
      if (m_HasRuntime)
        return true;

      ASTContext& C = m_CI.getASTContext();
      // The plugin has a lot of different ways to be compiled: in-tree,
      // out-of-tree and hybrid. When we pick up the wrong header files we
      // usually see a problem with C.Idents not being properly initialized.
      // This assert tries to catch such situations heuristically.
      assert(&C.Idents == &m_CI.getPreprocessor().getIdentifierTable()
             && "Miscompiled?");
      DeclarationName Name = &C.Idents.get("custom_derivatives");
      Sema &SemaR = m_CI.getSema();
      LookupResult R(SemaR, Name, SourceLocation(), Sema::LookupNamespaceName,
                     Sema::ForRedeclaration);
      SemaR.LookupQualifiedName(R, C.getTranslationUnitDecl(),
                                /*allowBuiltinCreation*/ false);
      m_HasRuntime = !R.empty();
      return m_HasRuntime;
    }


    // Define a pragma handler for #pragma clad
    class CladPragmaHandler : public PragmaHandler {
    public:
      CladPragmaHandler() : PragmaHandler("clad") { }
      void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                        Token &PragmaTok) {
        // Handle #pragma clad ON/OFF/DEFAULT
        if (PragmaTok.isNot(tok::identifier)) {
          PP.Diag(PragmaTok, diag::warn_pragma_diagnostic_invalid);
          return;
        }
        IdentifierInfo *II = PragmaTok.getIdentifierInfo();
        assert(II->isStr("clad"));

        tok::OnOffSwitch OOS;
        if (PP.LexOnOffSwitch(OOS))
          return; // failure

        if (OOS == tok::OOS_ON)
          CladPragmaEnabledLoc = PragmaTok.getLocation();
        else // tok::OOS_OFF or tok::OOS_DEFAULT
          CladPragmaEnabledLoc = SourceLocation();
      }
    };

  } // end namespace plugin
} // end namespace clad

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad", "Produces derivatives or arbitrary functions");

static PragmaHandlerRegistry::Add<CladPragmaHandler>
Y("clad", "Clad pragma directives handler.");
