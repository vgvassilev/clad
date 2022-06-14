//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "ClangPlugin.h"

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/EstimationModel.h"

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

#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include "clad/Differentiator/Compatibility.h"

#include <algorithm>

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
    /// Keeps track if we encountered #pragma clad on/off.
    // FIXME: Figure out how to make it a member of CladPlugin.
    std::vector<clang::SourceRange> CladEnabledRange;

    // Define a pragma handler for #pragma clad
    class CladPragmaHandler : public PragmaHandler {
    public:
      CladPragmaHandler() : PragmaHandler("clad") {}
      void HandlePragma(Preprocessor& PP, PragmaIntroducer Introducer,
                        Token& PragmaTok) override {
        // Handle #pragma clad ON/OFF/DEFAULT
        if (PragmaTok.isNot(tok::identifier)) {
          PP.Diag(PragmaTok, diag::warn_pragma_diagnostic_invalid);
          return;
        }
        IdentifierInfo* II = PragmaTok.getIdentifierInfo();
        assert(II->isStr("clad"));

        tok::OnOffSwitch OOS;
        if (PP.LexOnOffSwitch(OOS))
          return; // failure
        SourceLocation TokLoc = PragmaTok.getLocation();
        if (OOS == tok::OOS_ON) {
          SourceRange R(TokLoc, /*end*/ SourceLocation());
          // If a second ON is seen, ignore it if the interval is open.
          if (CladEnabledRange.empty() ||
              CladEnabledRange.back().getEnd().isValid())
            CladEnabledRange.push_back(R);
        } else if (!CladEnabledRange.empty()) { // OOS_OFF or OOS_DEFAULT
          assert(CladEnabledRange.back().getEnd().isInvalid());
          CladEnabledRange.back().setEnd(TokLoc);
        }
      }
    };

    CladPlugin::CladPlugin(CompilerInstance& CI, DifferentiationOptions& DO)
      : m_CI(CI), m_DO(DO), m_HasRuntime(false) { }
    CladPlugin::~CladPlugin() {}


    // We cannot use HandleTranslationUnit because codegen already emits code on
    // HandleTopLevelDecl calls and makes updateCall with no effect.
    bool CladPlugin::HandleTopLevelDecl(DeclGroupRef DGR) {
      if (!CheckBuiltins())
        return true;

      Sema& S = m_CI.getSema();

      if (!m_DerivativeBuilder)
        InitializeDerivativeBuilder();

      // if HandleTopLevelDecl was called through clad we don't need to process
      // it for diff requests
      if (m_HandleTopLevelDeclInternal)
        return true;

      DiffSchedule requests{};
      DiffCollector collector(DGR, CladEnabledRange, requests, m_CI.getSema());

      // FIXME: Remove the PerformPendingInstantiations altogether. We should
      // somehow make the relevant functions referenced.
      // Instantiate all pending for instantiations templates, because we will
      // need the full bodies to produce derivatives.
      // FIXME: Confirm if we really need `m_PendingInstantiationsInFlight`?
      if (!m_PendingInstantiationsInFlight) {
        m_PendingInstantiationsInFlight = true;
        S.PerformPendingInstantiations();
        m_PendingInstantiationsInFlight = false;
      }

      for (DiffRequest& request : requests)
        ProcessDiffRequest(request);
      return true; // Happiness
    }

    void ProcessTopLevelDecl(CladPlugin& P, Decl* D) {
      P.ProcessTopLevelDecl(D);
    }

    void CladPlugin::ProcessTopLevelDecl(Decl* D) {
      m_HandleTopLevelDeclInternal = true;
      m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(D));
      m_HandleTopLevelDeclInternal = false;
    }

    void DumpRequestedInfo(CladPlugin& P, const FunctionDecl* sourceFn,
                           const FunctionDecl* derivedFn) {
      P.DumpRequestedInfo(sourceFn, derivedFn);
    }

    void CladPlugin::DumpRequestedInfo(const FunctionDecl* sourceFn,
                                       const FunctionDecl* derivedFn) {
      // set up printing policy
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      Policy.Bool = true;
      // if enabled, print source code of the original functions
      if (m_DO.DumpSourceFn) {
        sourceFn->print(llvm::outs(), Policy);
      }
      // if enabled, print ASTs of the original functions
      if (m_DO.DumpSourceFnAST) {
        sourceFn->dumpColor();
      }

      // if enabled, print source code of the derived functions
      if (m_DO.DumpDerivedFn) {
        derivedFn->print(llvm::outs(), Policy);
      }

      // if enabled, print ASTs of the derived functions
      if (m_DO.DumpDerivedAST) {
        derivedFn->dumpColor();
      }

      // if enabled, print the derivatives in a file.
      if (m_DO.GenerateSourceFile) {
        std::error_code err;
        llvm::raw_fd_ostream f("Derivatives.cpp", err,
                               CLAD_COMPAT_llvm_sys_fs_Append);
        derivedFn->print(f, Policy);
        f.flush();
      }
    }

    void CladPlugin::InitializeDerivativeBuilder() {
      m_DerivativeBuilder.reset(new DerivativeBuilder(m_CI.getSema(), *this));

      // if enabled, load the dynamic library input from user to use
      // as a custom estimation model.
      if (m_DO.CustomEstimationModel) {
        std::string Err;
        if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(
                m_DO.CustomModelName.c_str(), &Err)) {
          auto& SemaInst = m_CI.getSema();
          unsigned diagID = SemaInst.Diags.getCustomDiagID(
              DiagnosticsEngine::Error, "Failed to load '%0', %1. Aborting.");
          clang::Sema::SemaDiagnosticBuilder stream =
              SemaInst.Diag(noLoc, diagID);
          stream << m_DO.CustomModelName << Err;
          return;
        }
        for (auto it = ErrorEstimationModelRegistry::begin(),
                  ie = ErrorEstimationModelRegistry::end();
             it != ie; ++it) {
          auto estimationPlugin = it->instantiate();
          m_DerivativeBuilder->SetErrorEstimationModel(
              estimationPlugin->InstantiateCustomModel(*m_DerivativeBuilder));
        }
      }

      // If enabled, set the proper fields in derivative builder.
      if (m_DO.PrintNumDiffErrorInfo) {
        m_DerivativeBuilder->setNumDiffErrDiag(true);
      }
    }

    FunctionDecl* CladPlugin::ProcessDiffRequest(DiffRequest& request) {
      return m_DerivativeBuilder->ProcessDiffRequest(request);
    }

    bool CladPlugin::CheckBuiltins() {
      // If we have included "clad/Differentiator/Differentiator.h" return.
      if (m_HasRuntime)
        return true;

      ASTContext& C = m_CI.getASTContext();
      // The plugin has a lot of different ways to be compiled: in-tree,
      // out-of-tree and hybrid. When we pick up the wrong header files we
      // usually see a problem with C.Idents not being properly initialized.
      // This assert tries to catch such situations heuristically.
      assert(&C.Idents == &m_CI.getPreprocessor().getIdentifierTable()
             && "Miscompiled?");
      // FIXME: Use `utils::LookupNSD` instead.
      DeclarationName Name = &C.Idents.get("clad");
      Sema &SemaR = m_CI.getSema();
      LookupResult R(SemaR, Name, SourceLocation(), Sema::LookupNamespaceName,
                     clad_compat::Sema_ForVisibleRedeclaration);
      SemaR.LookupQualifiedName(R, C.getTranslationUnitDecl(),
                                /*allowBuiltinCreation*/ false);
      m_HasRuntime = !R.empty();
      return m_HasRuntime;
    }
  } // end namespace plugin
} // end namespace clad

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad", "Produces derivatives or arbitrary functions");

static PragmaHandlerRegistry::Add<CladPragmaHandler>
    Y("clad", "Clad pragma directives handler.");
