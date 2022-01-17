//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "ClangPlugin.h"

#include "clad/Differentiator/ASTHelper.h"
#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/EstimationModel.h"
#include "clad/Differentiator/DerivedTypesHandler.h"
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

#include "llvm/Support/Registry.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include "clad/Differentiator/Compatibility.h"

#include <map>
#include <string>

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
        : m_CI(CI), m_DO(DO), m_HasRuntime(false) {}
    CladPlugin::~CladPlugin() {}

    // We cannot use HandleTranslationUnit because codegen already emits code on
    // HandleTopLevelDecl calls and makes updateCall with no effect.
    bool CladPlugin::HandleTopLevelDecl(DeclGroupRef DGR) {
      // llvm::errs()<<"Print declarations:\n";
      // for (auto it=DGR.begin(); it != DGR.end(); ++it) {
      //   auto DC = (*it)->getDeclContext();
      //   if (auto ND = dyn_cast<NamespaceDecl>(DC)) {
      //     if (ND->getNameAsString() == "clad") {
      //       continue;
      //     } else {
      //       llvm::errs()<<"ND: "<<ND->getNameAsString()<<"\n";
      //     }
      //   }
      //   // (*it)->dumpColor();
      //   if (auto ND = dyn_cast<NamedDecl>(*it)) {
      //     llvm::errs()<<ND->getNameAsString()<<"\n";
      //   }
      //   llvm::errs()<<"\n";
      // }
      // llvm::errs()<<"END\n\n";

      if (!CheckBuiltins())
        return true;

      Sema& S = m_CI.getSema();

      if (!m_DTH)
        m_DTH.reset(new DerivedTypesHandler(m_CI.getASTConsumer(), m_CI.getSema()));

      if (!m_DerivativeBuilder)
        m_DerivativeBuilder.reset(new DerivativeBuilder(m_CI.getSema(), *this, *m_DTH));

      // if HandleTopLevelDecl was called through clad we don't need to process
      // it for diff requests
      if (m_HandleTopLevelDeclInternal)
        return true;

      DiffSchedule requests{};
      llvm::SmallVector<ClassTemplateSpecializationDecl*, 16> derivedTypeRequests;
      DiffCollector collector(DGR, CladEnabledRange, m_Derivatives, requests,
                              derivedTypeRequests, m_CI.getSema());

      // FIXME: Remove the PerformPendingInstantiations altogether. We should
      // somehow make the relevant functions referenced.
      // Instantiate all pending for instantiations templates, because we will
      // need the full bodies to produce derivatives.
      if (!m_PendingInstantiationsInFlight) {
        m_PendingInstantiationsInFlight = true;
        S.PerformPendingInstantiations();
        m_PendingInstantiationsInFlight = false;
      }

      for (auto RD : derivedTypeRequests) {
        ProcessDerivedTypeRequest(RD);
      }

      for (DiffRequest& request : requests) {
        ProcessDiffRequest(request);
      }
      
      return true; // Happiness
    }

    void CladPlugin::ProcessTopLevelDecl(Decl* D) {
      m_HandleTopLevelDeclInternal = true;
      m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(D));
      m_HandleTopLevelDeclInternal = false;
    }
    void CladPlugin::ProcessDerivedTypeRequest(ClassTemplateSpecializationDecl* TS) {
      auto& semaRef = m_CI.getSema();
      auto yQType = TS->getTemplateArgs().get(0).getAsType();
      auto xQType = TS->getTemplateArgs().get(1).getAsType();
      m_DTH->InitialiseDerivedType(yQType, xQType);
    }
    FunctionDecl* CladPlugin::ProcessDiffRequest(DiffRequest& request) {
      const FunctionDecl* FD = request.Function;
      // set up printing policy
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      Policy.Bool = true;
      // if enabled, print source code of the original functions
      if (m_DO.DumpSourceFn) {
        FD->print(llvm::outs(), Policy);
      }
      // if enabled, print ASTs of the original functions
      if (m_DO.DumpSourceFnAST) {
        FD->dumpColor();
      }
      // if enabled, load the dynamic library input from user to use
      // as a custom estimation model.
      if (m_DO.CustomEstimationModel) {
        std::string Err;
        if (llvm::sys::DynamicLibrary::
                LoadLibraryPermanently(m_DO.CustomModelName.c_str(), &Err)) {
          auto& SemaInst = m_CI.getSema();
          unsigned diagID = SemaInst.Diags.getCustomDiagID(
              DiagnosticsEngine::Error, "Failed to load '%0', %1. Aborting.");
          clang::Sema::SemaDiagnosticBuilder stream = SemaInst.Diag(noLoc,
                                                                    diagID);
          stream << m_DO.CustomModelName << Err;
          return nullptr;
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

      FunctionDecl* DerivativeDecl = nullptr;
      Decl* DerivativeDeclContext = nullptr;
      FunctionDecl* OverloadedDerivativeDecl = nullptr;
      {
        // FIXME: Move the timing inside the DerivativeBuilder. This would
        // require to pass in the DifferentiationOptions in the DiffPlan.
        // derive the collected functions
        bool WantTiming = getenv("LIBCLAD_TIMING");
        SimpleTimer Timer(WantTiming);
        Timer.setOutput("Generation time for " + FD->getNameAsString());

        // TODO: Maybe find a better way to declare and use
        //  OverloadedDeclWithContext
        std::tie(DerivativeDecl, DerivativeDeclContext,
                 OverloadedDerivativeDecl) =
            m_DerivativeBuilder->Derive(FD, request);
      }
      if (OverloadedDerivativeDecl) {
        llvm::errs()<<"Dumping OverloadedDerivativeDecl:\n";
        OverloadedDerivativeDecl->print(llvm::errs(), Policy);
      }
      if (DerivativeDecl) {
        auto I = m_Derivatives.insert(DerivativeDecl);
        (void)I;
        assert(I.second);
        bool lastDerivativeOrder = 
          (request.CurrentDerivativeOrder == request.RequestedDerivativeOrder);
        // If this is the last required derivative order, replace the function
        // inside a call to clad::differentiate/gradient with its derivative.
        if (request.CallUpdateRequired && lastDerivativeOrder)
          request.updateCall(DerivativeDecl, OverloadedDerivativeDecl,
                             m_CI.getSema());

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
          llvm::raw_fd_ostream f("Derivatives.cpp",
                                 err, CLAD_COMPAT_llvm_sys_fs_Append);
          DerivativeDecl->print(f, Policy);
          f.flush();
        }
        // Call CodeGen only if the produced decl is a top-most decl.
        Decl* DerivativeDeclOrEnclosingContext = DerivativeDeclContext ?
          DerivativeDeclContext : DerivativeDecl;
        bool isTU = DerivativeDeclOrEnclosingContext->getDeclContext()->
          isTranslationUnit();
        if (isTU) {
          ProcessTopLevelDecl(DerivativeDeclOrEnclosingContext);
          if (OverloadedDerivativeDecl)
            ProcessTopLevelDecl(OverloadedDerivativeDecl);
        }

        // Last requested order was computed, return the result.
        if (lastDerivativeOrder)
          return DerivativeDecl;
        // If higher order derivatives are required, proceed to compute them
        // recursively.
        request.Function = DerivativeDecl;
        request.CurrentDerivativeOrder += 1;
        return ProcessDiffRequest(request);
      }
      return nullptr;
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

// instantiate our error estimation model registry so that we can register
// custom models passed by users as a shared lib
LLVM_INSTANTIATE_REGISTRY(ErrorEstimationModelRegistry)
