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
#ifndef NDEBUG
        IdentifierInfo* II = PragmaTok.getIdentifierInfo();
        assert(II->isStr("clad"));
#endif

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
        : m_CI(CI), m_DO(DO), m_HasRuntime(false) {
#if CLANG_VERSION_MAJOR > 8

      FrontendOptions& Opts = CI.getFrontendOpts();
      // Find the path to clad.
      llvm::StringRef CladSoPath;
      for (llvm::StringRef P : Opts.Plugins)
        if (llvm::sys::path::stem(P).endswith("clad")) {
          CladSoPath = P;
          break;
        }

      // Register clad as a backend pass.
      CodeGenOptions& CGOpts = CI.getCodeGenOpts();
      CGOpts.PassPlugins.push_back(CladSoPath.str());
#endif // CLANG_VERSION_MAJOR > 8
    }

    CladPlugin::~CladPlugin() {}

    // We cannot use HandleTranslationUnit because codegen already emits code on
    // HandleTopLevelDecl calls and makes updateCall with no effect.
    bool CladPlugin::HandleTopLevelDecl(DeclGroupRef DGR) {
      if (!CheckBuiltins())
        return true;

      Sema& S = m_CI.getSema();

      if (!m_DerivativeBuilder)
        m_DerivativeBuilder.reset(new DerivativeBuilder(m_CI.getSema(), *this));

      // if HandleTopLevelDecl was called through clad we don't need to process
      // it for diff requests
      if (m_HandleTopLevelDeclInternal)
        return true;

      DiffSchedule requests{};
      DiffCollector collector(DGR, CladEnabledRange, requests, m_CI.getSema());

      if (requests.empty())
        return true;

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

    void CladPlugin::ProcessTopLevelDecl(Decl* D) {
      m_HandleTopLevelDeclInternal = true;
      m_CI.getASTConsumer().HandleTopLevelDecl(DeclGroupRef(D));
      m_HandleTopLevelDeclInternal = false;
    }

    FunctionDecl* CladPlugin::ProcessDiffRequest(DiffRequest& request) {
      Sema& S = m_CI.getSema();
      // Required due to custom derivatives function templates that might be
      // used in the function that we need to derive.
      S.PerformPendingInstantiations();
      if (request.Function->getDefinition())
        request.Function = request.Function->getDefinition();
      request.UpdateDiffParamsInfo(m_CI.getSema());
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
          m_DerivativeBuilder->AddErrorEstimationModel(
              estimationPlugin->InstantiateCustomModel(*m_DerivativeBuilder));
        }
      }

      // If enabled, set the proper fields in derivative builder.
      if (m_DO.PrintNumDiffErrorInfo) {
        m_DerivativeBuilder->setNumDiffErrDiag(true);
      }

      FunctionDecl* DerivativeDecl = nullptr;
      bool alreadyDerived = false;
      FunctionDecl* OverloadedDerivativeDecl = nullptr;
      {
        // FIXME: Move the timing inside the DerivativeBuilder. This would
        // require to pass in the DifferentiationOptions in the DiffPlan.
        // derive the collected functions
        bool WantTiming = getenv("LIBCLAD_TIMING");
        SimpleTimer Timer(WantTiming);
        Timer.setOutput("Generation time for " + FD->getNameAsString());

        auto DFI = m_DFC.Find(request);
        if (DFI.IsValid()) {
          DerivativeDecl = DFI.DerivedFn();
          OverloadedDerivativeDecl = DFI.OverloadedDerivedFn();
          alreadyDerived = true;
        } else {
          auto deriveResult = m_DerivativeBuilder->Derive(request);
          DerivativeDecl = deriveResult.derivative;
          OverloadedDerivativeDecl = deriveResult.overload;
        }
      }

      if (DerivativeDecl) {
        if (!alreadyDerived) {
          m_DFC.Add(
              DerivedFnInfo(request, DerivativeDecl, OverloadedDerivativeDecl));

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
                                   CLAD_COMPAT_llvm_sys_fs_Append);
            DerivativeDecl->print(f, Policy);
            f.flush();
          }

          S.MarkFunctionReferenced(SourceLocation(), DerivativeDecl);
          if (OverloadedDerivativeDecl)
            S.MarkFunctionReferenced(SourceLocation(),
                                     OverloadedDerivativeDecl);

          // We ideally should not call `HandleTopLevelDecl` for declarations
          // inside a namespace. After parsing a namespace that is defined
          // directly in translation unit context , clang calls
          // `BackendConsumer::HandleTopLevelDecl`.
          // `BackendConsumer::HandleTopLevelDecl` emits LLVM IR of each
          // declaration inside the namespace using CodeGen. We need to manually
          // call `HandleTopLevelDecl` for each new declaration added to a
          // namespace because `HandleTopLevelDecl` has already been called for
          // a namespace by Clang when the namespace is parsed.

          // Call CodeGen only if the produced Decl is a top-most
          // decl or is contained in a namespace decl.
          DeclContext* derivativeDC = DerivativeDecl->getDeclContext();
          bool isTUorND =
              derivativeDC->isTranslationUnit() || derivativeDC->isNamespace();
          if (isTUorND) {
            ProcessTopLevelDecl(DerivativeDecl);
            if (OverloadedDerivativeDecl)
              ProcessTopLevelDecl(OverloadedDerivativeDecl);
          }
        }
        bool lastDerivativeOrder = (request.CurrentDerivativeOrder ==
                                    request.RequestedDerivativeOrder);
        // If this is the last required derivative order, replace the function
        // inside a call to clad::differentiate/gradient with its derivative.
        if (request.CallUpdateRequired && lastDerivativeOrder)
          request.updateCall(DerivativeDecl, OverloadedDerivativeDecl,
                             m_CI.getSema());

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

  // Routine to check clang version at runtime against the clang version for
  // which clad was built.
  bool checkClangVersion() {
    std::string runtimeVersion = clang::getClangFullCPPVersion();
    std::string builtVersion = CLANG_MAJOR_VERSION;
    if (runtimeVersion.find(builtVersion) == std::string::npos)
      return false;
    else
      return true;
  }

  void DerivedFnCollector::Add(const DerivedFnInfo& DFI) {
    assert(!AlreadyExists(DFI) &&
           "We are generating same derivative more than once, or calling "
           "`DerivedFnCollector::Add` more than once for the same derivative "
           ". Ideally, we shouldn't do either.");
    m_DerivedFnInfoCollection[DFI.OriginalFn()].push_back(DFI);
  }

  bool DerivedFnCollector::AlreadyExists(const DerivedFnInfo& DFI) const {
    auto subCollectionIt = m_DerivedFnInfoCollection.find(DFI.OriginalFn());
    if (subCollectionIt == m_DerivedFnInfoCollection.end())
      return false;
    auto& subCollection = subCollectionIt->second;
    auto it = std::find_if(subCollection.begin(), subCollection.end(),
                           [&DFI](const DerivedFnInfo& info) {
                             return DerivedFnInfo::
                                 RepresentsSameDerivative(DFI, info);
                           });
    return it != subCollection.end();
  }

  DerivedFnInfo DerivedFnCollector::Find(const DiffRequest& request) const {
    auto subCollectionIt = m_DerivedFnInfoCollection.find(request.Function);
    if (subCollectionIt == m_DerivedFnInfoCollection.end())
      return DerivedFnInfo();
    auto& subCollection = subCollectionIt->second;
    auto it = std::find_if(subCollection.begin(), subCollection.end(),
                           [&request](DerivedFnInfo DFI) {
                             return DFI.SatisfiesRequest(request);
                           });
    if (it == subCollection.end())
      return DerivedFnInfo();
    return *it;
  }
} // end namespace clad

// Attach the frontend plugin.

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad", "Produces derivatives or arbitrary functions");

static PragmaHandlerRegistry::Add<CladPragmaHandler>
    Y("clad", "Clad pragma directives handler.");

#include "clang/Basic/Version.h" // for CLANG_VERSION_MAJOR
#if CLANG_VERSION_MAJOR > 8

// Attach the backend plugin.

#include "ClangBackendPlugin.h"

#define BACKEND_PLUGIN_NAME "CladBackendPlugin"
// FIXME: Add a proper versioning that's based on CLANG_VERSION_STRING and
// a similar approach for clad (see Version.cpp and VERSION).
#define BACKEND_PLUGIN_VERSION "FIXME"
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, BACKEND_PLUGIN_NAME, BACKEND_PLUGIN_VERSION,
          clad::ClangBackendPluginPass::registerCallbacks};
}

#endif // CLANG_VERSION_MAJOR
