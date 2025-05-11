//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "ClangPlugin.h"

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/EstimationModel.h"
#include "clad/Differentiator/Sins.h"
#include "clad/Differentiator/Timers.h"
#include "clad/Differentiator/Version.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/LLVM.h" // isa, dyn_cast
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include "clad/Differentiator/Compatibility.h"

#include <algorithm>
#include <cstdlib>  // for getenv
#include <iostream> // for std::cerr
#include <memory>

using namespace clang;

namespace clad {
void InitTimers();

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
#if CLANG_VERSION_MAJOR > 11
      bool WantTiming = m_CI.getCodeGenOpts().TimePasses;
#else
      bool WantTiming = m_CI.getFrontendOpts().ShowTimers;
#endif

      if (WantTiming || getenv("CLAD_ENABLE_TIMING"))
        InitTimers();

      FrontendOptions& Opts = CI.getFrontendOpts();
      // Find the path to clad.
      llvm::StringRef CladSoPath;
      for (llvm::StringRef P : Opts.Plugins)
        if (llvm::sys::path::stem(P).ends_with("clad")) {
          CladSoPath = P;
          break;
        }

      if (!CladSoPath.empty()) {
        // Register clad as a backend pass.
        CodeGenOptions& CGOpts = CI.getCodeGenOpts();
        CGOpts.PassPlugins.push_back(CladSoPath.str());
      }

      // Add define for __CLAD__, so that CladFunction::CladFunction()
      // doesn't throw an error.
      auto predefines = m_CI.getPreprocessor().getPredefines();
      predefines.append("#define __CLAD__ 1\n");
      m_CI.getPreprocessor().setPredefines(predefines);
    }

    CladPlugin::~CladPlugin() {}

    ALLOW_ACCESS(MultiplexConsumer, Consumers,
                 std::vector<std::unique_ptr<ASTConsumer>>);

    void CladPlugin::Initialize(clang::ASTContext& C) {
      // We know we have a multiplexer. We commit a sin here by stealing it and
      // making the consumer pass-through so that we can delay all operations
      // until clad is happy.

      auto& MultiplexC = cast<MultiplexConsumer>(m_CI.getASTConsumer());
      auto& RobbedCs = ACCESS(MultiplexC, Consumers);
      assert(RobbedCs.back().get() == this && "Clad is not the last consumer");
      std::vector<std::unique_ptr<ASTConsumer>> StolenConsumers;

      // The range-based for loop in MultiplexConsumer::Initialize has
      // dispatched this call. Generally, it is unsafe to delete elements while
      // iterating but we know we are in the end of the loop and ::end() won't
      // be invalidated.
      std::move(RobbedCs.begin(), RobbedCs.end() - 1,
                std::back_inserter(StolenConsumers));
      RobbedCs.erase(RobbedCs.begin(), RobbedCs.end() - 1);
      m_Multiplexer.reset(new MultiplexConsumer(std::move(StolenConsumers)));
    }

    void CladPlugin::HandleTopLevelDeclForClad(DeclGroupRef DGR) {
      if (!CheckBuiltins())
        return;
#if CLANG_VERSION_MAJOR > 16
      Sema& S = m_CI.getSema();
      RequestOptions opts{};
      SetRequestOptions(opts);
      // Traverse all constexpr FunctionDecls for the static graph only once to
      // differentiate them immeditely.
      {
        TimedAnalysisRegion R("Rest of constexpr TU");
        for (Decl* D : DGR) {
          if (!isa<FunctionDecl>(D))
            continue;
          auto* FD = cast<FunctionDecl>(D);
          if (FD->isConstexpr() || !m_Multiplexer) {
            DiffCollector collector(DGR, CladEnabledRange, m_DiffRequestGraph,
                                    S, opts, m_DFC);
            break;
          }
        }
      }

      for (DiffRequest& request : m_DiffRequestGraph.getNodes()) {
        if (request.ImmediateMode && request.Function->isConstexpr()) {
          m_DiffRequestGraph.setCurrentProcessingNode(request);
          ProcessDiffRequest(request);
          m_DiffRequestGraph.markCurrentNodeProcessed();
        }
      }
#endif

      // We could not delay the processing of derivatives, inform act as if each
      // call is final. That would still have vgvassilev/clad#248 unresolved.
      if (!m_Multiplexer)
        FinalizeTranslationUnit();
    }

    static void printDerivative(clang::Decl* D, bool DeclarationOnly,
                                const DifferentiationOptions& DO) {
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      Policy.Bool = true;

      // if enabled, print source code of the derivatives
      if (DO.DumpDerivedFn) {
        D->print(llvm::outs(), Policy);
        if (DeclarationOnly)
          llvm::outs() << ";\n";
      }

      // if enabled, print ASTs of the derivatives
      if (DO.DumpDerivedAST)
        D->dumpColor();

      // if enabled, print the derivatives in a file
      if (DO.GenerateSourceFile) {
        std::error_code err;
        llvm::raw_fd_ostream f("Derivatives.cpp", err,
                               CLAD_COMPAT_llvm_sys_fs_Append);
        D->print(f, Policy);
        if (DeclarationOnly)
          f << ";\n";
        f.flush();
      }
    }

    FunctionDecl* CladPlugin::ProcessDiffRequest(DiffRequest& request) {
      Sema& S = m_CI.getSema();
      if (!m_DerivativeBuilder)
        m_DerivativeBuilder = std::make_unique<DerivativeBuilder>(
            S, *this, m_DFC, m_DiffRequestGraph);

      if (request.Global) {
        auto deriveResult = m_DerivativeBuilder->Derive(request);
        auto* VDDiff = cast_or_null<VarDecl>(deriveResult.derivative);
        ProcessTopLevelDecl(VDDiff);
        // Dump the declaration if requested.
        printDerivative(VDDiff, request.DeclarationOnly, m_DO);
        return nullptr;
      }

      // Required due to custom derivatives function templates that might be
      // used in the function that we need to derive.
      // FIXME: Remove the call to PerformPendingInstantiations().
      S.PerformPendingInstantiations();
      if (request.Function->getDefinition())
        request.Function = request.Function->getDefinition();
      request.UpdateDiffParamsInfo(m_CI.getSema());
      const FunctionDecl* FD = request.Function;
      ASTContext& C = S.getASTContext();
      clang::PrintingPolicy Policy = C.getPrintingPolicy();
#if CLANG_VERSION_MAJOR > 10
      // Our testsuite expects 'a<b<c> >' rather than 'a<b<c>>'.
      Policy.SplitTemplateClosers = true;
#endif
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
          unsigned diagID = S.Diags.getCustomDiagID(
              DiagnosticsEngine::Error, "Failed to load '%0', %1. Aborting.");
          clang::Sema::SemaDiagnosticBuilder stream = S.Diag(noLoc, diagID);
          stream << m_DO.CustomModelName << Err;
          return nullptr;
        }
        for (auto it = ErrorEstimationModelRegistry::begin(),
                  ie = ErrorEstimationModelRegistry::end();
             it != ie; ++it) {
          auto estimationPlugin = it->instantiate();
          m_DerivativeBuilder->AddErrorEstimationModel(
              estimationPlugin->InstantiateCustomModel(*m_DerivativeBuilder,
                                                       request));
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
        auto DFI = m_DFC.Find(request);
        if (DFI.IsValid()) {
          DerivativeDecl = DFI.DerivedFn();
          OverloadedDerivativeDecl = DFI.OverloadedDerivedFn();
          alreadyDerived = true;
        } else {
          auto deriveResult = m_DerivativeBuilder->Derive(request);
          DerivativeDecl = cast_or_null<FunctionDecl>(deriveResult.derivative);
          OverloadedDerivativeDecl = deriveResult.overload;
        }
      }

      if (DerivativeDecl) {
        if (!alreadyDerived) {
          m_DFC.Add(
              DerivedFnInfo(request, DerivativeDecl, OverloadedDerivativeDecl));

          printDerivative(DerivativeDecl, request.DeclarationOnly, m_DO);

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
          // FIXME: We could get rid of this by prepending the produced
          // derivatives in CladPlugin::HandleTranslationUnitDecl
          DeclContext* derivativeDC = DerivativeDecl->getLexicalDeclContext();
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

        if (request.DeclarationOnly)
          request.DerivedFDPrototypes.push_back(DerivativeDecl);

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

    void CladPlugin::SendToMultiplexer() {
      for (unsigned i = m_MultiplexerProcessedDelayedCallsIdx;
           i < m_DelayedCalls.size(); ++i) {
        auto DelayedCall = m_DelayedCalls[i];
        DeclGroupRef& D = DelayedCall.m_DGR;
        switch (DelayedCall.m_Kind) {
        case CallKind::HandleCXXStaticMemberVarInstantiation:
          m_Multiplexer->HandleCXXStaticMemberVarInstantiation(
              cast<VarDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleTopLevelDecl:
          m_Multiplexer->HandleTopLevelDecl(D);
          break;
        case CallKind::HandleInlineFunctionDefinition:
          m_Multiplexer->HandleInlineFunctionDefinition(
              cast<FunctionDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleInterestingDecl:
          m_Multiplexer->HandleInterestingDecl(D);
          break;
        case CallKind::HandleTagDeclDefinition:
          m_Multiplexer->HandleTagDeclDefinition(
              cast<TagDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleTagDeclRequiredDefinition:
          m_Multiplexer->HandleTagDeclRequiredDefinition(
              cast<TagDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleCXXImplicitFunctionInstantiation:
          m_Multiplexer->HandleCXXImplicitFunctionInstantiation(
              cast<FunctionDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleTopLevelDeclInObjCContainer:
          m_Multiplexer->HandleTopLevelDeclInObjCContainer(D);
          break;
        case CallKind::HandleImplicitImportDecl:
          m_Multiplexer->HandleImplicitImportDecl(
              cast<ImportDecl>(D.getSingleDecl()));
          break;
        case CallKind::CompleteTentativeDefinition:
          m_Multiplexer->CompleteTentativeDefinition(
              cast<VarDecl>(D.getSingleDecl()));
          break;
        case CallKind::CompleteExternalDeclaration:
          m_Multiplexer->CompleteExternalDeclaration(
              cast<VarDecl>(D.getSingleDecl()));
          break;
        case CallKind::AssignInheritanceModel:
          m_Multiplexer->AssignInheritanceModel(
              cast<CXXRecordDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleVTable:
          m_Multiplexer->HandleVTable(cast<CXXRecordDecl>(D.getSingleDecl()));
          break;
        case CallKind::InitializeSema:
          m_Multiplexer->InitializeSema(m_CI.getSema());
          break;
        };
      }

      m_MultiplexerProcessedDelayedCallsIdx = m_DelayedCalls.size();
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
                     CLAD_COMPAT_Sema_ForVisibleRedeclaration);
      SemaR.LookupQualifiedName(R, C.getTranslationUnitDecl(),
                                /*allowBuiltinCreation*/ false);
      m_HasRuntime = !R.empty();
      return m_HasRuntime;
    }

    static void SetTBRAnalysisOptions(const DifferentiationOptions& DO,
                                      RequestOptions& opts) {
      // If user has explicitly specified the mode for TBR analysis, use it.
      if (DO.EnableTBRAnalysis || DO.DisableTBRAnalysis)
        opts.EnableTBRAnalysis = DO.EnableTBRAnalysis && !DO.DisableTBRAnalysis;
      else
        opts.EnableTBRAnalysis = true; // Default mode.
    }

    static void SetActivityAnalysisOptions(const DifferentiationOptions& DO,
                                           RequestOptions& opts) {
      // If user has explicitly specified the mode for AA, use it.
      if (DO.EnableVariedAnalysis || DO.DisableVariedAnalysis)
        opts.EnableVariedAnalysis =
            DO.EnableVariedAnalysis && !DO.DisableVariedAnalysis;
      else
        opts.EnableVariedAnalysis = false; // Default mode.
    }

    static void SetUsefulAnalysisOptions(const DifferentiationOptions& DO,
                                         RequestOptions& opts) {
      // If user has explicitly specified the mode for TBR analysis, use it.
      if (DO.EnableUsefulAnalysis || DO.DisableUsefulAnalysis)
        opts.EnableUsefulAnalysis =
            DO.EnableUsefulAnalysis && !DO.DisableUsefulAnalysis;
      else
        opts.EnableUsefulAnalysis = false; // Default mode.
    }
    void CladPlugin::SetRequestOptions(RequestOptions& opts) const {
      SetTBRAnalysisOptions(m_DO, opts);
      SetActivityAnalysisOptions(m_DO, opts);
      SetUsefulAnalysisOptions(m_DO, opts);
    }

    void CladPlugin::FinalizeTranslationUnit() {
      Sema& S = m_CI.getSema();
      // Restore the TUScope that became a 0 in Sema::ActOnEndOfTranslationUnit.
      if (!m_CI.getPreprocessor().isIncrementalProcessingEnabled())
        S.TUScope = m_StoredTUScope;
      constexpr bool Enabled = true;
      Sema::GlobalEagerInstantiationScope GlobalInstantiations(S, Enabled);
      Sema::LocalEagerInstantiationScope LocalInstantiations(S);

      if (!m_DiffRequestGraph.isProcessingNode()) {
        // This check is to avoid recursive processing of the graph, as
        // HandleTopLevelDecl can be called recursively in non-standard
        // setup for code generation.
        DiffRequest request = m_DiffRequestGraph.getNextToProcessNode();
        while (request.Function || request.Global) {
          m_DiffRequestGraph.setCurrentProcessingNode(request);
          ProcessDiffRequest(request);
          m_DiffRequestGraph.markCurrentNodeProcessed();
          request = m_DiffRequestGraph.getNextToProcessNode();
        }
      }

      // Put the TUScope in a consistent state after clad is done.
      if (!m_CI.getPreprocessor().isIncrementalProcessingEnabled())
        S.TUScope = nullptr;

      // Force emission of the produced pending template instantiations.
      LocalInstantiations.perform();
      GlobalInstantiations.perform();
    }

    void CladPlugin::HandleTranslationUnit(ASTContext& C) {
      Sema& S = m_CI.getSema();
      RequestOptions opts{};
      SetRequestOptions(opts);
      // Traverse all collected DeclGroupRef only once to create the static
      // graph.
      TimedAnalysisRegion R("Rest of TU");
      for (auto DCI : m_DelayedCalls)
        for (Decl* D : DCI.m_DGR) {
          if (const auto* FD = dyn_cast<FunctionDecl>(D))
            if (FD->isConstexpr())
              continue;
          DiffCollector collector(DCI.m_DGR, CladEnabledRange,
                                  m_DiffRequestGraph, S, opts, m_DFC);
          break;
        }

      FinalizeTranslationUnit();
      SendToMultiplexer();
      m_Multiplexer->HandleTranslationUnit(C);
    }

    void CladPlugin::PrintStats() {
      llvm::errs() << "*** INFORMATION ABOUT THE DELAYED CALLS\n";
      for (const DelayedCallInfo& DCI : m_DelayedCalls) {
        llvm::errs() << "   ";
        switch (DCI.m_Kind) {
        case CallKind::HandleCXXStaticMemberVarInstantiation:
          llvm::errs() << "HandleCXXStaticMemberVarInstantiation";
          break;
        case CallKind::HandleTopLevelDecl:
          llvm::errs() << "HandleTopLevelDecl";
          break;
        case CallKind::HandleInlineFunctionDefinition:
          llvm::errs() << "HandleInlineFunctionDefinition";
          break;
        case CallKind::HandleInterestingDecl:
          llvm::errs() << "HandleInterestingDecl";
          break;
        case CallKind::HandleTagDeclDefinition:
          llvm::errs() << "HandleTagDeclDefinition";
          break;
        case CallKind::HandleTagDeclRequiredDefinition:
          llvm::errs() << "HandleTagDeclRequiredDefinition";
          break;
        case CallKind::HandleCXXImplicitFunctionInstantiation:
          llvm::errs() << "HandleCXXImplicitFunctionInstantiation";
          break;
        case CallKind::HandleTopLevelDeclInObjCContainer:
          llvm::errs() << "HandleTopLevelDeclInObjCContainer";
          break;
        case CallKind::HandleImplicitImportDecl:
          llvm::errs() << "HandleImplicitImportDecl";
          break;
        case CallKind::CompleteTentativeDefinition:
          llvm::errs() << "CompleteTentativeDefinition";
          break;
        case CallKind::CompleteExternalDeclaration:
          llvm::errs() << "CompleteExternalDeclaration";
          break;
        case CallKind::AssignInheritanceModel:
          llvm::errs() << "AssignInheritanceModel";
          break;
        case CallKind::HandleVTable:
          llvm::errs() << "HandleVTable";
          break;
        case CallKind::InitializeSema:
          llvm::errs() << "InitializeSema";
          break;
        };
        for (const clang::Decl* D : DCI.m_DGR) {
          llvm::errs() << " " << D;
          if (const auto* ND = dyn_cast<NamedDecl>(D))
            llvm::errs() << " " << ND->getNameAsString();
        }
        llvm::errs() << "\n";
      }

      // Print the graph of the diff requests.
      llvm::errs() << "\n*** INFORMATION ABOUT THE DIFF REQUESTS\n";
      m_DiffRequestGraph.dump();

      m_Multiplexer->PrintStats();
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
} // end namespace clad

// Attach the frontend plugin.

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad", "Produces derivatives or arbitrary functions");

static PragmaHandlerRegistry::Add<CladPragmaHandler>
    Y("clad", "Clad pragma directives handler.");

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
