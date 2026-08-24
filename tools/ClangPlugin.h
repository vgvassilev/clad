//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_CLANG_PLUGIN
#define CLAD_CLANG_PLUGIN

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/DiffScheduler.h"
#include "clad/Differentiator/Version.h"
#include "../lib/Differentiator/DerivativePrinter.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Sema/SemaConsumer.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>

namespace clang {
  class ASTContext;
  class CallExpr;
  class DeclGroupRef;
  class Expr;
  class FunctionDecl;
  class ParmVarDecl;
  class Sema;
} // namespace clang

namespace clad {

bool checkClangVersion();
namespace plugin {
/// What the command line last said about one analysis. Unset means no switch
/// named it, so it runs at the default Analyses.def gives it. Not spelled
/// 'Default', which is a CLAD_ANALYSIS parameter and would be substituted
/// inside the macro bodies naming this enum.
enum class AnalysisSwitch : std::uint8_t { Unset, On, Off };

struct DifferentiationOptions {
  // Plain bool, not `: 1` bit-fields: packing single-bit bools into shared
  // bytes makes the compiler emit each ctor-init-list write as a
  // read-modify-write of the storage byte. The first RMW reads the byte while
  // it is still uninitialised, which MSan reports as a SEGV on uninitialised
  // memory under -fsanitize=memory.
  bool DumpSourceFn = false;
  bool DumpSourceFnAST = false;
  bool DumpDerivedFn = false;
  bool DumpDerivedAST = false;
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Remark##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"
  bool GenerateSourceFile = false;
  bool DumpGeneratedSource = false;
  bool ValidateClangVersion = true;
  /// What the command line asked of each analysis. The two bools record which
  /// of the original -enable-<x>/-disable-<x> pair were seen, so that giving
  /// both is still diagnosed rather than resolved by order. See Analyses.def.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  AnalysisSwitch Id##Switch = AnalysisSwitch::Unset;                           \
  bool Enable##Id##Analysis = false;                                           \
  bool Disable##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"
  bool PrintNumDiffErrorInfo = false;
  bool EmitPortingHints = false;
};

/// Match one of the original per-analysis switches, -enable-<x> or
/// -disable-<x>. Returns false if \p Arg is not such a switch.
inline bool setAnalysisFromFlag(DifferentiationOptions& DO,
                                llvm::StringRef Arg) {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  if (Arg == "-enable-" Legacy) {                                              \
    DO.Enable##Id##Analysis = true;                                            \
    DO.Id##Switch = AnalysisSwitch::On;                                        \
    return true;                                                               \
  }                                                                            \
  if (Arg == "-disable-" Legacy) {                                             \
    DO.Disable##Id##Analysis = true;                                           \
    DO.Id##Switch = AnalysisSwitch::Off;                                       \
    return true;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
  return false;
}

enum class AnalysisFlagResult : std::uint8_t { NotMine, Ok, Error };

/// Match -fenable-analysis=<name> or -fdisable-analysis=<name>. The last such
/// switch on the command line decides, so a build that turns everything off by
/// default can be overridden one analysis at a time.
inline AnalysisFlagResult setAnalysisByName(DifferentiationOptions& DO,
                                            llvm::StringRef Arg) {
  AnalysisSwitch To = AnalysisSwitch::On;
  if (!Arg.consume_front("-fenable-analysis=")) {
    if (!Arg.consume_front("-fdisable-analysis="))
      return AnalysisFlagResult::NotMine;
    To = AnalysisSwitch::Off;
  }

  // 'all' asks for the conservative derivative rather than for a particular
  // list, so it covers analyses added after the command line was written.
  // Only disabling has that meaning: falling back is safe for every analysis
  // by construction, while whether one is sound to run is what its default
  // encodes, so there is no configuration 'enable all' would name.
  if (Arg == "all" && To == AnalysisSwitch::Off) {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  DO.Id##Switch = AnalysisSwitch::Off;
#include "clad/Differentiator/Analyses.def"
    return AnalysisFlagResult::Ok;
  }

#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  if (Arg == (Name)) {                                                         \
    DO.Id##Switch = To;                                                        \
    return AnalysisFlagResult::Ok;                                             \
  }
#include "clad/Differentiator/Analyses.def"

  llvm::errs() << "clad: Error: unknown analysis '" << Arg << "'; known:";
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc) llvm::errs() << " " Name;
#include "clad/Differentiator/Analyses.def"
  llvm::errs() << "; -fdisable-analysis also takes 'all'.\n";
  return AnalysisFlagResult::Error;
}

/// Match -Rclad-analysis=<name>, asking for remarks about what that analysis
/// left in the generated code. Named after -Rpass-missed, which it is the
/// analogue of: clang's own remark machinery runs in the backend over LLVM IR
/// and cannot see an AST-level plugin, so clad carries its own switch.
inline AnalysisFlagResult remarkAnalysisByName(DifferentiationOptions& DO,
                                               llvm::StringRef Arg) {
  if (!Arg.consume_front("-Rclad-analysis="))
    return AnalysisFlagResult::NotMine;
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  if (Arg == (Name)) {                                                         \
    DO.Remark##Id##Analysis = true;                                            \
    return AnalysisFlagResult::Ok;                                             \
  }
#include "clad/Differentiator/Analyses.def"
  llvm::errs() << "clad: Error: unknown analysis '" << Arg << "'; known:";
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc) llvm::errs() << " " Name;
#include "clad/Differentiator/Analyses.def"
  llvm::errs() << ".\n";
  return AnalysisFlagResult::Error;
}

    class CladExternalSource : public clang::ExternalSemaSource {
    // ExternalSemaSource
    void ReadUndefinedButUsed(
        llvm::MapVector<clang::NamedDecl*, clang::SourceLocation>& Undefined)
        override {
      // namespace { double f_darg0(double x); } will issue a warning that
      // f_darg0 has internal linkage but is not defined. This is because we
      // have not yet started to differentiate it. The warning is triggered by
      // Sema::ActOnEndOfTranslationUnit before Clad is given control.
      // To avoid the warning we should remove the entry from here.
      using namespace clang;
      Undefined.remove_if([](std::pair<NamedDecl*, SourceLocation> P) {
        NamedDecl* ND = P.first;

        if (!ND->getDeclName().isIdentifier())
          return false;

        // FIXME: We should replace this comparison with the canonical decl
        // from the differentiation plan...
        llvm::StringRef Name = ND->getName();
        return Name.contains("_darg") || Name.contains("_grad") ||
               Name.contains("_hessian") || Name.contains("_jacobian");
      });
    }
    };
    class CladPlugin : public clang::SemaConsumer {
    clang::CompilerInstance& m_CI;
    DifferentiationOptions m_DO;
    std::unique_ptr<DerivativeBuilder> m_DerivativeBuilder;
    bool m_HasRuntime = false;
    /// Lazily constructed because it needs Sema, which is not available
    /// until InitializeSema; reach it through getScheduler().
    std::unique_ptr<DiffScheduler> m_Scheduler;
    /// Built on first use; a run that reports nothing never renders anything.
    std::unique_ptr<DerivativePrinter> m_DerivativePrinter;
    enum class CallKind {
      HandleCXXStaticMemberVarInstantiation,
      HandleTopLevelDecl,
      HandleInlineFunctionDefinition,
      HandleInterestingDecl,
      HandleTagDeclDefinition,
      HandleTagDeclRequiredDefinition,
      HandleCXXImplicitFunctionInstantiation,
      HandleTopLevelDeclInObjCContainer,
      HandleImplicitImportDecl,
      CompleteTentativeDefinition,
      CompleteExternalDeclaration,
      AssignInheritanceModel,
      HandleVTable,
      InitializeSema,
    };
    struct DelayedCallInfo {
      CallKind m_Kind;
      clang::DeclGroupRef m_DGR;
      DelayedCallInfo(CallKind K, clang::DeclGroupRef DGR)
          : m_Kind(K), m_DGR(DGR) {}
      DelayedCallInfo(CallKind K, const clang::Decl* D)
          : m_Kind(K), m_DGR(const_cast<clang::Decl*>(D)) {}
      bool operator==(const DelayedCallInfo& other) const {
        if (m_Kind != other.m_Kind)
          return false;

        if (std::distance(m_DGR.begin(), m_DGR.end()) !=
            std::distance(other.m_DGR.begin(), other.m_DGR.end()))
          return false;

        clang::Decl* const* first1 = m_DGR.begin();
        clang::Decl* const* first2 = other.m_DGR.begin();
        clang::Decl* const* last1 = m_DGR.end();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        for (; first1 != last1; ++first1, ++first2)
          if (!(*first1 == *first2))
            return false;
        return true;
      }
    };
    /// The calls to the main action which clad delayed and will dispatch at
    /// then end of the translation unit.
    std::deque<DelayedCallInfo> m_DelayedCalls;
    /// The default clang consumers which are called after clad is done.
    std::unique_ptr<clang::MultiplexConsumer> m_Multiplexer;

    /// Have we processed all delayed calls.
    unsigned m_MultiplexerProcessedDelayedCallsIdx = 0;

    /// The Sema::TUScope to restore in CladPlugin::HandleTranslationUnit.
    clang::Scope* m_StoredTUScope = nullptr;

  public:
    CladPlugin(clang::CompilerInstance& CI, DifferentiationOptions& DO);
    ~CladPlugin() override;
    // ASTConsumer
    void Initialize(clang::ASTContext& Context) override;
    void HandleCXXStaticMemberVarInstantiation(clang::VarDecl* D) override {
      AppendDelayed({CallKind::HandleCXXStaticMemberVarInstantiation, D});
    }
    bool HandleTopLevelDecl(clang::DeclGroupRef D) override {
      if (D.isSingleDecl())
        if (auto* FD = llvm::dyn_cast<clang::FunctionDecl>(D.getSingleDecl()))
          // If we build the derivative in a non-standard (with no Multiplexer)
          // setup, we exit early to give control to the non-standard setup for
          // code generation.
          // FIXME: This should go away if Cling starts using the clang driver.
          if (!m_Multiplexer && m_Scheduler &&
              (m_Scheduler->getDerivedFns().IsCladDerivative(FD) ||
               m_Scheduler->getDerivedFns().IsCustomDerivative(FD)))
            return true;

      HandleTopLevelDeclForClad(D);
      AppendDelayed({CallKind::HandleTopLevelDecl, D});
      return true; // happyness, continue parsing
    }
    void HandleInlineFunctionDefinition(clang::FunctionDecl* D) override {
      AppendDelayed({CallKind::HandleInlineFunctionDefinition, D});
    }
    void HandleInterestingDecl(clang::DeclGroupRef D) override {
      AppendDelayed({CallKind::HandleInterestingDecl, D});
    }
    void HandleTagDeclDefinition(clang::TagDecl* D) override {
      AppendDelayed({CallKind::HandleTagDeclDefinition, D});
    }
    void HandleTagDeclRequiredDefinition(const clang::TagDecl* D) override {
      AppendDelayed({CallKind::HandleTagDeclRequiredDefinition, D});
    }
    void
    HandleCXXImplicitFunctionInstantiation(clang::FunctionDecl* D) override {
      AppendDelayed({CallKind::HandleCXXImplicitFunctionInstantiation, D});
    }
    void HandleTopLevelDeclInObjCContainer(clang::DeclGroupRef D) override {
      AppendDelayed({CallKind::HandleTopLevelDeclInObjCContainer, D});
    }
    void HandleImplicitImportDecl(clang::ImportDecl* D) override {
      AppendDelayed({CallKind::HandleImplicitImportDecl, D});
    }
    void CompleteTentativeDefinition(clang::VarDecl* D) override {
      AppendDelayed({CallKind::CompleteTentativeDefinition, D});
    }
#if CLANG_VERSION_MAJOR < 19
    void CompleteExternalDeclaration(clang::VarDecl* D) override {
      AppendDelayed({CallKind::CompleteExternalDeclaration, D});
    }
#else
    void CompleteExternalDeclaration(clang::DeclaratorDecl* D) override {
      AppendDelayed({CallKind::CompleteExternalDeclaration, D});
    }
#endif
    void AssignInheritanceModel(clang::CXXRecordDecl* D) override {
      AppendDelayed({CallKind::AssignInheritanceModel, D});
    }
    void HandleVTable(clang::CXXRecordDecl* D) override {
      AppendDelayed({CallKind::HandleVTable, D});
    }

    // Not delayed.
    void HandleTranslationUnit(clang::ASTContext& C) override;

    // No need to handle the listeners, they will be handled non-delayed by
    // the parent multiplexer.
    //
    // clang::ASTMutationListener *GetASTMutationListener() override;
    // clang::ASTDeserializationListener *GetASTDeserializationListener()
    // override;
    void PrintStats() override;

    bool shouldSkipFunctionBody(clang::Decl* D) override {
      return m_Multiplexer ? m_Multiplexer->shouldSkipFunctionBody(D) : true;
    }

    // SemaConsumer
    void InitializeSema(clang::Sema& S) override {
      // We are also a ExternalSemaSource.
      // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
      S.addExternalSource(new CladExternalSource()); // Owned by Sema.
      m_StoredTUScope = S.TUScope;
      AppendDelayed({CallKind::InitializeSema, nullptr});
    }
    void ForgetSema() override {
      // ForgetSema is called in the destructor of Sema which is much later
      // than where we can process anything. We can't delay this call.
      if (m_Multiplexer)
        m_Multiplexer->ForgetSema();
    }

    // FIXME: We should hide ProcessDiffRequest when we implement proper
    // handling of the differentiation plans.
    clang::FunctionDecl* ProcessDiffRequest(DiffRequest& request);

  private:
    DiffScheduler& getScheduler();
    void AppendDelayed(DelayedCallInfo DCI) {
      // Incremental processing handles the translation unit in chunks and it is
      // expected to have multiple calls to this functionality.
      assert((!m_MultiplexerProcessedDelayedCallsIdx ||
              m_CI.getPreprocessor().isIncrementalProcessingEnabled()) &&
             "Must start from index 0!");
      m_DelayedCalls.push_back(DCI);
#if CLANG_VERSION_MAJOR < 22
      // If we are in repl mode we might have an error or an undo operation
      // which discards the set of declarations that were collected already.
      // Clad should discard them, too, or it would replay them to the
      // multiplexer over a broken AST. Clang >= 22 is not affected
      // (llvm/llvm-project#182044).
      if (m_CI.getPreprocessor().isIncrementalProcessingEnabled() &&
          m_CI.getDiagnostics().hasErrorOccurred())
        m_MultiplexerProcessedDelayedCallsIdx = m_DelayedCalls.size();
#endif
    }
    void FinalizeTranslationUnit();
    void SendToMultiplexer();
    bool CheckBuiltins();
    void SetRequestOptions(RequestOptions& opts) const;
    /// Shows what a generated function looks like as text, and where inside
    /// that text each of its statements sits.
    void dumpGeneratedSource(clang::Decl* D);
    /// Reports what an analysis left behind, at the generated code it left it
    /// in. Renders that code only if there is something to report.
    void emitAnalysisRemarks(clang::Decl* D, const DiffRequest& request);

    void ProcessTopLevelDecl(clang::Decl* D) {
      DelayedCallInfo DCI{CallKind::HandleTopLevelDecl, D};
      assert(!llvm::is_contained(m_DelayedCalls, DCI) && "Already exists!");
      AppendDelayed(DCI);
      // We could not delay the process due to some strange way of
      // initialization, inform the consumers now.
      if (!m_Multiplexer)
        m_CI.getASTConsumer().HandleTopLevelDecl(DCI.m_DGR);
    }
    void HandleTopLevelDeclForClad(clang::DeclGroupRef DGR);
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
      CreateASTConsumer(clang::CompilerInstance& CI,
                        llvm::StringRef InFile) override {
        return std::unique_ptr<clang::ASTConsumer>(new ConsumerType(CI, m_DO));
      }

      bool ParseArgs(const clang::CompilerInstance& CI,
                     const std::vector<std::string>& args) override {
        for (unsigned i = 0, e = args.size(); i != e; ++i) {
          if (args[i] == "-fdump-source-fn") {
            m_DO.DumpSourceFn = true;
          } else if (args[i] == "-fdump-source-fn-ast") {
            m_DO.DumpSourceFnAST = true;
          } else if (args[i] == "-fdump-derived-fn") {
            m_DO.DumpDerivedFn = true;
          } else if (args[i] == "-fdump-derived-fn-ast") {
            m_DO.DumpDerivedAST = true;
          } else if (args[i] == "-fdump-generated-source") {
            m_DO.DumpGeneratedSource = true;
          } else if (AnalysisFlagResult R = remarkAnalysisByName(m_DO, args[i]);
                     R != AnalysisFlagResult::NotMine) {
            if (R == AnalysisFlagResult::Error)
              return false;
          } else if (args[i] == "-fgenerate-source-file") {
            m_DO.GenerateSourceFile = true;
          } else if (args[i] == "-fno-validate-clang-version") {
            m_DO.ValidateClangVersion = false;
          } else if (setAnalysisFromFlag(m_DO, args[i])) {
            // -enable-tbr, -disable-va, and the rest of Analyses.def.
          } else if (AnalysisFlagResult R = setAnalysisByName(m_DO, args[i]);
                     R != AnalysisFlagResult::NotMine) {
            if (R == AnalysisFlagResult::Error)
              return false;
          } else if (args[i] == "-fcustom-estimation-model") {
            llvm::errs() << "`-fcustom-estimation-model` is deprecated.";
            ++i;
            return false;
          } else if (args[i] == "-fprint-num-diff-errors") {
            m_DO.PrintNumDiffErrorInfo = true;
          } else if (args[i] == "-fclad-porting-hints") {
            m_DO.EmitPortingHints = true;
          } else if (args[i] == "-help") {
            // Print some help info.
            // CI.getFrontendOpts().ShowHelp does not give us control.
            llvm::errs()
                << "Options specific to Clad (preceded by -plugin-arg-clad):"
                << "-fdump-source-fn - Prints out the source code of the "
                   "function.\n"
                << "-fdump-source-fn-ast - Prints out the AST of the "
                   "function.\n"
                << "-fdump-derived-fn - Prints out the source code of the "
                   "derivative.\n"
                << "-fdump-derived-fn-ast - Prints out the AST of the "
                   "derivative.\n"
                << "-fdump-generated-source - For each derivative, prints "
                   "the text clad renders it as and, for every statement in "
                   "it, the line and column that text occupies. Diagnostics "
                   "about generated code point into that rendering.\n"
                << "-Rclad-analysis=<name> - Reports what the named analysis "
                   "left in the derivative, as remarks pointed at the "
                   "generated code itself. The analogue of -Rpass-missed, "
                   "which cannot reach an AST-level plugin.\n"
                << "-fgenerate-source-file - Produces a file containing the "
                   "derivatives.\n"
                << "-fno-validate-clang-version - Disables the validation of "
                   "the clang version.\n"
                << "-fcustom-estimation-model - allows user to send in a "
                   "shared object to use as the custom estimation model.\n"
                << "-fprint-num-diff-errors - allows users to print the "
                   "calculated numerical diff errors, this flag is overriden "
                   "by -DCLAD_NO_NUM_DIFF.\n"
                << "-fclad-porting-hints - When clad has no custom derivative "
                   "for a function defined outside the main source file and "
                   "falls back to differentiating its definition, emit a "
                   "remark naming the expected custom-derivative signature and "
                   "the non-differentiable marker. Useful when teaching clad "
                   "about a new library.\n";

            llvm::errs()
                << "Each analysis below is optional: it changes the code clad "
                   "generates, never the values that code computes. Enabling "
                   "one asks clad to prove more and store less; disabling one "
                   "falls back to the conservative derivative.\n"
                << "-fenable-analysis=<name> / -fdisable-analysis=<name> - "
                   "Turns one analysis on or off; the last such option on the "
                   "command line decides. -fdisable-analysis=all turns off "
                   "every analysis clad has, asking for the most conservative "
                   "derivative it can produce.\n";
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                     \
      llvm::errs()                                                             \
          << "-enable-" Legacy " / -disable-" Legacy " - Turns the " Name      \
             " analysis on or off for the whole translation unit, unless an "  \
             "individual request specifies otherwise. It " Desc ". Default: "  \
          << ((Default) ? "on" : "off") << ".\n";
#include "clad/Differentiator/Analyses.def"

            llvm::errs() << "-help - Prints out this screen.\n\n";
          } else if (args[i] == "-version" || args[i] == "-v") {
            // CI.getFrontendOpts().ShowHelp does not give us control.
            llvm::errs() << getCladFullVersion() << "\n";
          } else {
            llvm::errs() << "clad: Error: invalid option " << args[i] << "\n";
            return false; // Tells clang not to create the plugin.
          }
        }
        if (m_DO.ValidateClangVersion != false) {
          if (!checkClangVersion())
            return false;
        }
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                     \
      if (m_DO.Enable##Id##Analysis && m_DO.Disable##Id##Analysis) {           \
        llvm::errs() << "clad: Error: -enable-" Legacy " and -disable-" Legacy \
                        " cannot be used together.\n";                         \
        return false;                                                          \
      }
#include "clad/Differentiator/Analyses.def"
        return true;
      }

      PluginASTAction::ActionType getActionType() override {
        return AddAfterMainAction;
      }
    };
  } // end namespace plugin
} // end namespace clad

#endif // CLAD_CLANG_PLUGIN
