//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_CLANG_PLUGIN
#define CLAD_CLANG_PLUGIN

#include "DerivedFnInfo.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/Version.h"

#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Sema/SemaConsumer.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Timer.h"

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

  bool checkClangVersion();
  /// This class is designed to store collection of `DerivedFnInfo` objects.
  /// It's purpose is to avoid repeated generation of same derivatives by
  /// making it possible to reuse previously computed derivatives.
  class DerivedFnCollector {
    using DerivedFns = llvm::SmallVector<DerivedFnInfo, 16>;
    /// Mapping to efficiently find out information about all the derivatives of
    /// a function.
    llvm::DenseMap<const clang::FunctionDecl*, DerivedFns> m_DerivedFnInfoCollection;

  public:
    /// Adds a derived function to the collection.
    void Add(const DerivedFnInfo& DFI);

    /// Finds a `DerivedFnInfo` object in the collection that satisfies the
    /// given differentiation request.
    DerivedFnInfo Find(const DiffRequest& request) const;

    bool IsDerivative(const clang::FunctionDecl* FD) const;

  private:
    /// Returns true if the collection already contains a `DerivedFnInfo`
    /// object that represents the same derivative object as the provided
    /// argument `DFI`.
    bool AlreadyExists(const DerivedFnInfo& DFI) const;
  };
  class CladTimerGroup {
    llvm::TimerGroup m_Tg;
    std::vector<std::unique_ptr<llvm::Timer>> m_Timers;

  public:
    CladTimerGroup();
    void StartNewTimer(llvm::StringRef TimerName, llvm::StringRef TimerDesc);
    void StopTimer();
  };

  namespace plugin {
    struct DifferentiationOptions {
    DifferentiationOptions()
        : DumpSourceFn(false), DumpSourceFnAST(false), DumpDerivedFn(false),
          DumpDerivedAST(false), GenerateSourceFile(false),
          ValidateClangVersion(true), EnableTBRAnalysis(false),
          DisableTBRAnalysis(false), CustomEstimationModel(false),
          PrintNumDiffErrorInfo(false) {}

    bool DumpSourceFn : 1;
    bool DumpSourceFnAST : 1;
    bool DumpDerivedFn : 1;
    bool DumpDerivedAST : 1;
    bool GenerateSourceFile : 1;
    bool ValidateClangVersion : 1;
    bool EnableTBRAnalysis : 1;
    bool DisableTBRAnalysis : 1;
    bool CustomEstimationModel : 1;
    bool PrintNumDiffErrorInfo : 1;
    std::string CustomModelName;
    };

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
    CladTimerGroup m_CTG;
    DerivedFnCollector m_DFC;
    DiffSchedule m_DiffSchedule;
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
#if CLANG_VERSION_MAJOR > 9
      CompleteExternalDeclaration,
#endif
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
    std::vector<DelayedCallInfo> m_DelayedCalls;
    /// The default clang consumers which are called after clad is done.
    std::unique_ptr<clang::MultiplexConsumer> m_Multiplexer;

    /// Have we processed all delayed calls.
    bool m_HasMultiplexerProcessedDelayedCalls = false;

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
#if CLANG_VERSION_MAJOR > 9
    void CompleteExternalDeclaration(clang::VarDecl* D) override {
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
      return m_Multiplexer->shouldSkipFunctionBody(D);
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
      m_Multiplexer->ForgetSema();
    }

    // FIXME: We should hide ProcessDiffRequest when we implement proper
    // handling of the differentiation plans.
    clang::FunctionDecl* ProcessDiffRequest(DiffRequest& request);

  private:
    void AppendDelayed(DelayedCallInfo DCI) {
      assert(!m_HasMultiplexerProcessedDelayedCalls);
      m_DelayedCalls.push_back(DCI);
    }
    void SendToMultiplexer();
    bool CheckBuiltins();
    void SetRequestOptions(RequestOptions& opts) const;

    void ProcessTopLevelDecl(clang::Decl* D) {
      DelayedCallInfo DCI{CallKind::HandleTopLevelDecl, D};
      assert(!llvm::is_contained(m_DelayedCalls, DCI) && "Already exists!");
      AppendDelayed(DCI);
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
          } else if (args[i] == "-fgenerate-source-file") {
            m_DO.GenerateSourceFile = true;
          } else if (args[i] == "-fno-validate-clang-version") {
            m_DO.ValidateClangVersion = false;
          } else if (args[i] == "-enable-tbr") {
            m_DO.EnableTBRAnalysis = true;
          } else if (args[i] == "-disable-tbr") {
            m_DO.DisableTBRAnalysis = true;
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
                << "-fno-validate-clang-version - Disables the validation of "
                   "the clang version.\n"
                << "-enable-tbr - Ensures that TBR analysis is enabled during "
                   "reverse-mode differentiation unless explicitly specified "
                   "in an individual request.\n"
                << "-disable-tbr - Ensures that TBR analysis is disabled "
                   "during reverse-mode differentiation unless explicitly "
                   "specified in an individual request.\n"
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
        if (m_DO.ValidateClangVersion != false) {
          if (!checkClangVersion())
            return false;
        }
        if (m_DO.EnableTBRAnalysis && m_DO.DisableTBRAnalysis) {
          llvm::errs() << "clad: Error: -enable-tbr and -disable-tbr cannot "
                          "be used together.\n";
          return false;
        }
        return true;
      }

      PluginASTAction::ActionType getActionType() override {
        return AddAfterMainAction;
      }
    };
  } // end namespace plugin
} // end namespace clad

#endif // CLAD_CLANG_PLUGIN
