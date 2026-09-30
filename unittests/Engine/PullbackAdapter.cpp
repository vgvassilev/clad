#include "../../tools/ClangPlugin.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/DiffScheduler.h"
#include "clad/Differentiator/Options.h"
#include "clad/Differentiator/ReverseModeVisitor.h"
#include "clad/Differentiator/VisitorBase.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/Sema.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/SaveAndRestore.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace clang;

class AdapterProbe : public clad::VisitorBase {
public:
  AdapterProbe(clad::DerivativeBuilder& B, const clad::DiffRequest& R)
      : VisitorBase(B, R) {}
  clad::DerivativeAndOverload Derive() override { return {}; }
  using VisitorBase::getCurrentScope;
  Scope* derivativeScope() const { return m_DerivativeFnScope; }
};

// Capture diagnostics without suppressing semantic checks in Sema. Suppressing
// all diagnostics can change whether Sema rejects a call, invalidating a test.
class DiagnosticCapture : public DiagnosticConsumer {
public:
  std::string Messages;
  void HandleDiagnostic(DiagnosticsEngine::Level Level,
                        const Diagnostic& Info) override {
    DiagnosticConsumer::HandleDiagnostic(Level, Info);
    llvm::SmallString<128> Message;
    Info.FormatDiagnostic(Message);
    Messages += Message.str().str() + "\n";
  }
};

class EngineContext {
  CompilerInstance& m_CI;
  clad::Options m_Options;
  clad::plugin::CladPlugin m_Plugin;
  clad::DiffInterval m_Interval;
  clad::DiffScheduler m_Scheduler;
  clad::DerivativeBuilder m_Builder;
  Scope m_TUScope;
  clad::DiffRequest m_Request;
  AdapterProbe m_Probe;
  llvm::SaveAndRestore<Scope*> m_SaveScope;

public:
  explicit EngineContext(CompilerInstance& CI)
      : m_CI(CI), m_Plugin(CI, m_Options),
        m_Scheduler(CI.getSema(), m_Options, m_Interval),
        m_Builder(CI.getSema(), m_Plugin, m_Scheduler),
        m_TUScope(nullptr, Scope::DeclScope, CI.getDiagnostics()),
        m_Probe(m_Builder, m_Request),
        m_SaveScope(m_Probe.getCurrentScope(), &m_TUScope) {
    m_TUScope.setEntity(CI.getASTContext().getTranslationUnitDecl());
  }

  Sema& sema() { return m_CI.getSema(); }
  FunctionDecl* function(llvm::StringRef Name, DeclContext* DC = nullptr) {
    if (!DC)
      DC = m_CI.getASTContext().getTranslationUnitDecl();
    for (Decl* D : DC->decls()) {
      if (auto* FD = dyn_cast<FunctionDecl>(D)) {
        if (FD->getQualifiedNameAsString() == Name)
          return FD;
      } else if (auto* NS = dyn_cast<NamespaceDecl>(D)) {
        if (FunctionDecl* FD = function(Name, NS))
          return FD;
      } else if (auto* RD = dyn_cast<RecordDecl>(D)) {
        if (FunctionDecl* FD = function(Name, RD))
          return FD;
      }
    }
    return nullptr;
  }

  void request(FunctionDecl* Original, bool Duplicate = false) {
    m_Request.Function = Original;
    m_Request.BaseFunctionName = Original ? Original->getNameAsString() : "";
    m_Request.Mode = clad::DiffMode::pullback;
    m_Request.CallUpdateRequired = true;
    m_Request.CustomDerivative = nullptr;
    m_Request.PullbackStateParam = QualType();
    m_Request.DVI.clear();
    if (Original && Original->getNumParams()) {
      m_Request.DVI.push_back(Original->getParamDecl(0));
      if (Duplicate)
        m_Request.DVI.push_back(Original->getParamDecl(0));
    }
  }

  FunctionDecl* adapter(FunctionDecl* Derivative, bool Success,
                        bool Custom = false,
                        clad::VisitorBase::OverloadKind Kind =
                            clad::VisitorBase::OverloadKind::PullbackCustom,
                        llvm::StringRef ExpectedDiagnostic = {}) {
    Sema& S = sema();
    SCOPED_TRACE(Derivative ? Derivative->getQualifiedNameAsString()
                            : "<missing derivative>");
    if (Custom)
      m_Request.CustomDerivative =
          S.BuildDeclRefExpr(Derivative, Derivative->getType(), VK_LValue,
                             Derivative->getLocation());
    if (Kind == clad::VisitorBase::OverloadKind::Default)
      m_Request.Mode = clad::DiffMode::reverse;
    FunctionDecl* Result = observeSema(
        [&] { return m_Probe.CreateDerivativeOverload(Derivative, Kind); },
        ExpectedDiagnostic);
    EXPECT_EQ(Result != nullptr, Success);
    if (Result) {
      EXPECT_NE(Result->getBody(), nullptr);
      std::set<std::string> Names;
      for (const ParmVarDecl* P : Result->parameters())
        EXPECT_TRUE(Names.insert(P->getNameAsString()).second);
    }
    return Result;
  }

  // Bad planner input must be rejected before reverse generation publishes a
  // declaration, and the compiler must remain usable for another request.
  void rejectGeneratedLayout() {
    auto Result = observeSema(
        [&] {
          clad::ReverseModeVisitor V(m_Builder, m_Request);
          return V.Derive();
        },
        "generated pullback parameter layout does not match its function type");
    EXPECT_EQ(Result.derivative, nullptr);
    EXPECT_EQ(Result.overload, nullptr);
  }

  void deriveDeclaration() {
    m_Request.DeclarationOnly = true;
    auto Result = observeSema(
        [&] {
          clad::ReverseModeVisitor V(m_Builder, m_Request);
          return V.Derive();
        },
        {});
    auto* FD = dyn_cast_or_null<FunctionDecl>(Result.derivative);
    ASSERT_NE(FD, nullptr);
    EXPECT_EQ(FD->getNumParams(), 3U);
    EXPECT_EQ(FD->getBody(), nullptr);
    EXPECT_EQ(Result.overload, nullptr);
  }

  void rejectCustom(ValueDecl* Derivative, bool WithState,
                    llvm::StringRef ExpectedDiagnostic) {
    Sema& S = sema();
    m_Request.CustomDerivative =
        S.BuildDeclRefExpr(Derivative, Derivative->getType(), VK_LValue,
                           Derivative->getLocation());
    if (WithState) {
      auto* FD = cast<FunctionDecl>(Derivative);
      m_Request.PullbackStateParam =
          FD->getParamDecl(FD->getNumParams() - 1)->getType();
    }
    auto Result = observeSema([&] { return m_Builder.Derive(m_Request); },
                              ExpectedDiagnostic);
    EXPECT_EQ(Result.derivative, nullptr);
    EXPECT_EQ(Result.overload, nullptr);
  }
  ValueDecl* value(llvm::StringRef Name) {
    for (Decl* D : m_CI.getASTContext().getTranslationUnitDecl()->decls())
      if (auto* VD = dyn_cast<ValueDecl>(D))
        if (VD->getName() == Name)
          return VD;
    return nullptr;
  }

  bool needsAdapter() { return m_Probe.PullbackNeedsOverload(); }
  std::string name(bool CallUpdate = true) {
    m_Request.CallUpdateRequired = CallUpdate;
    return m_Request.ComputeDerivativeName();
  }
  void selectAll() {
    m_Request.DVI.clear();
    for (const ParmVarDecl* P : m_Request.Function->parameters())
      m_Request.DVI.push_back(P);
  }
  void nullSelection() { m_Request.DVI.push_back(nullptr); }

private:
  template <typename Action>
  auto observeSema(Action A,
                   llvm::StringRef ExpectedDiagnostic) -> decltype(A()) {
    Sema& S = sema();
    DeclContext* ContextBefore = S.CurContext;
    Scope* ScopeBefore = m_Probe.getCurrentScope();
    Scope* DerivativeScopeBefore = m_Probe.derivativeScope();
    const std::size_t FunctionsBefore = S.getFunctionScopes().size();
    DiagnosticConsumer* PreviousClient = S.getDiagnostics().getClient();
    auto OwnedClient = S.getDiagnostics().takeClient();
    DiagnosticCapture Capture;
    S.getDiagnostics().setClient(&Capture, /*ShouldOwnClient=*/false);
    auto Result = A();
    EXPECT_EQ(S.CurContext, ContextBefore);
    EXPECT_EQ(m_Probe.getCurrentScope(), ScopeBefore);
    EXPECT_EQ(m_Probe.derivativeScope(), DerivativeScopeBefore);
    EXPECT_EQ(S.getFunctionScopes().size(), FunctionsBefore);
    if (!ExpectedDiagnostic.empty())
      EXPECT_NE(Capture.Messages.find(ExpectedDiagnostic.str()),
                std::string::npos)
          << Capture.Messages;
    else
      EXPECT_EQ(Capture.getNumErrors(), 0U);
    // These are expected internal errors, not errors in the source being
    // parsed. Restore the client's ownership and error state before allowing
    // the frontend to finish or exercising a subsequent valid request.
    S.getDiagnostics().Reset();
    const bool OwnedPreviousClient = bool(OwnedClient);
    S.getDiagnostics().setClient(OwnedClient ? OwnedClient.release()
                                             : PreviousClient,
                                 /*ShouldOwnClient=*/OwnedPreviousClient);
    return Result;
  }
};

using Check = std::function<void(EngineContext&)>;
class TestConsumer : public ASTConsumer {
  CompilerInstance& m_CI;
  Check m_Check;

public:
  TestConsumer(CompilerInstance& CI, Check C)
      : m_CI(CI), m_Check(std::move(C)) {}
  void HandleTranslationUnit(ASTContext&) override {
    EngineContext E(m_CI);
    m_Check(E);
  }
};
class TestAction : public ASTFrontendAction {
  Check m_Check;

public:
  explicit TestAction(Check C) : m_Check(std::move(C)) {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance& CI,
                                                 llvm::StringRef) override {
    return std::make_unique<TestConsumer>(CI, m_Check);
  }
};

const char* const Source = R"cpp(
namespace clad { template <class T> struct pullback_state {}; }
// The statically linked plugin is also an automatic frontend action. Complete
// this fixture while parsing, before its delayed consumer replay finishes;
// our post-parse Sema probe must not instantiate a new class in that phase.
clad::pullback_state<double> completed_state;
void with_state(double x, double seed, double* dx, clad::pullback_state<double> state);
extern void (*indirect)(double x, double seed, double* dx);
struct Box { explicit Box(double x); };
void constructor_derivative(double x, Box* d_this, double* dx);
double primal(double x) { return x*x; }
namespace outer { namespace inner { double namespaced(double x); } }
double multi(double x, double y);
double collision_primal(double _d_y, double _d_y0);
void collision_derivative(double x, double y, double _d_y, double* d_x);
double variadic(double x, ...);
void good(double x, double seed, double* dx);
void variadic_derivative(double x, ...);
double nonvoid(double x, double seed, double* dx);
void too_short(double x);
void wrong_type(int x, double seed, double* dx);
void duplicated(double x, double seed, double* dx, double* duplicate_dx);
void no_arguments();
void scalar_legacy(double x, double dx);
void pointer_legacy(double x, double* dx);
double pointer_primal(double* x) { return *x; }
void qualified(const double* x, double seed, double* dx);
double deep_primal(double** x) { return **x; }
void unsafe_qualified(const double** x, double seed, double** dx);
)cpp";

void run(Check C, const char* Code = Source,
         const std::vector<std::string>& Args = {"-std=c++17"}) {
  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(
      std::make_unique<TestAction>(std::move(C)), Code, Args));
}

TEST(PullbackAdapter, RejectsMissingAndUnsupportedCallees) {
  run([](EngineContext& E) {
    E.request(/*Original=*/nullptr);
    E.adapter(/*Derivative=*/nullptr, /*Success=*/false);
    E.request(E.function("primal"));
    E.adapter(/*Derivative=*/nullptr, /*Success=*/false);
    E.adapter(E.function("variadic_derivative"), /*Success=*/false);
    E.adapter(E.function("nonvoid"), /*Success=*/false);
    E.request(E.function("variadic"));
    E.adapter(E.function("good"), /*Success=*/false);
  });
}

TEST(PullbackAdapter, RejectsCFunctionWithoutPrototype) {
  run(
      [](EngineContext& E) {
        E.request(E.function("primal"));
        E.adapter(E.function("good"), /*Success=*/false);
      },
      /*Code=*/"double primal(); void good();", {"-xc", "-std=c11"});
}

TEST(PullbackAdapter, ValidatesTypedLayoutBeforePublishing) {
  run([](EngineContext& E) {
    E.request(E.function("primal"));
    E.adapter(E.function("too_short"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::PullbackCustom,
              "unexpected derivative parameter count");
    E.adapter(E.function("wrong_type"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::PullbackCustom,
              "unexpected derivative parameter type");
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, RejectsUnsafeQualificationAndAcceptsSafeConversion) {
  run([](EngineContext& E) {
    E.request(E.function("deep_primal"));
    E.adapter(E.function("unsafe_qualified"), /*Success=*/false,
              /*Custom=*/true, clad::VisitorBase::OverloadKind::PullbackCustom,
              "unexpected derivative parameter type");
    E.request(E.function("pointer_primal"));
    E.adapter(E.function("qualified"), /*Success=*/true, /*Custom=*/true);
  });
}

TEST(PullbackAdapter, PrototypeFailureRestoresSemaForNextRequest) {
  run([](EngineContext& E) {
    E.request(E.function("primal"), /*Duplicate=*/true);
    EXPECT_TRUE(E.needsAdapter());
    E.adapter(E.function("duplicated"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::PullbackCustom,
              "did not consume the complete derivative parameter layout");
    E.request(E.function("primal"));
    EXPECT_FALSE(E.needsAdapter());
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, PublicCustomStateIsRejectedWithoutCallContext) {
  run([](EngineContext& E) {
    E.request(E.function("primal"));
    E.rejectCustom(E.function("with_state"), /*WithState=*/true,
                   "public pullback interface cannot expose a custom");
    E.request(E.function("primal"));
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, InvalidCustomCallAndIndirectCalleeAreRejected) {
  run([](EngineContext& E) {
    E.request(E.function("primal"));
    E.rejectCustom(E.function("too_short"), /*WithState=*/false,
                   "too many arguments");
    E.request(E.function("primal"));
    E.rejectCustom(E.value("indirect"), /*WithState=*/false, {});
    E.request(E.function("primal"));
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, CustomAdapterFailurePropagatesWithoutPublishing) {
  run([](EngineContext& E) {
    E.request(E.function("primal"), /*Duplicate=*/true);
    E.rejectCustom(E.function("duplicated"), /*WithState=*/false,
                   "did not consume the complete derivative parameter layout");
    E.request(E.function("primal"));
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, ConstructorAdapterKeepsObjectCotangentSlot) {
  run([](EngineContext& E) {
    E.request(E.function("Box::Box"));
    FunctionDecl* Wrapper =
        E.adapter(E.function("constructor_derivative"), /*Success=*/true);
    ASSERT_NE(Wrapper, nullptr);
    EXPECT_EQ(Wrapper->getNumParams(), 3U);
    EXPECT_TRUE(cast<CXXMethodDecl>(Wrapper)->isStatic());
    EXPECT_EQ(Wrapper->getParamDecl(1)->getType().getCanonicalType(),
              E.function("constructor_derivative")
                  ->getParamDecl(1)
                  ->getType()
                  .getCanonicalType());
  });
}

TEST(PullbackAdapter, GeneratedLayoutFailurePreservesSema) {
  run([](EngineContext& E) {
    E.request(E.function("primal"), /*Duplicate=*/true);
    E.rejectGeneratedLayout();
    E.request(E.function("primal"));
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, NamespaceFailureRestoresScopesForNextRequest) {
  run([](EngineContext& E) {
    E.request(E.function("outer::inner::namespaced"), /*Duplicate=*/true);
    E.adapter(E.function("duplicated"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::PullbackCustom,
              "did not consume the complete derivative parameter layout");
    E.request(E.function("primal"));
    E.adapter(E.function("good"), /*Success=*/true);
  });
}

TEST(PullbackAdapter, NullSelectionRequiresAdapter) {
  run([](EngineContext& E) {
    E.request(E.function("primal"));
    E.nullSelection();
    EXPECT_TRUE(E.needsAdapter());
  });
}

TEST(PullbackAdapter, LegacyOverloadsRejectInvalidCount) {
  run([](EngineContext& E) {
    E.request(E.function("primal"));
    E.adapter(E.function("no_arguments"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::Default,
              "unexpected derivative parameter count for overload");
    E.adapter(E.function("good"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::Default,
              "unexpected derivative parameter count for overload");
    // A valid AST can still describe an incompatible legacy adjoint ABI.
    // Casting the wrapper's void* slot to a scalar must fail inside the body,
    // unwind both frames, and leave Sema usable for the next valid wrapper.
    E.adapter(E.function("scalar_legacy"), /*Success=*/false, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::Default, "not allowed");
    E.adapter(E.function("pointer_legacy"), /*Success=*/true, /*Custom=*/false,
              clad::VisitorBase::OverloadKind::Default);
  });
}

TEST(PullbackAdapter, NumericSuffixAvoidsAllPrimalNames) {
  run([](EngineContext& E) {
    E.request(E.function("collision_primal"));
    FunctionDecl* Wrapper =
        E.adapter(E.function("collision_derivative"), /*Success=*/true);
    ASSERT_NE(Wrapper, nullptr);
    EXPECT_EQ(Wrapper->getParamDecl(2)->getNameAsString(), "_d_y1");
  });
}

TEST(PullbackAdapter, FullPartialAndNestedNamesRemainDistinct) {
  run([](EngineContext& E) {
    E.request(E.function("primal"));
    EXPECT_EQ(E.name(), "primal_pullback");
    E.request(E.function("multi"));
    EXPECT_EQ(E.name(), "multi_pullback_0");
    EXPECT_EQ(E.name(/*CallUpdate=*/false), "multi_pullback");
    E.selectAll();
    EXPECT_EQ(E.name(), "multi_pullback");
  });
}

TEST(PullbackAdapter, NullAndNonStandardNatAreNotReserved) {
  run(
      [](EngineContext& E) {
        EXPECT_FALSE(clad::utils::isStdNATType(QualType(), E.sema()));
        EXPECT_FALSE(clad::utils::isStdNATType(
            E.function("tag")->getParamDecl(0)->getType(), E.sema()));
      },
      /*Code=*/"struct __nat {}; double tag(__nat x);");
}

TEST(PullbackAdapter, StdInlineNamespaceNatIsReserved) {
  run(
      [](EngineContext& E) {
        EXPECT_TRUE(clad::utils::isStdNATType(
            E.function("tag")->getParamDecl(1)->getType(), E.sema()));
        E.request(E.function("tag"));
        // The legacy sentinel terminates the primal list; it is not a user
        // input.
        EXPECT_FALSE(E.needsAdapter());
        E.deriveDeclaration();
        E.adapter(E.function("tag_derivative"), /*Success=*/true);
      },
      /*Code=*/R"cpp(
namespace std { inline namespace abi { struct __nat {}; } }
double tag(double x, std::__nat sentinel);
void tag_derivative(double x, double seed, double* dx);
)cpp");
}
} // namespace
