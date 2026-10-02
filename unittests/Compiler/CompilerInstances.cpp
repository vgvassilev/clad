#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/DynamicLibrary.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

namespace {
class DerivedFunctionConsumer : public clang::ASTConsumer {
  unsigned& Seen;

public:
  explicit DerivedFunctionConsumer(unsigned& seen) : Seen(seen) {}
  void HandleTranslationUnit(clang::ASTContext& C) override {
    for (const clang::Decl* D : C.getTranslationUnitDecl()->decls()) {
      const auto* FD = llvm::dyn_cast<clang::FunctionDecl>(D);
      if (!FD || !FD->getIdentifier() || !FD->hasBody())
        continue;
      if (FD->getName() == "loop_grad")
        Seen |= 1;
      if (FD->getName() == "square_dvec")
        Seen |= 2;
      if (FD->getName() == "output_jac")
        Seen |= 4;
    }
  }
};

class DerivativeCheckingAction : public clang::SyntaxOnlyAction {
  unsigned& Seen;

public:
  explicit DerivativeCheckingAction(unsigned& seen) : Seen(seen) {}
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance&, llvm::StringRef) override {
    return std::make_unique<DerivedFunctionConsumer>(Seen);
  }
};
} // namespace

TEST(CompilerInstances, LookupCachesStayWithTheirSema) {
  std::string error;
  ASSERT_FALSE(llvm::sys::DynamicLibrary::LoadLibraryPermanently(
      CLAD_TEST_PLUGIN, &error))
      << error;
  const char* code = R"(
#include "clad/Differentiator/Differentiator.h"
double loop(double x, int n) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    r += x * x;
    x *= 2;
  }
  return r;
}
double square(double x) { return x * x; }
void output(double x, double out[]) { out[0] = x * x; }
void request() {
  clad::gradient(loop);
  clad::differentiate<clad::opts::vector_mode>(square);
  clad::jacobian(output);
}
)";
  std::vector<std::string> args = {"-std=c++17",
                                   "-DCLAD_NO_NUM_DIFF",
                                   std::string("-I") + CLAD_TEST_INCLUDE,
                                   std::string("-I") +
                                       CLAD_TEST_GENERATED_INCLUDE,
                                   "-Xclang",
                                   "-add-plugin",
                                   "-Xclang",
                                   "clad"};
  for (int i = 0; i < 2; ++i) {
    SCOPED_TRACE(i);
    unsigned Seen = 0;
    ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
        std::make_unique<DerivativeCheckingAction>(Seen), code, args));
    ASSERT_EQ(Seen, 7u);
  }
}
