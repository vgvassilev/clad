#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/DynamicLibrary.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <vector>

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
    ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
        std::make_unique<clang::SyntaxOnlyAction>(), code, args));
  }
}
