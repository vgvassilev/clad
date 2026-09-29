#include "clad/Differentiator/MultiplexExternalRMVSource.h"
#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"

using namespace clad;
using namespace clang;

namespace {

// A mock external source to verify that methods are forwarded correctly.
class MockExternalRMVSource : public ExternalRMVSource {
public:
  int InitialiseRMVCount = 0;
  int ForgetRMVCount = 0;
  int ActOnStartOfDeriveCount = 0;
  int ActBeforeCreatingDerivedFnParamTypesCount = 0;
  int ExtraParamsAdded = 0;
  int ActAfterCreatingDerivedFnParamsCount = 0;

  // We can't actually pass a real ReverseModeVisitor easily in a unit test,
  // but since we only check the multiplexing logic, we can cast a dummy pointer.
  void InitialiseRMV(ReverseModeVisitor& RMV) override {
    InitialiseRMVCount++;
  }

  void ForgetRMV() override {
    ForgetRMVCount++;
  }

  void ActOnStartOfDerive() override {
    ActOnStartOfDeriveCount++;
  }

  void ActBeforeCreatingDerivedFnParamTypes(unsigned& numExtraParams) override {
    ActBeforeCreatingDerivedFnParamTypesCount++;
    numExtraParams += ExtraParamsAdded;
  }
  
  void ActAfterCreatingDerivedFnParams(
      llvm::SmallVectorImpl<clang::ParmVarDecl*>& params) override {
    ActAfterCreatingDerivedFnParamsCount++;
  }
};

} // namespace

TEST(MultiplexExternalRMVSourceTest, ForwardsCallsToMultipleSources) {
  MultiplexExternalRMVSource Multiplexer;

  MockExternalRMVSource Source1;
  Source1.ExtraParamsAdded = 2;

  MockExternalRMVSource Source2;
  Source2.ExtraParamsAdded = 3;

  Multiplexer.AddSource(Source1);
  Multiplexer.AddSource(Source2);

  // Test forwarding of simple methods.
  // We use a fake pointer for RMV since it's just forwarded by reference.
  ReverseModeVisitor* fakeRMV = reinterpret_cast<ReverseModeVisitor*>(0x1234);
  Multiplexer.InitialiseRMV(*fakeRMV);
  
  EXPECT_EQ(Source1.InitialiseRMVCount, 1);
  EXPECT_EQ(Source2.InitialiseRMVCount, 1);

  Multiplexer.ForgetRMV();
  EXPECT_EQ(Source1.ForgetRMVCount, 1);
  EXPECT_EQ(Source2.ForgetRMVCount, 1);

  Multiplexer.ActOnStartOfDerive();
  EXPECT_EQ(Source1.ActOnStartOfDeriveCount, 1);
  EXPECT_EQ(Source2.ActOnStartOfDeriveCount, 1);

  // Test forwarding of methods with by-reference accumulated state.
  unsigned numExtraParams = 0;
  Multiplexer.ActBeforeCreatingDerivedFnParamTypes(numExtraParams);
  EXPECT_EQ(Source1.ActBeforeCreatingDerivedFnParamTypesCount, 1);
  EXPECT_EQ(Source2.ActBeforeCreatingDerivedFnParamTypesCount, 1);
  EXPECT_EQ(numExtraParams, 5); // 2 + 3

  llvm::SmallVector<clang::ParmVarDecl*, 4> params;
  Multiplexer.ActAfterCreatingDerivedFnParams(params);
  EXPECT_EQ(Source1.ActAfterCreatingDerivedFnParamsCount, 1);
  EXPECT_EQ(Source2.ActAfterCreatingDerivedFnParamsCount, 1);
}
