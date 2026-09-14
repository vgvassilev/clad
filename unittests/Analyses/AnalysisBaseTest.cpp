#include "AnalysisBase.h"

#include "gtest/gtest.h"

namespace {
class MergeAccess : public clad::AnalysisBase {
public:
  MergeAccess() : AnalysisBase(nullptr) {}
  using AnalysisBase::merge;
};

class AnalysisBaseMergeTest : public testing::Test {
protected:
  MergeAccess Analysis;
  // merge treats declaration keys as opaque; nullptr is a supported key.
  static constexpr const clang::VarDecl* Key = nullptr;

  clad::VarData bit(bool Value) {
    clad::VarData Data;
    Data.m_Type = clad::VarData::FUND_TYPE;
    Data.m_Val.m_FundData = Value;
    return Data;
  }

  void put(clad::VarsData& Data, bool Value) { Data[Key] = bit(Value); }
  bool value(clad::VarsData& Data) { return Data[Key].m_Val.m_FundData; }

  // 0 = absent, 1 = present/false, 2 = present/true.
  int visible(clad::VarsData* Data) {
    for (; Data; Data = Data->m_Prev) {
      auto Found = Data->find(Key);
      if (Found != Data->end())
        return Found->second.m_Val.m_FundData ? 2 : 1;
    }
    return 0;
  }
};

TEST_F(AnalysisBaseMergeTest, ReportsMissingArrayElement) {
  clad::VarData Target, Source;
  Target.m_Type = Source.m_Type = clad::VarData::ARR_TYPE;
  Target.m_Val.m_ArrData = std::make_unique<clad::ArrMap>();
  Source.m_Val.m_ArrData = std::make_unique<clad::ArrMap>();
  clad::ProfileID Default, Index;
  Index.AddInteger(1);
  (*Target.m_Val.m_ArrData)[Default] = bit(false);
  (*Source.m_Val.m_ArrData)[Default] = bit(false);
  (*Source.m_Val.m_ArrData)[Index] = bit(true);

  EXPECT_TRUE(Analysis.merge(Target, Source));
  EXPECT_TRUE(Target[Index]->m_Val.m_FundData);
  EXPECT_FALSE(Analysis.merge(Target, Source));
}

TEST_F(AnalysisBaseMergeTest, PreservesArrayDefaultForMissingElement) {
  clad::VarData Target, Source;
  Target.m_Type = Source.m_Type = clad::VarData::ARR_TYPE;
  Target.m_Val.m_ArrData = std::make_unique<clad::ArrMap>();
  Source.m_Val.m_ArrData = std::make_unique<clad::ArrMap>();
  clad::ProfileID Default, Index;
  Index.AddInteger(1);
  (*Target.m_Val.m_ArrData)[Default] = bit(true);
  (*Source.m_Val.m_ArrData)[Default] = bit(false);
  (*Source.m_Val.m_ArrData)[Index] = bit(false);

  EXPECT_FALSE(Analysis.merge(Target, Source));
  EXPECT_TRUE(Target[Index]->m_Val.m_FundData);
  EXPECT_FALSE(Analysis.merge(Target, Source));
}

TEST_F(AnalysisBaseMergeTest, ReportsNewVariableEvenWithFalseBit) {
  clad::VarsData Root, Target, Source;
  Target.m_Prev = Source.m_Prev = &Root;
  put(Source, false);
  EXPECT_TRUE(Analysis.merge(&Target, &Source));
  EXPECT_EQ(visible(&Target), 1);
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
}

TEST_F(AnalysisBaseMergeTest, AncestorRefreshDoesNotSignalWorklistGrowth) {
  clad::VarsData Root, Target, Source;
  Target.m_Prev = Source.m_Prev = &Root;
  put(Root, true);
  put(Target, false);
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
  EXPECT_TRUE(value(Target));
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
}

TEST_F(AnalysisBaseMergeTest, ReportsAncestorBitMergedIntoInherited) {
  clad::VarsData Root, Parent, Target, Source;
  Parent.m_Prev = Source.m_Prev = &Root;
  Target.m_Prev = &Parent;
  put(Root, true);
  put(Parent, false);
  EXPECT_TRUE(Analysis.merge(&Target, &Source));
  EXPECT_TRUE(value(Target));
  EXPECT_FALSE(value(Parent));
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
}

TEST_F(AnalysisBaseMergeTest, LocalTrueShadowsInheritedFalse) {
  clad::VarsData Root, Parent, Target, Source;
  Parent.m_Prev = Source.m_Prev = &Root;
  Target.m_Prev = &Parent;
  put(Root, false);
  put(Parent, false);
  put(Target, true);
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
  EXPECT_TRUE(value(Target));
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
  EXPECT_TRUE(value(Target));
}

TEST_F(AnalysisBaseMergeTest, LocalFalseShadowsInheritedTrue) {
  clad::VarsData Root, Parent, Target, Source;
  Parent.m_Prev = Source.m_Prev = &Root;
  Target.m_Prev = &Parent;
  put(Root, false);
  put(Parent, true);
  put(Target, false);

  EXPECT_FALSE(Analysis.merge(&Target, &Source));
  EXPECT_FALSE(value(Target));
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
  EXPECT_FALSE(value(Target));
}

TEST_F(AnalysisBaseMergeTest, CopyDownAloneIsNotAChange) {
  clad::VarsData Root, Parent, Target, Source;
  Parent.m_Prev = Source.m_Prev = &Root;
  Target.m_Prev = &Parent;
  put(Root, false);
  put(Parent, false);
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
  EXPECT_EQ(visible(&Target), 1);
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
}

TEST_F(AnalysisBaseMergeTest, MergeDoesNotMutateSourceOrAncestor) {
  clad::VarsData Root, Target, Source;
  Target.m_Prev = Source.m_Prev = &Root;
  put(Root, false);
  put(Source, true);
  EXPECT_TRUE(Analysis.merge(&Target, &Source));
  EXPECT_TRUE(value(Target));
  EXPECT_TRUE(value(Source));
  EXPECT_FALSE(value(Root));
  EXPECT_FALSE(Analysis.merge(&Target, &Source));
}

TEST_F(AnalysisBaseMergeTest, SelfMergeIsUnchanged) {
  clad::VarsData Target;
  put(Target, true);
  EXPECT_FALSE(Analysis.merge(&Target, &Target));
  EXPECT_TRUE(value(Target));
}

TEST_F(AnalysisBaseMergeTest, ScalarForksJoinAndThenStabilize) {
  for (int Code = 0; Code < 243; ++Code) {
    SCOPED_TRACE(Code);
    clad::VarsData Root, Left, Right, Target, Source;
    Left.m_Prev = Right.m_Prev = &Root;
    Target.m_Prev = &Left;
    Source.m_Prev = &Right;
    int States = Code;
    int InState[5];
    int Idx = 0;
    for (auto* Data : {&Root, &Left, &Right, &Target, &Source}) {
      int State = States % 3;
      States /= 3;
      InState[Idx++] = State;
      if (State)
        put(*Data, State == 2);
    }
    int Before = visible(&Target);
    int Incoming = visible(&Source);
    int Expected =
        Before == 2 || Incoming == 2 ? 2 : (Before || Incoming ? 1 : 0);
    // Phase 3 refreshes local state without reporting worklist growth. This
    // occurs when Target's local false is refreshed from Root's true while
    // Left, Right, and Source introduce no branch-specific state.
    bool IsAncestorRefreshOnly =
        (InState[0] == 2 && InState[1] == 0 && InState[2] == 0 &&
         InState[3] == 1 && InState[4] == 0);
    bool ExpectedChangeReported =
        (Expected != Before) && !IsAncestorRefreshOnly;
    EXPECT_EQ(Analysis.merge(&Target, &Source), ExpectedChangeReported);
    EXPECT_EQ(visible(&Target), Expected);
    EXPECT_EQ(visible(&Source), Incoming);
    EXPECT_FALSE(Analysis.merge(&Target, &Source));
    EXPECT_EQ(visible(&Target), Expected);
  }
}
} // namespace
