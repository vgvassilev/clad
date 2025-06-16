#include "clad/Differentiator/Timers.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <functional>
#include <memory>
#include <string>

namespace clad {

// Implementation heavily inspired by llvm::TimePassesHandler.
class CladTimeInfo {
  /// Group of timers that are part of the analysis stage.
  llvm::TimerGroup m_AnalysisTG;
  /// Group of timers that are part of the transformation stage.
  llvm::TimerGroup m_DiffTG;

  using TimerVector = llvm::SmallVector<llvm::Timer, 4>;
  llvm::StringMap<TimerVector> m_TimingData;

  using Timers = llvm::SmallVector<llvm::Timer*, 8>;
  Timers m_AnalysisTimers;
  Timers m_DiffTimers;

  /// Returns the timer for the specified \p ID or creates a new one.
  llvm::Timer& GetTimer(llvm::StringRef ID, llvm::TimerGroup& TG) {
    auto& Ts = m_TimingData[ID];
    if (Ts.empty())
      Ts.emplace_back(ID, ID, TG);
    return Ts.front();
  }

  void StartTimer(llvm::StringRef ID, Timers& Ts, llvm::TimerGroup& TG) {
    // Stop the previous timer to prevent double counting when an timed
    // region requests another timer.
    if (!Ts.empty()) {
      assert(Ts.back()->isRunning());
      Ts.back()->stopTimer();
    }

    llvm::Timer& T = GetTimer(ID, TG);
    Ts.push_back(&T);
    if (!T.isRunning())
      T.startTimer();
  }

  static void StopTimer(Timers& Ts) {
    assert(!Ts.empty() && "empty stack in StopTimer");
    llvm::Timer* T = Ts.pop_back_val();
    assert(T && "timer should be present");
    if (T->isRunning())
      T->stopTimer();

    // Restart the previously stopped timer.
    if (!Ts.empty()) {
      assert(!Ts.back()->isRunning());
      Ts.back()->startTimer();
    }
  }

public:
  CladTimeInfo()
      : m_AnalysisTG("analysis", "Clad Analysis Timing Report"),
        m_DiffTG("ast generation", "Clad AST Generation Timing Report") {}

  /// Destructor handles the print action if it has not been handled before.
  ~CladTimeInfo() { print(); }

  // We intend this to be unique per-compilation, thus no copies.
  CladTimeInfo(const CladTimeInfo&) = delete;
  CladTimeInfo(CladTimeInfo&&) = delete;
  void operator=(const CladTimeInfo&) = delete;
  CladTimeInfo& operator=(CladTimeInfo&&) = delete;

  void StartAnalysisTimer(llvm::StringRef Name) {
    StartTimer(Name, m_AnalysisTimers, m_AnalysisTG);
  }
  void StopAnalysisTimer() { StopTimer(m_AnalysisTimers); }
  void StartDiffTimer(llvm::StringRef Name) {
    StartTimer(Name, m_DiffTimers, m_DiffTG);
  }
  void StopDiffTimer() { StopTimer(m_DiffTimers); }
  void print() {
    if (!CladTimeInfo::TheTimingInfo)
      return;
    std::unique_ptr<llvm::raw_ostream> OS = llvm::CreateInfoOutputFile();
    m_DiffTG.print(*OS, true);
    m_AnalysisTG.print(*OS, true);
  }
  static void Init();
  static CladTimeInfo* TheTimingInfo;
};
CladTimeInfo* CladTimeInfo::TheTimingInfo;
void InitTimers() {
  assert(!CladTimeInfo::TheTimingInfo);
  static llvm::ManagedStatic<CladTimeInfo> CTI;
  CladTimeInfo::TheTimingInfo = &*CTI;
}

TimedAnalysisRegion::TimedAnalysisRegion(llvm::StringRef Name) {
  if (CladTimeInfo::TheTimingInfo)
    CladTimeInfo::TheTimingInfo->StartAnalysisTimer(Name);
}
TimedAnalysisRegion::TimedAnalysisRegion(
    const std::function<std::string()>& NameProvider)
    : TimedAnalysisRegion(CladTimeInfo::TheTimingInfo ? NameProvider() : "") {}
TimedAnalysisRegion::~TimedAnalysisRegion() {
  if (CladTimeInfo::TheTimingInfo)
    CladTimeInfo::TheTimingInfo->StopAnalysisTimer();
}

TimedGenerationRegion::TimedGenerationRegion(llvm::StringRef Name) {
  if (CladTimeInfo::TheTimingInfo)
    CladTimeInfo::TheTimingInfo->StartDiffTimer(Name);
}
TimedGenerationRegion::TimedGenerationRegion(
    const std::function<std::string()>& NameProvider)
    : TimedGenerationRegion(CladTimeInfo::TheTimingInfo ? NameProvider() : "") {
}
TimedGenerationRegion::~TimedGenerationRegion() {
  if (CladTimeInfo::TheTimingInfo)
    CladTimeInfo::TheTimingInfo->StopDiffTimer();
}
} // namespace clad
