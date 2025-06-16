#ifndef CLAD_DIFFERENTIATOR_TIMERS_H
#define CLAD_DIFFERENTIATOR_TIMERS_H

#include "llvm/ADT/StringRef.h"

#include <functional>
#include <string>

namespace clad {

struct TimedAnalysisRegion {
  TimedAnalysisRegion(llvm::StringRef Name);
  /// Only runs the lambda if timers are enabled. Useful for more complex
  /// operations such as ::print.
  TimedAnalysisRegion(const std::function<std::string()>& NameProvider);
  ~TimedAnalysisRegion();
};
struct TimedGenerationRegion {
  TimedGenerationRegion(llvm::StringRef Name);
  /// Only runs the lambda if timers are enabled. Useful for more complex
  /// operations such as ::print.
  TimedGenerationRegion(const std::function<std::string()>& NameProvider);
  ~TimedGenerationRegion();
};
} // namespace clad
#endif // CLAD_DIFFERENTIATOR_TIMERS_H
