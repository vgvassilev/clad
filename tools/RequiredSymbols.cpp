#include "clang/AST/QualTypeNames.h"

// MSVC has no equivalent spelling, and needs none: an unresolved symbol fails
// the link there rather than binding to null.
#if defined(__GNUC__) || defined(__clang__)
#define CLAD_USED __attribute__((used))
#else
#define CLAD_USED
#endif

namespace clad {
namespace internal {
// Force-reference clang symbols a statically-linked plugin host may not pull
// in itself; tools/CMakeLists.txt links clangAST to supply them.
CLAD_USED void* symbol_requester() {
  static void* const kRequiredSymbols[] = {
      // Spelled out so an upstream signature change fails the build here,
      // instead of forcing in a symbol makeTypeReadable no longer calls.
      reinterpret_cast<void*>(
          static_cast<clang::QualType (*)(clang::QualType,
                                          const clang::ASTContext&, bool)>(
              &clang::TypeName::getFullyQualifiedType)),
  };
  return kRequiredSymbols[0];
}
} // namespace internal
} // namespace clad
