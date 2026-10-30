//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "GeneratedSourceFile.h"

#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>

using namespace clang;

namespace clad {
void GeneratedSourceFile::write(CompilerInstance& CI, llvm::StringRef Path) {
  // What a translation unit that did not compile made is not worth writing.
  if (m_Entries.empty() || CI.getDiagnostics().hasErrorOccurred()) {
    m_Entries.clear();
    return;
  }
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    unsigned ID = CI.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "could not write the derivatives to '%0': %1");
    CI.getDiagnostics().Report(ID) << Path << EC.message();
    m_Entries.clear();
    return;
  }
  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  PrintingPolicy Policy(LangOpts);
  Policy.Bool = true;
  for (const Entry& E : m_Entries) {
    E.D->print(Out, Policy);
    if (E.DeclarationOnly)
      Out << ";\n";
  }
  m_Entries.clear();
}
} // namespace clad
