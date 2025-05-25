//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/Version.h"

#ifdef HAVE_VCS_VERSION_INC
#include "VCSVersion.inc"
#endif

#include "clang/Basic/Version.h"

#include "llvm/Support/raw_ostream.h"

namespace clad {
  std::string getCladRevision() {
#ifdef CLAD_REVISION
    return CLAD_REVISION;
#else
    return "";
#endif // CLAD_REVISION
  }

  std::string getCladRepositoryPath() {
#ifdef CLAD_REPOSITORY
    return CLAD_REPOSITORY;
#else
    return "";
#endif // CLAD_REPOSITORY
  }

  std::string getCladFullRepositoryVersion() {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    std::string Path = getCladRepositoryPath();
    std::string Revision = getCladRevision();
    if (!Path.empty() || !Revision.empty()) {
      OS << '(';
      if (!Path.empty())
        OS << Path;
      if (!Revision.empty()) {
        if (!Path.empty())
          OS << ' ';
        OS << Revision;
      }
      OS << ')';
    }
    return buf;
  }

  std::string getCladFullVersion() {
    std::string buf;
    llvm::raw_string_ostream OS(buf);
    OS << "clad version " CLAD_VERSION_STRING;

    std::string repo = getCladFullRepositoryVersion();
    if (!repo.empty())
      OS << " " << repo;

    std::string clangRepo = clang::getClangFullVersion();
    if (!clangRepo.empty())
      OS << " [for " << clangRepo << "]";

    return buf;
  }
} // end namespace clad
