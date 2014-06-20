//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/Version.h"

namespace clad {
  std::string getCladRevision() {
#ifdef CLAD_GIT_REVISION
    return CLAD_GIT_REVISION;
#else
    return "";
#endif
  }

  std::string getClangCompatRevision() {
#ifdef CLAD_CLANG_COMPAT_REVISION
    return CLAD_CLANG_COMPAT_REVISION;
#else
    return "";
#endif
  }
} // end namespace clad
