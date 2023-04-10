//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/Version.h"

#ifdef HAVE_VCS_VERSION_INC
#include "VCSVersion.inc"
#endif

namespace clad {
  std::string getCladRevision() {
#ifdef CLAD_REVISION
    return CLAD_REVISION;
#else
    return "";
#endif
  }
} // end namespace clad
