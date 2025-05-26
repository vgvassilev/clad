//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR_VERSION
#define CLAD_DIFFERENTIATOR_VERSION

#include "clad/Differentiator/Version.inc"

#include <string>

namespace clad {
  std::string getCladRevision();
  std::string getCladRepositoryPath();
  std::string getCladFullRepositoryVersion();
  std::string getCladFullVersion();
} // end namespace clad

#endif //CLAD_DIFFERENTIATOR_VERSION
