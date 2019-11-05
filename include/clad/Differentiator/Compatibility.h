//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_COMPATIBILITY
#define CLAD_COMPATIBILITY

#include "clang/Basic/Version.h"
//#define CLANG_VERSION 7.0.0
//#define CLANG_VERSION_STRING "7.0.0"
//#define CLANG_VERSION_MAJOR 7
//#define CLANG_VERSION_MINOR 0
//#define CLANG_VERSION_PATCHLEVEL 0
#include "llvm/Config/llvm-config.h"
//#define LLVM_VERSION_MAJOR 7
//#define LLVM_VERSION_MINOR 0
//#define LLVM_VERSION_PATCH 0
//#define LLVM_VERSION_STRING "7.0.0"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"

#if CLANG_VERSION_MAJOR==5

// Compatibility helper function for creation CompoundStmt

static inline clang::CompoundStmt* clang__CompoundStmt__Create(
        const clang::ASTContext &C, clang::ArrayRef<clang::Stmt *> Stmts,
        clang::SourceLocation LB, clang::SourceLocation RB)
{
   return new (C) clang::CompoundStmt(C, Stmts, LB, RB);
}

//
#define Sema__ForVisibleRedeclaration Sema::ForRedeclaration

//
#define DeclaratorContext Declarator

#endif //CLANG_VERSION_MAJOR==5

#if CLANG_VERSION_MAJOR==6

//#define LOOKUP_ARGN B

static inline clang::CompoundStmt* clang__CompoundStmt__Create(
        const clang::ASTContext &C, clang::ArrayRef<clang::Stmt *> Stmts,
        clang::SourceLocation LB, clang::SourceLocation RB)
{
   return clang::CompoundStmt::Create(C, Stmts, LB, RB);
}

//
#define Sema__ForVisibleRedeclaration Sema::ForVisibleRedeclaration

//
//#define DeclaratorContext DeclaratorContext

#endif //CLANG_VERSION_MAJOR==6


#endif //CLAD_COMPATIBILITY
