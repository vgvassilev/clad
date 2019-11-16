//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------

#ifndef CLAD_COMPATIBILITY
#define CLAD_COMPATIBILITY

#include "clang/Basic/Version.h"
#include "llvm/Config/llvm-config.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "clang/Sema/Sema.h"

namespace clad_compat {

// Compatibility helper function for creation CompoundStmt. Clang 6 and above use Create.

static inline clang::CompoundStmt* CompoundStmt_Create(
        const clang::ASTContext &C, clang::ArrayRef<clang::Stmt *> Stmts,
        clang::SourceLocation LB, clang::SourceLocation RB)
{
#if CLANG_VERSION_MAJOR == 5
   return new (C) clang::CompoundStmt(C, Stmts, LB, RB);
#elif CLANG_VERSION_MAJOR >= 6
   return clang::CompoundStmt::Create(C, Stmts, LB, RB);
#endif
}


// Clang 6 rename Sema::ForRedeclaration to Sema::ForVisibleRedeclaration

#if CLANG_VERSION_MAJOR == 5
   const auto Sema_ForVisibleRedeclaration = clang::Sema::ForRedeclaration;
#elif CLANG_VERSION_MAJOR >= 6
   const auto Sema_ForVisibleRedeclaration = clang::Sema::ForVisibleRedeclaration;
#endif


// Clang 6 rename Declarator to DeclaratorContext, but Declarator is used
// as name for another class.
#if CLANG_VERSION_MAJOR == 5
   using DeclaratorContext = clang::Declarator;
#elif CLANG_VERSION_MAJOR >= 6
   using DeclaratorContext = clang::DeclaratorContext;
#endif


// Clang 7 add one extra param in UnaryOperator constructor.
#if CLANG_VERSION_MAJOR < 7
   #define CLAD_COMPAT_CLANG7_UnaryOperator_ExtraParams /**/
#elif CLANG_VERSION_MAJOR >= 7
   #define CLAD_COMPAT_CLANG7_UnaryOperator_ExtraParams ,Node->canOverflow()
#endif

} // namespace clad_compat

#endif //CLAD_COMPATIBILITY
