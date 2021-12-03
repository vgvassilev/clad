/// This file contains functions for easily querying, modifying and creating
/// AST nodes.

#ifndef CLAD_AST_HELPERS_H
#define CLAD_AST_HELPERS_H

#include "clang/AST/Type.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "clad/Differentiator/Compatibility.h"

namespace clang {
  class ASTContext;
  class CXXNewExpr;
  class CXXRecordDecl;
  class CXXConstructorDecl;
  class CXXMethodDecl;
  class Decl;
  class DeclarationName;
  class DeclarationNameInfo;
  class DeclRefExpr;
  class DeclStmt;
  class Expr;
  class FieldDecl;
  class IdentifierInfo;
  class MemberExpr;
  class NamespaceDecl;
  class ParmVarDecl;
  class ValueDecl;
  class VarDecl;
  class Sema;
  class SourceLocation;
} // namespace clang
namespace clad {
  class ASTHelper {
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;

  public:
    struct Scope {
      static const unsigned
          FunctionBeginScope = clang::Scope::FunctionPrototypeScope |
                               clang::Scope::FunctionDeclarationScope |
                               clang::Scope::DeclScope;
      static const unsigned FunctionBodyScope = clang::Scope::FnScope |
                                                clang::Scope::DeclScope;
    };
    ASTHelper(clang::Sema& sema);

    clang::CXXRecordDecl* FindCXXRecordDecl(clang::DeclarationName recordName);
    static clang::CXXRecordDecl*
    FindCXXRecordDecl(clang::Sema& semaRef, clang::DeclarationName recordName);

    clang::QualType GetFundamentalType(llvm::StringRef typeName) const;
    static clang::QualType GetFundamentalType(clang::Sema& semaRef,
                                              llvm::StringRef typeName);

    bool IsFundamentalType(llvm::StringRef name) const;
    static bool IsFundamentalType(clang::Sema& semaRef, llvm::StringRef name);

    clang::QualType ComputeQTypeFromTypeName(llvm::StringRef typeName) const;
    static clang::QualType ComputeQTypeFromTypeName(clang::Sema& semaRef,
                                                    llvm::StringRef typeName);

    clang::ValueDecl* FindRecordDeclMember(clang::CXXRecordDecl* RD,
                                           llvm::StringRef memberName);

    static clang::ValueDecl* FindRecordDeclMember(clang::Sema& semaRef,
                                                  clang::CXXRecordDecl* RD,
                                                  llvm::StringRef memberName);

    clang::MemberExpr* BuildMemberExpr(clang::Expr* base,
                                       clang::ValueDecl* member);

    static clang::MemberExpr* BuildMemberExpr(clang::Sema& semaRef,
                                              clang::Expr* base,
                                              clang::ValueDecl* member);

    clang::CXXNewExpr* CreateNewExprFor(clang::QualType qType,
                                        clang::Expr* initializer,
                                        clang::SourceLocation B);

    static clang::CXXNewExpr* CreateNewExprFor(clang::Sema& semaRef,
                                               clang::QualType qType,
                                               clang::Expr* initializer,
                                               clang::SourceLocation B);

    clang::CXXConstructorDecl* FindCopyConstructor(clang::CXXRecordDecl* RD);

    static clang::CXXConstructorDecl*
    FindCopyConstructor(clang::Sema& semaRef, clang::CXXRecordDecl* RD);

    clang::FunctionDecl* FindUniqueFnDecl(clang::DeclContext* DC,
                                          clang::DeclarationName fnName);

    static clang::FunctionDecl* FindUniqueFnDecl(clang::Sema& semaRef,
                                                 clang::DeclContext* DC,
                                                 clang::DeclarationName fnName);

    clang::DeclarationName CreateDeclName(llvm::StringRef name);
    static clang::DeclarationName CreateDeclName(clang::Sema& semaRef,
                                                 llvm::StringRef name);

    clang::DeclarationNameInfo CreateDeclNameInfo(llvm::StringRef name);
    static clang::DeclarationNameInfo CreateDeclNameInfo(clang::Sema& semaRef,
                                                         llvm::StringRef name);

    clang::Expr* IgnoreParenImpCastsUnaryOp(clang::Expr* E);

    static clang::Expr* IgnoreParenImpCastsUnaryOp(clang::Sema& semaRef,
                                                   clang::Expr* E);

    clang::Expr* BuildCXXCopyConstructExpr(clang::QualType qType,
                                           clang::Expr* E);
    // NOTE: This currently only creates default construct expressions for class
    // types
    static clang::Expr* BuildCXXCopyConstructExpr(clang::Sema& semaRef,
                                                  clang::QualType qType,
                                                  clang::Expr* E);

    clang::QualType FindCorrespondingType(llvm::StringRef name);
    static clang::QualType FindCorrespondingType(clang::Sema& semaRef,
                                                 llvm::StringRef name);

    clang::NamespaceDecl* FindCladNamespace();
    static clang::NamespaceDecl* FindCladNamespace(clang::Sema& semaRef);

    clang::SourceLocation GetValidSL();
    static clang::SourceLocation GetValidSL(clang::Sema& semaRef);

    clang::SourceRange GetValidSR();
    static clang::SourceRange GetValidSR(clang::Sema& semaRef);

    clang::VarDecl* BuildVarDecl(clang::DeclContext* DC,
                                 clang::IdentifierInfo* II,
                                 clang::QualType qType);
    static clang::VarDecl* BuildVarDecl(clang::Sema& semaRef,
                                        clang::DeclContext* DC,
                                        clang::IdentifierInfo* II,
                                        clang::QualType qType);

    clang::ParmVarDecl* BuildParmVarDecl(clang::DeclContext* DC,
                                         clang::IdentifierInfo* II,
                                         clang::QualType qType);
    static clang::ParmVarDecl* BuildParmVarDecl(clang::Sema& semaRef,
                                                clang::DeclContext* DC,
                                                clang::IdentifierInfo* II,
                                                clang::QualType qType);

    clang::DeclRefExpr*
    BuildDeclRefExpr(clang::ValueDecl* VD,
                     clang::QualType qType = clang::QualType());
    static clang::DeclRefExpr*
    BuildDeclRefExpr(clang::Sema& semaRef, clang::ValueDecl* VD,
                     clang::QualType qType = clang::QualType());
    clang::DeclStmt* BuildDeclStmt(clang::Decl* D);
    static clang::DeclStmt* BuildDeclStmt(clang::Sema& semaRef, clang::Decl* D);

    clang::FieldDecl* BuildFieldDecl(clang::DeclContext* DC,
                                     clang::IdentifierInfo* II,
                                     clang::QualType qType);
    static clang::FieldDecl* BuildFieldDecl(clang::Sema& sema,
                                            clang::DeclContext* DC,
                                            clang::IdentifierInfo* II,
                                            clang::QualType qType);

    void RegisterFn(clang::DeclContext* DC, clang::FunctionDecl* FD);
    static void RegisterFn(clang::Sema& semaRef, clang::DeclContext* DC,
                           clang::FunctionDecl* FD);

    clang::CompoundStmt* BuildCompoundStmt(llvm::ArrayRef<clang::Stmt*> block);
    static clang::CompoundStmt*
    BuildCompoundStmt(clang::Sema& semaRef, llvm::ArrayRef<clang::Stmt*> block);

    clang::FunctionDecl* BuildFnDecl(clang::DeclContext* DC,
                                     clang::DeclarationName fnName,
                                     clang::QualType fnQType);
    static clang::FunctionDecl* BuildFnDecl(clang::Sema& semaRef,
                                            clang::DeclContext* DC,
                                            clang::DeclarationName fnName,
                                            clang::QualType fnQType);

    clang::CXXMethodDecl* BuildMemFnDecl(clang::CXXRecordDecl* RD,
                                         clang::DeclarationNameInfo nameInfo,
                                         clang::QualType qType);
    static clang::CXXMethodDecl* BuildMemFnDecl(clang::Sema& semaRef, clang::CXXRecordDecl* RD,
                                         clang::DeclarationNameInfo nameInfo,
                                         clang::QualType qType);
  };
} // namespace clad
#endif