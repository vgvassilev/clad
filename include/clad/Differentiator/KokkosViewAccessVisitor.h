//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_KOKKOS_VIEW_ACCESS_VISITOR_H
#define CLAD_KOKKOS_VIEW_ACCESS_VISITOR_H

#include "clad/Differentiator/CladUtils.h"

namespace clad {

static bool isIdenticalStmt(const clang::ASTContext &Ctx, const clang::Stmt *Stmt1,
                            const clang::Stmt *Stmt2, bool IgnoreSideEffects) {

  using namespace clang;

  if (!Stmt1 || !Stmt2) {
    return !Stmt1 && !Stmt2;
  }

  // If Stmt1 & Stmt2 are of different class then they are not
  // identical statements.
  if (Stmt1->getStmtClass() != Stmt2->getStmtClass())
    return false;

  const Expr *Expr1 = dyn_cast<Expr>(Stmt1);
  const Expr *Expr2 = dyn_cast<Expr>(Stmt2);

  if (Expr1 && Expr2) {
    // If Stmt1 has side effects then don't warn even if expressions
    // are identical.
    if (!IgnoreSideEffects && Expr1->HasSideEffects(Ctx))
      return false;
    // If either expression comes from a macro then don't warn even if
    // the expressions are identical.
    if ((Expr1->getExprLoc().isMacroID()) || (Expr2->getExprLoc().isMacroID()))
      return false;

    // If all children of two expressions are identical, return true.
    Expr::const_child_iterator I1 = Expr1->child_begin();
    Expr::const_child_iterator I2 = Expr2->child_begin();
    while (I1 != Expr1->child_end() && I2 != Expr2->child_end()) {
      if (!*I1 || !*I2 || !isIdenticalStmt(Ctx, *I1, *I2, IgnoreSideEffects))
        return false;
      ++I1;
      ++I2;
    }
    // If there are different number of children in the statements, return
    // false.
    if (I1 != Expr1->child_end())
      return false;
    if (I2 != Expr2->child_end())
      return false;
  }

  switch (Stmt1->getStmtClass()) {
  default:
    return false;
  case Stmt::CallExprClass:
  case Stmt::ArraySubscriptExprClass:
  case Stmt::OMPArraySectionExprClass:
  case Stmt::OMPArrayShapingExprClass:
  case Stmt::OMPIteratorExprClass:
  case Stmt::ImplicitCastExprClass:
  case Stmt::ParenExprClass:
  case Stmt::BreakStmtClass:
  case Stmt::ContinueStmtClass:
  case Stmt::NullStmtClass:
    return true;
  case Stmt::CStyleCastExprClass: {
    const CStyleCastExpr* CastExpr1 = cast<CStyleCastExpr>(Stmt1);
    const CStyleCastExpr* CastExpr2 = cast<CStyleCastExpr>(Stmt2);

    return CastExpr1->getTypeAsWritten() == CastExpr2->getTypeAsWritten();
  }
  case Stmt::ReturnStmtClass: {
    const ReturnStmt *ReturnStmt1 = cast<ReturnStmt>(Stmt1);
    const ReturnStmt *ReturnStmt2 = cast<ReturnStmt>(Stmt2);

    return isIdenticalStmt(Ctx, ReturnStmt1->getRetValue(),
                           ReturnStmt2->getRetValue(), IgnoreSideEffects);
  }
  case Stmt::ForStmtClass: {
    const ForStmt *ForStmt1 = cast<ForStmt>(Stmt1);
    const ForStmt *ForStmt2 = cast<ForStmt>(Stmt2);

    if (!isIdenticalStmt(Ctx, ForStmt1->getInit(), ForStmt2->getInit(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, ForStmt1->getCond(), ForStmt2->getCond(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, ForStmt1->getInc(), ForStmt2->getInc(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, ForStmt1->getBody(), ForStmt2->getBody(),
                         IgnoreSideEffects))
      return false;
    return true;
  }
  case Stmt::DoStmtClass: {
    const DoStmt *DStmt1 = cast<DoStmt>(Stmt1);
    const DoStmt *DStmt2 = cast<DoStmt>(Stmt2);

    if (!isIdenticalStmt(Ctx, DStmt1->getCond(), DStmt2->getCond(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, DStmt1->getBody(), DStmt2->getBody(),
                         IgnoreSideEffects))
      return false;
    return true;
  }
  case Stmt::WhileStmtClass: {
    const WhileStmt *WStmt1 = cast<WhileStmt>(Stmt1);
    const WhileStmt *WStmt2 = cast<WhileStmt>(Stmt2);

    if (!isIdenticalStmt(Ctx, WStmt1->getCond(), WStmt2->getCond(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, WStmt1->getBody(), WStmt2->getBody(),
                         IgnoreSideEffects))
      return false;
    return true;
  }
  case Stmt::IfStmtClass: {
    const IfStmt *IStmt1 = cast<IfStmt>(Stmt1);
    const IfStmt *IStmt2 = cast<IfStmt>(Stmt2);

    if (!isIdenticalStmt(Ctx, IStmt1->getCond(), IStmt2->getCond(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, IStmt1->getThen(), IStmt2->getThen(),
                         IgnoreSideEffects))
      return false;
    if (!isIdenticalStmt(Ctx, IStmt1->getElse(), IStmt2->getElse(),
                         IgnoreSideEffects))
      return false;
    return true;
  }
  case Stmt::CompoundStmtClass: {
    const CompoundStmt *CompStmt1 = cast<CompoundStmt>(Stmt1);
    const CompoundStmt *CompStmt2 = cast<CompoundStmt>(Stmt2);

    if (CompStmt1->size() != CompStmt2->size())
      return false;

    CompoundStmt::const_body_iterator I1 = CompStmt1->body_begin();
    CompoundStmt::const_body_iterator I2 = CompStmt2->body_begin();
    while (I1 != CompStmt1->body_end() && I2 != CompStmt2->body_end()) {
      if (!isIdenticalStmt(Ctx, *I1, *I2, IgnoreSideEffects))
        return false;
      ++I1;
      ++I2;
    }

    return true;
  }
  case Stmt::CompoundAssignOperatorClass:
  case Stmt::BinaryOperatorClass: {
    const BinaryOperator *BinOp1 = cast<BinaryOperator>(Stmt1);
    const BinaryOperator *BinOp2 = cast<BinaryOperator>(Stmt2);
    return BinOp1->getOpcode() == BinOp2->getOpcode();
  }
  case Stmt::CharacterLiteralClass: {
    const CharacterLiteral *CharLit1 = cast<CharacterLiteral>(Stmt1);
    const CharacterLiteral *CharLit2 = cast<CharacterLiteral>(Stmt2);
    return CharLit1->getValue() == CharLit2->getValue();
  }
  case Stmt::DeclRefExprClass: {
    const DeclRefExpr *DeclRef1 = cast<DeclRefExpr>(Stmt1);
    const DeclRefExpr *DeclRef2 = cast<DeclRefExpr>(Stmt2);
    return DeclRef1->getDecl() == DeclRef2->getDecl();
  }
  case Stmt::IntegerLiteralClass: {
    const IntegerLiteral *IntLit1 = cast<IntegerLiteral>(Stmt1);
    const IntegerLiteral *IntLit2 = cast<IntegerLiteral>(Stmt2);

    llvm::APInt I1 = IntLit1->getValue();
    llvm::APInt I2 = IntLit2->getValue();
    if (I1.getBitWidth() != I2.getBitWidth())
      return false;
    return  I1 == I2;
  }
  case Stmt::FloatingLiteralClass: {
    const FloatingLiteral *FloatLit1 = cast<FloatingLiteral>(Stmt1);
    const FloatingLiteral *FloatLit2 = cast<FloatingLiteral>(Stmt2);
    return FloatLit1->getValue().bitwiseIsEqual(FloatLit2->getValue());
  }
  case Stmt::StringLiteralClass: {
    const StringLiteral *StringLit1 = cast<StringLiteral>(Stmt1);
    const StringLiteral *StringLit2 = cast<StringLiteral>(Stmt2);
    return StringLit1->getBytes() == StringLit2->getBytes();
  }
  case Stmt::MemberExprClass: {
    const MemberExpr *MemberStmt1 = cast<MemberExpr>(Stmt1);
    const MemberExpr *MemberStmt2 = cast<MemberExpr>(Stmt2);
    return MemberStmt1->getMemberDecl() == MemberStmt2->getMemberDecl();
  }
  case Stmt::UnaryOperatorClass: {
    const UnaryOperator *UnaryOp1 = cast<UnaryOperator>(Stmt1);
    const UnaryOperator *UnaryOp2 = cast<UnaryOperator>(Stmt2);
    return UnaryOp1->getOpcode() == UnaryOp2->getOpcode();
  }
  }
}

  class KokkosViewAccessVisitor {
    public:
      KokkosViewAccessVisitor (clang::Sema& _semaRef, clang::ASTContext& _m_Context) : 
        semaRef(_semaRef), m_Context(_m_Context) {}
      
      void Visit(const clang::Stmt *Node, bool record_view_names = false, bool RHS = true) {
        if (llvm::isa<clang::CallExpr>(Node)) {
          if (llvm::isa<clang::CXXOperatorCallExpr>(Node)) {
            auto OCE = llvm::dyn_cast<clang::CXXOperatorCallExpr>(Node);

            std::string constructedTypeName = OCE->getDirectCallee()->getQualifiedNameAsString();

            if(constructedTypeName.find("Kokkos::View") != std::string::npos 
              && constructedTypeName.find("::operator()") != std::string::npos) {

              view_accesses.push_back(OCE);
              view_accesses_location.push_back(OCE->getBeginLoc());
              view_accesses_RHS.push_back(RHS);
            }
          }
          else {
            record_view_names = true;
          }
        }
        if (llvm::isa<clang::DeclRefExpr>(Node)) {
          auto DRE = llvm::dyn_cast<clang::DeclRefExpr>(Node);
          auto VD = llvm::dyn_cast<clang::VarDecl>(DRE->getDecl());

          if(VD && record_view_names) {
            std::string constructedTypeName = clang::QualType::getAsString(VD->getType().split(), clang::PrintingPolicy{ {} });
            if (utils::IsKokkosView(constructedTypeName)) {
              std::string name = DRE->getNameInfo().getName().getAsString();

              if(!std::count(view_names.begin(), view_names.end(), name)) {
                view_names.push_back(name);
                view_DeclRefExpr.push_back(DRE);
              }
            }
            return;
          }
        }
        if (llvm::isa<clang::BinaryOperator>(Node) && llvm::dyn_cast<clang::BinaryOperator>(Node)->isAssignmentOp()) {
          Visit(llvm::dyn_cast<clang::BinaryOperator>(Node)->getLHS(),record_view_names,false);
          Visit(llvm::dyn_cast<clang::BinaryOperator>(Node)->getRHS(),record_view_names,RHS);
          return;
        }

        for (const clang::Stmt *SubStmt : Node->children())
          Visit(SubStmt, record_view_names);
      }

      bool VisitViewAccess(const clang::Stmt *Node, const clang::ParmVarDecl *param) {
        if (llvm::isa<clang::DeclRefExpr>(Node)) {
          auto VD = llvm::dyn_cast<clang::DeclRefExpr>(Node)->getDecl();
          if (llvm::isa<clang::ParmVarDecl>(VD)) {
            if (VD == param)
              return true;
          }
        }
        bool tmp = false;
        for (const clang::Stmt *SubStmt : Node->children())
          tmp = tmp || VisitViewAccess(SubStmt, param);
        return tmp;
      }

      bool VisitViewAccess(const clang::CXXOperatorCallExpr* view_access, std::vector<clang::ParmVarDecl*> params) {
        bool use_all_params = true;
        for (auto PVD : params) {
          use_all_params = use_all_params && VisitViewAccess(view_access, PVD);
        }
        if (!use_all_params) {
          unsigned diagID = semaRef.Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning, "The view access does not use all the parameters of the lambda call; an atomic will be required for the reverse mode. ");
          clang::Sema::SemaDiagnosticBuilder stream = semaRef.Diag(view_access->getBeginLoc(), diagID);
        }
        return use_all_params;
      }

      void VisitViewAccesses(std::vector<clang::ParmVarDecl*> params) {
        for (size_t i = 0; i < view_accesses.size(); ++i) {
          if (view_accesses_RHS[i])
            view_accesses_is_thread_safe.push_back(VisitViewAccess(view_accesses[i], params));
          else
            view_accesses_is_thread_safe.push_back(false);
        }
        // Check for nested view accesses:
        for (size_t i = 0; i < view_accesses.size(); ++i) {
          for (size_t i_arg = 1; i_arg < view_accesses[i]->getNumArgs(); ++i_arg) {
            
              if (llvm::isa<clang::CXXOperatorCallExpr>(view_accesses[i]->getArg(i_arg))) {
                auto OCE = llvm::dyn_cast<clang::CXXOperatorCallExpr>(view_accesses[i]->getArg(i_arg));

                std::string constructedTypeName = OCE->getDirectCallee()->getQualifiedNameAsString();

                if(constructedTypeName.find("Kokkos::View") != std::string::npos 
                  && constructedTypeName.find("::operator()") != std::string::npos) {
                  view_accesses_is_thread_safe[i] = false;
                  {
                    unsigned diagID1 = semaRef.Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning, 
                    "The view access has a nested view access-- continued");
                    clang::Sema::SemaDiagnosticBuilder stream1 = semaRef.Diag(view_accesses[i]->getBeginLoc(), diagID1);
                  }
                  {
                    unsigned diagID2 = semaRef.Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning, 
                    "continued -- here; an atomic will be required for the reverse mode. ");
                    clang::Sema::SemaDiagnosticBuilder stream2 = semaRef.Diag(view_accesses[i]->getArg(i_arg)->getBeginLoc(), diagID2);
                  }
                  break;
                }
              }
          }
        }
        // Check if two view accesses could be similar for two different param
        // instance.
        for (size_t i = 0; i < view_accesses.size(); ++i) {
          if (!view_accesses_is_thread_safe[i])
            continue;
          std::string name_i = llvm::dyn_cast<clang::DeclRefExpr>(view_accesses[i]->getArg(0))->getNameInfo().getName().getAsString();
          for (size_t j = i+1; j < view_accesses.size(); ++j) {

            // If the two views have different name, continue
            std::string name_j = llvm::dyn_cast<clang::DeclRefExpr>(view_accesses[j]->getArg(0))->getNameInfo().getName().getAsString();
            if (name_i != name_j)
              continue;
            
            bool updated_thread_safe = true;
            bool all_other_same_args = true;

            for (size_t i_arg = 1; i_arg < view_accesses[i]->getNumArgs(); ++i_arg) {
              bool include_param = false;
              for (auto PVD : params) {
                if (VisitViewAccess(view_accesses[i]->getArg(i_arg), PVD) && VisitViewAccess(view_accesses[j]->getArg(i_arg), PVD)) {
                  include_param = true;
                  if (!isIdenticalStmt(m_Context, view_accesses[i]->getArg(i_arg), view_accesses[j]->getArg(i_arg), true)) {
                    updated_thread_safe = false;
                  }
                }
              }
              if (!include_param) {
                if (!isIdenticalStmt(m_Context, view_accesses[i]->getArg(i_arg), view_accesses[j]->getArg(i_arg), true)) {
                  all_other_same_args = false;
                }
              }   
            }

            if (all_other_same_args) {
              if (view_accesses_is_thread_safe[i]) {

                if (!updated_thread_safe) {
                  {
                    unsigned diagID1 = semaRef.Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning, 
                    "The view access might not be thread safe in reverse mode -- continued");
                    clang::Sema::SemaDiagnosticBuilder stream1 = semaRef.Diag(view_accesses[i]->getBeginLoc(), diagID1);
                  }
                  {
                    unsigned diagID2 = semaRef.Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning, 
                    "continued -- due to this view access; an atomic will be required for the reverse mode. ");
                    clang::Sema::SemaDiagnosticBuilder stream2 = semaRef.Diag(view_accesses[j]->getBeginLoc(), diagID2);
                  }
                }

                view_accesses_is_thread_safe[i] = updated_thread_safe;
              }
              if (view_accesses_is_thread_safe[j])
                view_accesses_is_thread_safe[j] = updated_thread_safe;
            }
          }
        }
      }

      bool isAccessThreadSafe(const clang::CXXOperatorCallExpr* view_access) {
        for (size_t i = 0; i < view_accesses_location.size(); ++i) {
          if (view_access->getBeginLoc() == view_accesses_location[i])
            return view_accesses_is_thread_safe[i];
        }
        unsigned diagID = semaRef.Diags.getCustomDiagID(clang::DiagnosticsEngine::Warning, "The view access has not been visited yet. ");
        clang::Sema::SemaDiagnosticBuilder stream = semaRef.Diag(view_access->getBeginLoc(), diagID);
        return false;
      }

      void clear() {
        view_names.clear();
        view_DeclRefExpr.clear();
        view_accesses_is_thread_safe.clear();
        view_accesses.clear();
        view_accesses_RHS.clear();
        view_accesses_location.clear();
      }

      clang::Sema& semaRef;
      clang::ASTContext& m_Context;
      std::vector<std::string> view_names;
      std::vector<const clang::DeclRefExpr*> view_DeclRefExpr;
      std::vector<bool> view_accesses_is_thread_safe;
      std::vector<const clang::CXXOperatorCallExpr*> view_accesses;
      std::vector<bool> view_accesses_RHS;
      std::vector<clang::SourceLocation> view_accesses_location;
  };
} // end namespace clad


#endif // CLAD_KOKKOS_VIEW_ACCESS_VISITOR_H
