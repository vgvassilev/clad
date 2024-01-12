//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_KOKKOS_VIEW_ACCESS_VISITOR_H
#define CLAD_KOKKOS_VIEW_ACCESS_VISITOR_H

#include "clad/Differentiator/CladUtils.h"

namespace clad {

  class KokkosViewAccessVisitor {
    public:
      KokkosViewAccessVisitor (clang::Sema& _semaRef) : semaRef(_semaRef) {}

      void Visit(const clang::Stmt *Node, bool record_view_names = false) {
        if (llvm::isa<clang::CallExpr>(Node)) {
          if (llvm::isa<clang::CXXOperatorCallExpr>(Node)) {
            auto OCE = llvm::dyn_cast<clang::CXXOperatorCallExpr>(Node);

            std::string constructedTypeName = OCE->getDirectCallee()->getQualifiedNameAsString();

            if(constructedTypeName.find("Kokkos::View") != std::string::npos && constructedTypeName.find("::operator()") != std::string::npos) {
              view_accesses.push_back(OCE);
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
          view_accesses_is_thread_safe.push_back(VisitViewAccess(view_accesses[i], params));
        }
      }

      void clear() {
        view_names.clear();
        view_DeclRefExpr.clear();
        view_accesses_is_thread_safe.clear();
        view_accesses.clear();
      }

      clang::Sema& semaRef;
      std::vector<std::string> view_names;
      std::vector<const clang::DeclRefExpr*> view_DeclRefExpr;
      std::vector<bool> view_accesses_is_thread_safe;
      std::vector<const clang::CXXOperatorCallExpr*> view_accesses;
  };
} // end namespace clad


#endif // CLAD_KOKKOS_VIEW_ACCESS_VISITOR_H
