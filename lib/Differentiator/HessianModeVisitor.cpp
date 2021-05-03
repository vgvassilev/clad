//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/HessianModeVisitor.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/Expr.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>

#include "clad/Differentiator/Compatibility.h"

using namespace clang;

namespace clad {
  HessianModeVisitor::HessianModeVisitor(DerivativeBuilder& builder)
      : VisitorBase(builder) {}

  HessianModeVisitor::~HessianModeVisitor() {}

  DeclWithContext HessianModeVisitor::Derive(const clang::FunctionDecl* FD,
                                             const DiffRequest& request) {
    DiffParams args{};
    IndexTable indexes{};
    if (request.Args)
      std::tie(args, indexes) = parseDiffArgs(request.Args, FD);
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

    std::vector<FunctionDecl*> secondDerivativeColumns;

    // Ascertains the independent arguments and differentiates the function
    // in forward and reverse mode by calling ProcessDiffRequest twice each
    // iteration, storing each generated second derivative function
    // (corresponds to columns of Hessian matrix) in a vector for private method
    // merge.
    for (auto independentArg : args) {
      DiffRequest independentArgRequest = request;
      // Converts an independent argument from VarDecl to a StringLiteral Expr
      QualType CharTyConst = m_Context.CharTy.withConst();
      QualType StrTy = clad_compat::getConstantArrayType(
          m_Context,
          CharTyConst,
          llvm::APInt(32, independentArg->getNameAsString().size() + 1),
          nullptr,
          ArrayType::Normal,
          /*IndexTypeQuals*/ 0);
      StringLiteral* independentArgString =
          StringLiteral::Create(m_Context,
                                independentArg->getName(),
                                StringLiteral::Ascii,
                                false,
                                StrTy,
                                noLoc);

      // Derives function once in forward mode w.r.t to independentArg
      independentArgRequest.Args = independentArgString;
      independentArgRequest.Mode = DiffMode::forward;
      independentArgRequest.CallUpdateRequired = false;
      FunctionDecl* firstDerivative =
          plugin::ProcessDiffRequest(m_CladPlugin, independentArgRequest);

      // Further derives function w.r.t to all args in reverse mode
      independentArgRequest.Mode = DiffMode::reverse;
      independentArgRequest.Function = firstDerivative;
      independentArgRequest.Args = nullptr;
      FunctionDecl* secondDerivative =
          plugin::ProcessDiffRequest(m_CladPlugin, independentArgRequest);

      secondDerivativeColumns.push_back(secondDerivative);
    }
    return Merge(secondDerivativeColumns, request);
  }

  // Combines all generated second derivative functions into a
  // single hessian function by creating CallExprs to each individual
  // secon derivative function in FunctionBody.
  DeclWithContext
  HessianModeVisitor::Merge(std::vector<FunctionDecl*> secDerivFuncs,
                            const DiffRequest& request) {
    DiffParams args;
    // request.Function is original function passed in from clad::hessian
    m_Function = request.Function;
    std::copy(m_Function->param_begin(),
              m_Function->param_end(),
              std::back_inserter(args));

    std::string hessianFuncName = request.BaseFunctionName + "_hessian";
    IdentifierInfo* II = &m_Context.Idents.get(hessianFuncName);
    DeclarationNameInfo name(II, noLoc);

    llvm::SmallVector<QualType, 16> paramTypes(m_Function->getNumParams() + 1);

    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(paramTypes),
                   [](const ParmVarDecl* PVD) { return PVD->getType(); });

    paramTypes.back() = m_Context.getPointerType(m_Function->getReturnType());

    QualType hessianFunctionType = m_Context.getFunctionType(
        m_Context.VoidTy,
        llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
        // Cast to function pointer.
        FunctionProtoType::ExtProtoInfo());

    // Create the gradient function declaration.
    DeclContext* DC = const_cast<DeclContext*>(m_Function->getDeclContext());
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(m_CurScope);
    m_Sema.CurContext = DC;

    DeclWithContext result = m_Builder.cloneFunction(m_Function,
                                                     *this,
                                                     DC,
                                                     m_Sema,
                                                     m_Context,
                                                     noLoc,
                                                     name,
                                                     hessianFunctionType);
    FunctionDecl* hessianFD = result.first;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), hessianFD);

    llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(params),
                   [&](const ParmVarDecl* PVD) {
                     auto VD =
                         ParmVarDecl::Create(m_Context,
                                             hessianFD,
                                             noLoc,
                                             noLoc,
                                             PVD->getIdentifier(),
                                             PVD->getType(),
                                             PVD->getTypeSourceInfo(),
                                             PVD->getStorageClass(),
                                             // Clone default arg if present.
                                             (PVD->hasDefaultArg()
                                                  ? Clone(PVD->getDefaultArg())
                                                  : nullptr));
                     if (VD->getIdentifier())
                       m_Sema.PushOnScopeChains(VD,
                                                getCurrentScope(),
                                                /*AddToContext*/ false);
                     auto it = std::find(std::begin(args), std::end(args), PVD);
                     if (it != std::end(args))
                       *it = VD;
                     return VD;
                   });

    // The output paremeter "_result".
    params.back() = ParmVarDecl::Create(
        m_Context,
        hessianFD,
        noLoc,
        noLoc,
        &m_Context.Idents.get("hessianMatrix"),
        paramTypes.back(),
        m_Context.getTrivialTypeSourceInfo(paramTypes.back(), noLoc),
        params.front()->getStorageClass(),
        /* No default value */ nullptr);

    if (params.back()->getIdentifier())
      m_Sema.PushOnScopeChains(params.back(),
                               getCurrentScope(),
                               /*AddToContext*/ false);

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
        llvm::makeArrayRef(params.data(), params.size());
    hessianFD->setParams(paramsRef);
    Expr* m_Result = BuildDeclRef(params.back());
    std::vector<Stmt*> CompStmtSave;

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();

    // Creates callExprs to the second derivative functions genereated
    // and creates maps array elements to input array.
    for (size_t i = 0, e = secDerivFuncs.size(); i < e; ++i) {
      Expr* exprFunc = BuildDeclRef(secDerivFuncs[i]);
      const int numIndependentArgs = secDerivFuncs[i]->getNumParams();

      auto size_type = m_Context.getSizeType();
      auto size_type_bits = m_Context.getIntWidth(size_type);
      // Create the idx literal.
      auto idx = IntegerLiteral::Create(
          m_Context,
          llvm::APInt(size_type_bits, (i * (numIndependentArgs - 1))),
          size_type,
          noLoc);
      // Create the hessianMatrix[idx] expression.
      auto arrayExpr =
          m_Sema.CreateBuiltinArraySubscriptExpr(m_Result, noLoc, idx, noLoc)
              .get();
      // Creates the &hessianMatrix[idx] expression.
      auto addressArrayExpr =
          m_Sema.BuildUnaryOp(nullptr, noLoc, UO_AddrOf, arrayExpr).get();

      // Transforms ParmVarDecls into Expr paramters for insertion into function
      std::vector<Expr*> DeclRefToParams;
      DeclRefToParams.resize(params.size());
      std::transform(params.begin(),
                     std::prev(params.end()),
                     std::begin(DeclRefToParams),
                     [&](ParmVarDecl* PVD) {
                       auto VD = BuildDeclRef(PVD);
                       return VD;
                     });
      DeclRefToParams.pop_back();
      DeclRefToParams.push_back(addressArrayExpr);

      Expr* call =
          m_Sema
              .ActOnCallExpr(getCurrentScope(),
                             exprFunc,
                             noLoc,
                             llvm::MutableArrayRef<Expr*>(DeclRefToParams),
                             noLoc)
              .get();
      CompStmtSave.push_back(call);
    }

    auto StmtsRef =
        llvm::makeArrayRef(CompStmtSave.data(), CompStmtSave.size());
    CompoundStmt* CS =
        clad_compat::CompoundStmt_Create(m_Context, StmtsRef, noLoc, noLoc);
    hessianFD->setBody(CS);
    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return result;
  }
} // end namespace clad