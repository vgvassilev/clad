//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/HessianModeVisitor.h"

#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
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
HessianModeVisitor::HessianModeVisitor(DerivativeBuilder& builder,
                                       const DiffRequest& request)
    : VisitorBase(builder, request) {}

/// Converts the string str into a StringLiteral
static const StringLiteral* CreateStringLiteral(ASTContext& C,
                                                const std::string& str) {
  QualType CharTyConst = C.CharTy.withConst();
  QualType StrTy = clad_compat::getConstantArrayType(
      C, CharTyConst, llvm::APInt(/*numBits=*/32, str.size() + 1),
      /*SizeExpr=*/nullptr,
      /*ASM=*/clad_compat::ArraySizeModifier_Normal,
      /*IndexTypeQuals*/ 0);
  const StringLiteral* SL = StringLiteral::Create(
      C, str, /*Kind=*/clad_compat::StringLiteralKind_Ordinary,
      /*Pascal=*/false, StrTy, noLoc);
  return SL;
}

  /// Derives the function w.r.t both forward and reverse mode and returns the
  /// FunctionDecl obtained from reverse mode differentiation
static FunctionDecl* DeriveUsingForwardAndReverseMode(
    Sema& SemaRef, clad::plugin::CladPlugin& CP,
    clad::DerivativeBuilder& Builder, DiffRequest IndependentArgRequest,
    const Expr* ForwardModeArgs, const Expr* ReverseModeArgs,
    DerivedFnCollector& DFC) {
  // Derives function once in forward mode w.r.t to ForwardModeArgs
  IndependentArgRequest.Args = ForwardModeArgs;
  IndependentArgRequest.Mode = DiffMode::forward;
  IndependentArgRequest.CallUpdateRequired = false;
  IndependentArgRequest.UpdateDiffParamsInfo(SemaRef);
  // FIXME: Find a way to do this without accessing plugin namespace functions
  FunctionDecl* firstDerivative =
      Builder.HandleNestedDiffRequest(IndependentArgRequest);

  // Further derives function w.r.t to ReverseModeArgs
  DiffRequest ReverseModeRequest{};
  ReverseModeRequest.Mode = DiffMode::reverse;
  ReverseModeRequest.Function = firstDerivative;
  ReverseModeRequest.Args = ReverseModeArgs;
  ReverseModeRequest.BaseFunctionName = firstDerivative->getNameAsString();
  ReverseModeRequest.UpdateDiffParamsInfo(SemaRef);

  FunctionDecl* secondDerivative =
      Builder.HandleNestedDiffRequest(ReverseModeRequest);
  return secondDerivative;
}

/// Derives the function two times with forward mode AD and returns the
/// FunctionDecl obtained.
static FunctionDecl* DeriveUsingForwardModeTwice(
    Sema& SemaRef, clad::plugin::CladPlugin& CP,
    clad::DerivativeBuilder& Builder, DiffRequest IndependentArgRequest,
    const Expr* ForwardModeArgs, DerivedFnCollector& DFC) {
  // Set derivative order in the request to 2.
  IndependentArgRequest.RequestedDerivativeOrder = 2;
  IndependentArgRequest.Args = ForwardModeArgs;
  IndependentArgRequest.Mode = DiffMode::forward;
  IndependentArgRequest.CallUpdateRequired = false;
  IndependentArgRequest.UpdateDiffParamsInfo(SemaRef);
  // Derive the function twice in forward mode.
  FunctionDecl* secondDerivative =
      Builder.HandleNestedDiffRequest(IndependentArgRequest);
  return secondDerivative;
}

  DerivativeAndOverload
  HessianModeVisitor::Derive(const clang::FunctionDecl* FD,
                             const DiffRequest& request) {
    DiffParams args{};
    IndexIntervalTable indexIntervalTable{};
    DiffInputVarsInfo DVI;
    if (request.Args) {
      DVI = request.DVI;
      for (auto dParam : DVI) {
        args.push_back(dParam.param);
        indexIntervalTable.push_back(dParam.paramIndexInterval);
      }
    }
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

    std::vector<FunctionDecl*> secondDerivativeFuncs;
    llvm::SmallVector<size_t, 16> IndependentArgsSize{};
    size_t TotalIndependentArgsSize = 0;

    // request.Function is original function passed in from clad::hessian
    assert(m_DiffReq == request);

    std::string hessianFuncName = request.BaseFunctionName + "_hessian";
    if (request.Mode == DiffMode::hessian_diagonal)
      hessianFuncName += "_diagonal";
    // To be consistent with older tests, nothing is appended to 'f_hessian' if
    // we differentiate w.r.t. all the parameters at once.
    if (args.size() != FD->getNumParams() ||
        !std::equal(m_DiffReq->param_begin(), m_DiffReq->param_end(),
                    args.begin())) {
      for (auto arg : args) {
        auto it =
            std::find(m_DiffReq->param_begin(), m_DiffReq->param_end(), arg);
        auto idx = std::distance(m_DiffReq->param_begin(), it);
        hessianFuncName += ('_' + std::to_string(idx));
      }
    }

    llvm::SmallVector<QualType, 16> paramTypes(m_DiffReq->getNumParams() + 1);
    std::transform(m_DiffReq->param_begin(), m_DiffReq->param_end(),
                   std::begin(paramTypes),
                   [](const ParmVarDecl* PVD) { return PVD->getType(); });
    paramTypes.back() = m_Context.getPointerType(m_DiffReq->getReturnType());

    const auto* originalFnProtoType =
        cast<FunctionProtoType>(m_DiffReq->getType());
    QualType hessianFunctionType = m_Context.getFunctionType(
        m_Context.VoidTy,
        llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
        // Cast to function pointer.
        originalFnProtoType->getExtProtoInfo());

    // Check if the function is already declared as a custom derivative.
    // FIXME: We should not use const_cast to get the decl context here.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());
    if (FunctionDecl* customDerivative = m_Builder.LookupCustomDerivativeDecl(
            hessianFuncName, DC, hessianFunctionType))
      return DerivativeAndOverload{customDerivative, nullptr};

    // Ascertains the independent arguments and differentiates the function
    // in forward and reverse mode by calling ProcessDiffRequest twice each
    // iteration, storing each generated second derivative function
    // (corresponds to columns of Hessian matrix) in a vector for private method
    // merge.
    for (auto PVD : FD->parameters()) {
      auto it = std::find(std::begin(args), std::end(args), PVD);
      if (it != args.end()) {
        // Using the properties of a vector to find the index of the requested
        // arg
        auto argIndex = it - args.begin();
        if (isArrayOrPointerType(PVD->getType())) {
          if (indexIntervalTable.size() == 0 ||
              indexIntervalTable[argIndex].size() == 0) {
            std::string suggestedArgsStr{};
            if (auto SL = dyn_cast<StringLiteral>(
                    request.Args->IgnoreParenImpCasts())) {
              llvm::StringRef str = SL->getString().trim();
              llvm::StringRef name{};
              do {
                std::tie(name, str) = str.split(',');
                if (name.trim().str() == PVD->getNameAsString()) {
                  suggestedArgsStr += (suggestedArgsStr.empty() ? "" : ", ") +
                                      PVD->getNameAsString() +
                                      "[0:<last index of " +
                                      PVD->getNameAsString() + ">]";
                } else {
                  suggestedArgsStr += (suggestedArgsStr.empty() ? "" : ", ") +
                                      name.trim().str();
                }
              } while (!str.empty());
            } else {
              suggestedArgsStr =
                  PVD->getNameAsString() + "[0:<last index of b>]";
            }
            std::string helperMsg("clad::hessian(" + FD->getNameAsString() +
                                  ", \"" + suggestedArgsStr + "\")");
            diag(DiagnosticsEngine::Error,
                 request.Args ? request.Args->getEndLoc() : noLoc,
                 "Hessian mode differentiation w.r.t. array or pointer "
                 "parameters needs explicit declaration of the indices of the "
                 "array using the args parameter; did you mean '%0'",
                 {helperMsg});
            return {};
          }

          IndependentArgsSize.push_back(indexIntervalTable[argIndex].size());
          TotalIndependentArgsSize += indexIntervalTable[argIndex].size();

          // Derive the function w.r.t. to each requested index of the current
          // array in forward mode and then in reverse mode w.r.t to all
          // requested args
          for (auto i = indexIntervalTable[argIndex].Start;
               i < indexIntervalTable[argIndex].Finish; i++) {
            auto independentArgString =
                PVD->getNameAsString() + "[" + std::to_string(i) + "]";
            auto ForwardModeIASL =
                CreateStringLiteral(m_Context, independentArgString);
            FunctionDecl* DFD = nullptr;
            if (request.Mode == DiffMode::hessian_diagonal)
              DFD = DeriveUsingForwardModeTwice(m_Sema, m_CladPlugin, m_Builder,
                                                request, ForwardModeIASL,
                                                m_Builder.m_DFC);
            else
              DFD = DeriveUsingForwardAndReverseMode(
                  m_Sema, m_CladPlugin, m_Builder, request, ForwardModeIASL,
                  request.Args, m_Builder.m_DFC);
            secondDerivativeFuncs.push_back(DFD);
          }
        } else {
          IndependentArgsSize.push_back(1);
          TotalIndependentArgsSize++;
          // Derive the function w.r.t. to the current arg in forward mode and
          // then in reverse mode w.r.t to all requested args
          auto ForwardModeIASL =
              CreateStringLiteral(m_Context, PVD->getNameAsString());
          FunctionDecl* DFD = nullptr;
          if (request.Mode == DiffMode::hessian_diagonal)
            DFD = DeriveUsingForwardModeTwice(m_Sema, m_CladPlugin, m_Builder,
                                              request, ForwardModeIASL,
                                              m_Builder.m_DFC);
          else
            DFD = DeriveUsingForwardAndReverseMode(
                m_Sema, m_CladPlugin, m_Builder, request, ForwardModeIASL,
                request.Args, m_Builder.m_DFC);
          secondDerivativeFuncs.push_back(DFD);
        }
      }
    }
    return Merge(secondDerivativeFuncs, IndependentArgsSize,
                 TotalIndependentArgsSize, hessianFuncName, DC,
                 hessianFunctionType, paramTypes);
  }

  // Combines all generated second derivative functions into a
  // single hessian function by creating CallExprs to each individual
  // secon derivative function in FunctionBody.
  DerivativeAndOverload
  HessianModeVisitor::Merge(std::vector<FunctionDecl*> secDerivFuncs,
                            SmallVector<size_t, 16> IndependentArgsSize,
                            size_t TotalIndependentArgsSize,
                            const std::string& hessianFuncName, DeclContext* DC,
                            QualType hessianFunctionType,
                            llvm::SmallVector<QualType, 16> paramTypes) {
    DiffParams args;
    std::copy(m_DiffReq->param_begin(), m_DiffReq->param_end(),
              std::back_inserter(args));

    IdentifierInfo* II = &m_Context.Idents.get(hessianFuncName);
    DeclarationNameInfo name(II, noLoc);

    // Create the gradient function declaration.
    llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
    llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope(),
                                           getEnclosingNamespaceOrTUScope());
    m_Sema.CurContext = DC;

    DeclWithContext result = m_Builder.cloneFunction(
        m_DiffReq.Function, *this, DC, noLoc, name, hessianFunctionType);
    FunctionDecl* hessianFD = result.first;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), hessianFD);

    llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
    std::transform(m_DiffReq->param_begin(), m_DiffReq->param_end(),
                   std::begin(params), [&](const ParmVarDecl* PVD) {
                     auto VD =
                         ParmVarDecl::Create(m_Context,
                                             hessianFD,
                                             noLoc,
                                             noLoc,
                                             PVD->getIdentifier(),
                                             PVD->getType(),
                                             PVD->getTypeSourceInfo(),
                                             PVD->getStorageClass(),
                                             /*DefArg=*/nullptr);
                     if (VD->getIdentifier())
                       m_Sema.PushOnScopeChains(VD,
                                                getCurrentScope(),
                                                /*AddToContext*/ false);
                     auto it = std::find(std::begin(args), std::end(args), PVD);
                     if (it != std::end(args))
                       *it = VD;
                     return VD;
                   });

    // The output parameter "hessianMatrix" or "diagonalHessianVector"
    std::string outputParamName = "hessianMatrix";
    if (m_DiffReq.Mode == DiffMode::hessian_diagonal)
      outputParamName = "diagonalHessianVector";
    params.back() = ParmVarDecl::Create(
        m_Context, hessianFD, noLoc, noLoc,
        &m_Context.Idents.get(outputParamName), paramTypes.back(),
        m_Context.getTrivialTypeSourceInfo(paramTypes.back(), noLoc),
        params.front()->getStorageClass(),
        /* No default value */ nullptr);

    if (params.back()->getIdentifier())
      m_Sema.PushOnScopeChains(params.back(),
                               getCurrentScope(),
                               /*AddToContext*/ false);

    llvm::ArrayRef<ParmVarDecl*> paramsRef =
        clad_compat::makeArrayRef(params.data(), params.size());
    hessianFD->setParams(paramsRef);
    Expr* m_Result = BuildDeclRef(params.back());
    std::vector<Stmt*> CompStmtSave;

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();

    // Creates callExprs to the second derivative functions genereated
    // and creates maps array elements to input array.
    for (size_t i = 0, e = secDerivFuncs.size(); i < e; ++i) {
      auto size_type = m_Context.getSizeType();
      auto size_type_bits = m_Context.getIntWidth(size_type);

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

      /// If we are differentiating a member function then create a parameter
      /// that can represent the derivative for the implicit `this` pointer. It
      /// is required because reverse mode derived function expects an explicit
      /// parameter for storing derivative with respect to `implicit` this
      /// object.
      ///
      // FIXME: Add support for class type in the hessian matrix. For this, we
      // need to add a way to represent hessian matrix when class type objects
      // are involved.
      if (const auto* MD = dyn_cast<CXXMethodDecl>(m_DiffReq.Function)) {
        const CXXRecordDecl* RD = MD->getParent();
        if (MD->isInstance() && !RD->isLambda()) {
          QualType thisObjectType =
              clad_compat::CXXMethodDecl_GetThisObjectType(m_Sema, MD);
          // Derivatives should never be of `const` types. Even if the original 
          // variable is of `const` type. This behaviour is consistent with the built-in
          // scalar numerical types as well.
          thisObjectType.removeLocalConst();
          auto dThisVD = BuildVarDecl(thisObjectType, "_d_this",
                                      /*Init=*/nullptr, false, /*TSI=*/nullptr,
                                      VarDecl::InitializationStyle::CallInit);
          CompStmtSave.push_back(BuildDeclStmt(dThisVD));
          Expr* dThisExpr = BuildDeclRef(dThisVD);
          DeclRefToParams.push_back(
              BuildOp(UnaryOperatorKind::UO_AddrOf, dThisExpr));
        }
      }

      if (m_DiffReq.Mode == DiffMode::hessian_diagonal) {
        const size_t HessianMatrixStartIndex = i;
        // Call the derived function for second derivative.
        Expr* call = BuildCallExprToFunction(secDerivFuncs[i], DeclRefToParams);

        // Create the offset argument.
        llvm::APInt offsetValue(size_type_bits, HessianMatrixStartIndex);
        Expr* OffsetArg =
            IntegerLiteral::Create(m_Context, offsetValue, size_type, noLoc);
        // Create a assignment expression to store the value of call expression
        // into the diagonalHessianVector with index HessianMatrixStartIndex.
        Expr* SliceExprLHS = BuildOp(BO_Add, m_Result, OffsetArg);
        Expr* DerefExpr = BuildOp(UO_Deref, BuildParens(SliceExprLHS));
        Expr* AssignExpr = BuildOp(BO_Assign, DerefExpr, call);
        CompStmtSave.push_back(AssignExpr);
      } else {
        const size_t HessianMatrixStartIndex = i * TotalIndependentArgsSize;
        size_t columnIndex = 0;
        // Create Expr parameters for each independent arg in the CallExpr
        for (size_t indArgSize : IndependentArgsSize) {
          llvm::APInt offsetValue(size_type_bits,
                                  HessianMatrixStartIndex + columnIndex);
          // Create the offset argument.
          Expr* OffsetArg =
              IntegerLiteral::Create(m_Context, offsetValue, size_type, noLoc);
          // Create the hessianMatrix + OffsetArg expression.
          Expr* SliceExpr = BuildOp(BO_Add, m_Result, OffsetArg);

          DeclRefToParams.push_back(SliceExpr);
          columnIndex += indArgSize;
        }
        Expr* call = BuildCallExprToFunction(secDerivFuncs[i], DeclRefToParams);
        CompStmtSave.push_back(call);
      }
    }

    auto StmtsRef =
        clad_compat::makeArrayRef(CompStmtSave.data(), CompStmtSave.size());
    CompoundStmt* CS =
        clad_compat::CompoundStmt_Create(m_Context, StmtsRef /**/ CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam2(clang::FPOptionsOverride()), noLoc, noLoc);
    hessianFD->setBody(CS);
    endScope(); // Function body scope
    m_Sema.PopFunctionScopeInfo();
    m_Sema.PopDeclContext();
    endScope(); // Function decl scope

    return DerivativeAndOverload{result.first,
                                 /*OverloadFunctionDecl=*/nullptr};
  }
} // end namespace clad
