//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/HessianModeVisitor.h"

#include "clad/Differentiator/CladUtils.h"
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
  HessianModeVisitor::HessianModeVisitor(DerivativeBuilder& builder)
      : VisitorBase(builder) {}

  HessianModeVisitor::~HessianModeVisitor() {}

  /// Derives the function w.r.t both forward and reverse mode and returns the
  /// FunctionDecl obtained from reverse mode differentiation
  static FunctionDecl* DeriveUsingForwardAndReverseMode(
      clad::plugin::CladPlugin& CP, DiffRequest IndependentArgRequest,
      const Expr* ForwardModeArgs, const Expr* ReverseModeArgs) {
    // Derives function once in forward mode w.r.t to ForwardModeArgs
    IndependentArgRequest.Args = ForwardModeArgs;
    IndependentArgRequest.Mode = DiffMode::forward;
    IndependentArgRequest.CallUpdateRequired = false;
    // FIXME: Find a way to do this without accessing plugin namespace functions
    FunctionDecl* firstDerivative =
        plugin::ProcessDiffRequest(CP, IndependentArgRequest);

    // Further derives function w.r.t to ReverseModeArgs
    IndependentArgRequest.Mode = DiffMode::reverse;
    IndependentArgRequest.Function = firstDerivative;
    IndependentArgRequest.Args = ReverseModeArgs;
    IndependentArgRequest.BaseFunctionName = firstDerivative->getNameAsString();
    FunctionDecl* secondDerivative =
        plugin::ProcessDiffRequest(CP, IndependentArgRequest);

    return secondDerivative;
  }

  OverloadedDeclWithContext
  HessianModeVisitor::Derive(const clang::FunctionDecl* FD,
                             const DiffRequest& request) {
    DiffParams args{};
    IndexIntervalTable indexIntervalTable{};
    if (request.Args)
      std::tie(args, indexIntervalTable) = parseDiffArgs(request.Args, FD);
    else
      std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

    std::vector<FunctionDecl*> secondDerivativeColumns;
    llvm::SmallVector<size_t, 16> IndependentArgsSize{};
    size_t TotalIndependentArgsSize = 0;

    // request.Function is original function passed in from clad::hessian
    m_Function = request.Function;

    std::string hessianFuncName = request.BaseFunctionName + "_hessian";
    // To be consistent with older tests, nothing is appended to 'f_hessian' if
    // we differentiate w.r.t. all the parameters at once.
    if (!std::equal(m_Function->param_begin(), m_Function->param_end(),
                    std::begin(args))) {
      for (auto arg : args) {
        auto it =
            std::find(m_Function->param_begin(), m_Function->param_end(), arg);
        auto idx = std::distance(m_Function->param_begin(), it);
        hessianFuncName += ('_' + std::to_string(idx));
      }
    }

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
                utils::CreateStringLiteral(m_Context, independentArgString);
            auto DFD =
                DeriveUsingForwardAndReverseMode(m_CladPlugin, request,
                                                 ForwardModeIASL, request.Args);
            secondDerivativeColumns.push_back(DFD);
          }

        } else {
          IndependentArgsSize.push_back(1);
          TotalIndependentArgsSize++;
          // Derive the function w.r.t. to the current arg in forward mode and
          // then in reverse mode w.r.t to all requested args
          auto ForwardModeIASL =
              utils::CreateStringLiteral(m_Context, PVD->getNameAsString());
          auto DFD =
              DeriveUsingForwardAndReverseMode(m_CladPlugin, request,
                                               ForwardModeIASL, request.Args);
          secondDerivativeColumns.push_back(DFD);
        }
      }
    }
    return Merge(secondDerivativeColumns, IndependentArgsSize,
                 TotalIndependentArgsSize, hessianFuncName);
  }

  // Combines all generated second derivative functions into a
  // single hessian function by creating CallExprs to each individual
  // secon derivative function in FunctionBody.
  OverloadedDeclWithContext
  HessianModeVisitor::Merge(std::vector<FunctionDecl*> secDerivFuncs,
                            SmallVector<size_t, 16> IndependentArgsSize,
                            size_t TotalIndependentArgsSize,
                            std::string hessianFuncName) {
    DiffParams args;
    std::copy(m_Function->param_begin(),
              m_Function->param_end(),
              std::back_inserter(args));

    IdentifierInfo* II = &m_Context.Idents.get(hessianFuncName);
    DeclarationNameInfo name(II, noLoc);

    llvm::SmallVector<QualType, 16> paramTypes(m_Function->getNumParams() + 1);

    std::transform(m_Function->param_begin(),
                   m_Function->param_end(),
                   std::begin(paramTypes),
                   [](const ParmVarDecl* PVD) { return PVD->getType(); });

    paramTypes.back() = GetCladArrayRefOfType(m_Function->getReturnType());

    auto originalFnProtoType = cast<FunctionProtoType>(m_Function->getType());
    QualType hessianFunctionType = m_Context.getFunctionType(
        m_Context.VoidTy,
        llvm::ArrayRef<QualType>(paramTypes.data(), paramTypes.size()),
        // Cast to function pointer.
        originalFnProtoType->getExtProtoInfo());

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

    // The output parameter "hessianMatrix".
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
      const size_t HessianMatrixStartIndex = i * TotalIndependentArgsSize;
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

      size_t columnIndex = 0;
      // Create Expr parameters for each independent arg in the CallExpr
      for (size_t j = 0, n = IndependentArgsSize.size(); j < n; j++) {
        llvm::APInt offsetValue(size_type_bits,
                                HessianMatrixStartIndex + columnIndex);
        // Create the offset argument.
        auto OffsetArg =
            IntegerLiteral::Create(m_Context, offsetValue, size_type, noLoc);
        llvm::APInt sizeValue(size_type_bits, IndependentArgsSize[j]);
        // Create the size argument.
        auto SizeArg =
            IntegerLiteral::Create(m_Context, sizeValue, size_type, noLoc);
        SmallVector<Expr*, 2> Args({OffsetArg, SizeArg});
        // Create the hessianMatrix.slice(OffsetArg, SizeArg) expression.
        auto SliceExpr = BuildArrayRefSliceExpr(m_Result, Args);

        DeclRefToParams.push_back(SliceExpr);
        columnIndex += IndependentArgsSize[j];
      }
      Expr* call = BuildCallExprToFunction(secDerivFuncs[i], DeclRefToParams);
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

    return OverloadedDeclWithContext{result.first, result.second,
                                     /*OverloadFunctionDecl=*/nullptr};
  }
} // end namespace clad