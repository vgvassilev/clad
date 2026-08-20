//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "clad/Differentiator/HessianModeVisitor.h"

#include "ConstantFolder.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/ErrorEstimator.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/StmtClone.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/TemplateBase.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Template.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SaveAndRestore.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace clang;

namespace clad {
HessianModeVisitor::HessianModeVisitor(DerivativeBuilder& builder,
                                       const DiffRequest& request)
    : VisitorBase(builder, request) {}

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
  // FIXME: Find a way to do this without accessing plugin namespace functions
  IndependentArgRequest.UpdateDiffParamsInfo(SemaRef);
  FunctionDecl* firstDerivative =
      Builder.FindDerivedFunction(IndependentArgRequest);

  // Further derives function w.r.t to ReverseModeArgs
  DiffRequest ReverseModeRequest{};
  ReverseModeRequest.Mode = DiffMode::reverse;
  ReverseModeRequest.Function = firstDerivative;
  ReverseModeRequest.Args = ReverseModeArgs;
  ReverseModeRequest.BaseFunctionName = firstDerivative->getNameAsString();
  ReverseModeRequest.m_CladLoopCheckpoints =
      IndependentArgRequest.m_CladLoopCheckpoints;
  // This reverse pass is the bulk of a hessian: it runs once per direction
  // over a function the size of the first-order derivative. Let it use TBR so
  // it only tapes what the reverse sweep reads back.
  ReverseModeRequest.EnableTBRAnalysis =
      IndependentArgRequest.EnableTBRAnalysis;

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
  // Derive the function twice in forward mode.
  IndependentArgRequest.UpdateDiffParamsInfo(SemaRef);
  FunctionDecl* secondDerivative =
      Builder.FindDerivedFunction(IndependentArgRequest);
  return secondDerivative;
}

std::pair<FunctionDecl*, FunctionDecl*>
HessianModeVisitor::DeriveVectorProductFunctions(const DiffParams& args) {
  const FunctionDecl* FD = m_DiffReq.Function;

  // The same factory built the request the planner scheduled, so this finds
  // that derivative rather than starting a second one.
  DiffRequest pushforwardReq = m_DiffReq.pushforwardRequestForHessian(m_Sema);
  FunctionDecl* pushforwardFD = m_Builder.FindDerivedFunction(pushforwardReq);
  // The wrapper builds its calls by position: one tangent per parameter.
  if (!pushforwardFD || pushforwardFD->getNumParams() != 2 * FD->getNumParams())
    return {nullptr, nullptr}; // LCOV_EXCL_LINE

  DiffRequest pullbackReq{};
  pullbackReq.Function = pushforwardFD;
  pullbackReq.BaseFunctionName = pushforwardFD->getNameAsString();
  pullbackReq.Mode = DiffMode::pullback;
  pullbackReq.EnableTBRAnalysis = m_DiffReq.EnableTBRAnalysis;
  pullbackReq.EnableVariedAnalysis = m_DiffReq.EnableVariedAnalysis;
  // The user's checkpointing pragmas address the reverse pass, which in this
  // scheme is the pullback; the per-direction scheme forwarded them to its
  // reverse requests just the same.
  pullbackReq.m_CladLoopCheckpoints = m_DiffReq.m_CladLoopCheckpoints;
  // Adjoints for the parameters the hessian was requested for, in parameter
  // order: those are the rows the wrapper fills in.
  size_t numAdjoints = 0;
  std::string subsetSuffix;
  for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i)
    if (std::find(args.begin(), args.end(), FD->getParamDecl(i)) !=
        args.end()) {
      pullbackReq.DVI.push_back(pushforwardFD->getParamDecl(i));
      ++numAdjoints;
      subsetSuffix += "_" + std::to_string(i);
    }
  // A pullback w.r.t. a subset of the parameters is its own function: two
  // subsets of equal size would otherwise produce two inline definitions
  // with one signature and different bodies, of which the linker silently
  // keeps one for both. Encode the subset the way restricted grads and
  // hessians encode theirs.
  if (numAdjoints != FD->getNumParams())
    pullbackReq.BaseFunctionName += subsetSuffix;

  FunctionDecl* pullbackFD = m_Builder.HandleNestedDiffRequest(pullbackReq);
  // The wrapper builds the call by position, so anything but
  // `(pushforward parameters..., _d_y, adjoints...)` is not ours to call.
  if (!pullbackFD || pullbackFD->getNumParams() !=
                         pushforwardFD->getNumParams() + 1 + numAdjoints)
    return {nullptr, nullptr}; // LCOV_EXCL_LINE
  return {pushforwardFD, pullbackFD};
}

DerivativeAndOverload HessianModeVisitor::BuildHessianFromVectorProducts(
    FunctionDecl* pushforwardFD, FunctionDecl* pullbackFD,
    const DiffParams& args, const IndexIntervalTable& indexIntervalTable,
    llvm::SmallVector<size_t, 16> IndependentArgsSize,
    size_t TotalIndependentArgsSize, const std::string& hessianFuncName,
    DeclContext* DC, QualType hessianFunctionType) {
  const FunctionDecl* FD = m_DiffReq.Function;
  const unsigned numParams = FD->getNumParams();

  // Where each parameter sits in the request: its place in `args` (which
  // `indexIntervalTable` is keyed by) and its place among the requested
  // parameters (which `IndependentArgsSize` is keyed by, in parameter order).
  llvm::SmallVector<int, 8> posInArgs(numParams, -1);
  llvm::SmallVector<int, 8> posInRequested(numParams, -1);
  int requested = 0;
  for (unsigned i = 0; i != numParams; ++i) {
    const auto* it = std::find(args.begin(), args.end(), FD->getParamDecl(i));
    if (it == args.end())
      continue;
    posInArgs[i] = static_cast<int>(it - args.begin());
    posInRequested[i] = requested++;
  }

  IdentifierInfo* II = &m_Context.Idents.get(hessianFuncName);
  DeclarationNameInfo name(II, noLoc);

  llvm::SaveAndRestore<DeclContext*> SaveContext(m_Sema.CurContext);
  llvm::SaveAndRestore<Scope*> SaveScope(getCurrentScope(),
                                         getEnclosingNamespaceOrTUScope());
  m_Sema.CurContext = DC;

  // `result` owns the namespace Scopes cloneFunction opens; its destructor
  // pops them before SaveScope restores.
  ClonedFunction result = m_Builder.cloneFunction(
      m_DiffReq.Function, *this, DC, noLoc, name, hessianFunctionType);
  FunctionDecl* hessianFD = result.fd;

  beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
             Scope::DeclScope);
  m_Sema.PushFunctionScope();
  m_Sema.PushDeclContext(getCurrentScope(), hessianFD);

  llvm::ArrayRef<QualType> paramTypes =
      llvm::cast<FunctionProtoType>(hessianFunctionType)->getParamTypes();
  llvm::SmallVector<ParmVarDecl*, 4> params(paramTypes.size());
  std::transform(m_DiffReq->param_begin(), m_DiffReq->param_end(),
                 std::begin(params), [&](const ParmVarDecl* PVD) {
                   auto* VD = ParmVarDecl::Create(
                       m_Context, hessianFD, noLoc, noLoc, PVD->getIdentifier(),
                       PVD->getType(), PVD->getTypeSourceInfo(),
                       PVD->getStorageClass(),
                       /*DefArg=*/nullptr);
                   if (VD->getIdentifier())
                     m_Sema.PushOnScopeChains(VD, getCurrentScope(),
                                              /*AddToContext=*/false);
                   return VD;
                 });
  params.back() = ParmVarDecl::Create(
      m_Context, hessianFD, noLoc, noLoc,
      &m_Context.Idents.get("hessianMatrix"), paramTypes.back(),
      m_Context.getTrivialTypeSourceInfo(paramTypes.back(), noLoc),
      params.front()->getStorageClass(), /*DefArg=*/nullptr);
  if (params.back()->getIdentifier())
    m_Sema.PushOnScopeChains(params.back(), getCurrentScope(),
                             /*AddToContext=*/false);
  hessianFD->setParams(clad_compat::makeArrayRef(params.data(), params.size()));
  Expr* Result = BuildDeclRef(params.back());

  beginScope(Scope::FnScope | Scope::DeclScope);
  m_DerivativeFnScope = getCurrentScope();

  std::vector<Stmt*> block;

  // The seed that picks the pushforward out of the returned pair: the adjoint
  // of the value is zero, the adjoint of the directional derivative is one, so
  // the pullback returns the derivative of the directional derivative.
  QualType seedType = pullbackFD->getParamDecl(2 * numParams)->getType();
  VarDecl* seedVD = BuildVarDecl(seedType, "_d_y", getZeroInit(seedType),
                                 /*DirectInit=*/true);
  block.push_back(BuildDeclStmt(seedVD));
  Expr* seedRef = BuildDeclRef(seedVD);
  QualType dblType = m_Context.DoubleTy;
  block.push_back(BuildOp(
      BO_Assign,
      utils::BuildMemberExpr(m_Sema, getCurrentScope(), seedRef, "pushforward"),
      ConstantFolder::synthesizeLiteral(dblType, m_Context, /*val=*/1)));

  // One tangent per parameter, reused across directions: the loop seeds a
  // single entry and clears it again. A parameter no direction runs through
  // keeps a zero tangent -- for a pointer that is a null tangent, which
  // forward mode reads as an identically zero derivative.
  llvm::SmallVector<Expr*, 8> tangents(numParams);
  for (unsigned i = 0; i != numParams; ++i) {
    QualType tangentType =
        pushforwardFD->getParamDecl(numParams + i)->getType();
    if (posInArgs[i] < 0) {
      tangents[i] = getZeroInit(tangentType);
      continue;
    }
    QualType paramType = FD->getParamDecl(i)->getType();
    // The seeding assignments below need the variable mutable even when the
    // parameter (and with it the pushforward's tangent) is const.
    QualType storageType = utils::GetNonConstValueType(tangentType);
    if (utils::isArrayOrPointerType(paramType)) {
      // Indices are seeded in place, so the buffer has to reach the last
      // requested one.
      QualType elemType = utils::GetNonConstValueType(paramType);
      QualType sizeType = clad_compat::getSizeType(m_Context);
      storageType = m_Context.getConstantArrayType(
          elemType,
          llvm::APInt(m_Context.getIntWidth(sizeType),
                      indexIntervalTable[posInArgs[i]].Finish),
          /*SizeExpr=*/nullptr, clad_compat::ArraySizeModifier_Normal,
          /*IndexTypeQuals=*/0);
    }
    // Named after the parameter it is the tangent of, as elsewhere in clad.
    VarDecl* VD = BuildVarDecl(storageType,
                               "_d_" + FD->getParamDecl(i)->getNameAsString(),
                               getZeroInit(storageType), /*DirectInit=*/true);
    block.push_back(BuildDeclStmt(VD));
    tangents[i] = BuildDeclRef(VD);
  }

  auto sizeType = clad_compat::getSizeType(m_Context);
  auto sizeTypeBits = m_Context.getIntWidth(sizeType);

  // Row `d` of the hessian is the pullback seeded with the direction `e_d`.
  // Unrolling keeps the mixed scalar-and-array case simple; each direction
  // costs three statements rather than a function of its own.
  size_t row = 0;
  for (unsigned i = 0; i != numParams; ++i) {
    if (posInArgs[i] < 0)
      continue;
    bool isArray = utils::isArrayOrPointerType(FD->getParamDecl(i)->getType());
    size_t start = isArray ? indexIntervalTable[posInArgs[i]].Start : 0;
    size_t finish = isArray ? indexIntervalTable[posInArgs[i]].Finish : 1;

    for (size_t idx = start; idx != finish; ++idx, ++row) {
      // The tangent entry this direction seeds.
      Expr* seeded = CloneNode(tangents[i]);
      if (isArray) {
        Expr* idxExpr = ConstantFolder::synthesizeLiteral(sizeType, m_Context,
                                                          /*val=*/idx);
        seeded = BuildArraySubscript(seeded, idxExpr);
      }
      block.push_back(BuildOp(
          BO_Assign, seeded,
          ConstantFolder::synthesizeLiteral(dblType, m_Context, /*val=*/1)));

      llvm::SmallVector<Expr*, 16> callArgs;
      for (unsigned p = 0; p != numParams; ++p)
        callArgs.push_back(BuildDeclRef(params[p]));
      for (unsigned p = 0; p != numParams; ++p)
        callArgs.push_back(CloneNode(tangents[p]));
      callArgs.push_back(CloneNode(seedRef));
      // The adjoints are the row itself: each requested parameter owns the
      // stretch of the row that the per-direction scheme gives it, so the
      // pullback can accumulate straight into the matrix.
      size_t column = 0;
      for (unsigned p = 0; p != numParams; ++p) {
        if (posInRequested[p] < 0)
          continue;
        llvm::APInt offset(sizeTypeBits,
                           row * TotalIndependentArgsSize + column);
        Expr* offsetArg =
            IntegerLiteral::Create(m_Context, offset, sizeType, noLoc);
        callArgs.push_back(BuildOp(BO_Add, CloneNode(Result), offsetArg));
        column += IndependentArgsSize[posInRequested[p]];
      }
      block.push_back(BuildCallExprToFunction(pullbackFD, callArgs));

      // Clear the seed so the next direction starts from e_0.
      Expr* cleared = CloneNode(tangents[i]);
      if (isArray) {
        Expr* idxExpr = ConstantFolder::synthesizeLiteral(sizeType, m_Context,
                                                          /*val=*/idx);
        cleared = BuildArraySubscript(cleared, idxExpr);
      }
      block.push_back(BuildOp(
          BO_Assign, cleared,
          ConstantFolder::synthesizeLiteral(dblType, m_Context, /*val=*/0)));
    }
  }

  auto stmtsRef = clad_compat::makeArrayRef(block.data(), block.size());
  CompoundStmt* CS = clad_compat::CompoundStmt_Create(
      m_Context,
      stmtsRef /**/ CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam2(
          clang::FPOptionsOverride()),
      noLoc, noLoc);
  hessianFD->setBody(CS);
  endScope(); // Function body scope
  m_Sema.PopFunctionScopeInfo();
  m_Sema.PopDeclContext();
  endScope(); // Function decl scope

  return DerivativeAndOverload{result.fd, /*OverloadFunctionDecl=*/nullptr};
}

DerivativeAndOverload HessianModeVisitor::Derive() {
  const FunctionDecl* FD = m_DiffReq.Function;
  DiffParams args{};
  IndexIntervalTable indexIntervalTable{};
  if (m_DiffReq.Args)
    for (auto dParam : m_DiffReq.DVI) {
      args.push_back(dParam.param);
      indexIntervalTable.push_back(dParam.paramIndexInterval);
    }
  else
    std::copy(FD->param_begin(), FD->param_end(), std::back_inserter(args));

  std::string hessianFuncName = m_DiffReq.BaseFunctionName + "_hessian";
  if (m_DiffReq.Mode == DiffMode::hessian_diagonal)
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

  QualType hessianFunctionType = GetDerivativeType();

  // Check if the function is already declared as a custom derivative.
  // FIXME: We should not use const_cast to get the decl context here.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* DC = const_cast<DeclContext*>(m_DiffReq->getDeclContext());

  // Sizes and validation before any derivation is paid for: how many
  // directions each requested parameter contributes, and the diagnostic for
  // an array parameter that comes without an index interval.
  llvm::SmallVector<size_t, 16> IndependentArgsSize{};
  size_t TotalIndependentArgsSize = 0;
  for (const ParmVarDecl* PVD : FD->parameters()) {
    const auto* it = std::find(std::begin(args), std::end(args), PVD);
    if (it == args.end())
      continue;
    // Using the properties of a vector to find the index of the requested arg
    auto argIndex = it - args.begin();
    if (isArrayOrPointerType(PVD->getType())) {
      if (indexIntervalTable.empty() ||
          indexIntervalTable[argIndex].size() == 0) {
        std::string suggestedArgsStr{};
        if (const auto* SL = dyn_cast<StringLiteral>(
                m_DiffReq.Args->IgnoreParenImpCasts())) {
          llvm::StringRef str = SL->getString().trim();
          llvm::StringRef name{};
          // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
          do {
            std::tie(name, str) = str.split(',');
            if (name.trim().str() == PVD->getNameAsString()) {
              suggestedArgsStr += (suggestedArgsStr.empty() ? "" : ", ") +
                                  PVD->getNameAsString() +
                                  "[0:<last index of " +
                                  PVD->getNameAsString() + ">]";
            } else {
              suggestedArgsStr +=
                  (suggestedArgsStr.empty() ? "" : ", ") + name.trim().str();
            }
          } while (!str.empty());
        } else {
          suggestedArgsStr = PVD->getNameAsString() + "[0:<last index of b>]";
        }
        std::string helperMsg("clad::hessian(" + FD->getNameAsString() +
                              ", \"" + suggestedArgsStr + "\")");
        SourceLocation L = PVD->getBeginLoc();
        if (m_DiffReq.Args)
          L = m_DiffReq.Args->getExprLoc();
        diag(DiagnosticsEngine::Error, L,
             "hessian mode differentiation w.r.t. array or pointer "
             "parameters needs explicit declaration of the indices of the "
             "array using the args parameter; did you mean '%0'")
            << helperMsg << L;
        return {};
      }

      IndependentArgsSize.push_back(indexIntervalTable[argIndex].size());
      TotalIndependentArgsSize += indexIntervalTable[argIndex].size();
    } else {
      IndependentArgsSize.push_back(1);
      TotalIndependentArgsSize++;
    }
  }

  // Deriving per direction emits one second-order function per direction, each
  // the size of a gradient, so the code grows as the square of the parameter
  // count. A hessian-vector product carries the direction at run time instead
  // and needs the same two derivatives however many directions are requested.
  // The planner made that choice and scheduled the pushforward accordingly;
  // per-direction derivatives were not scheduled, so there is no falling back.
  if (m_DiffReq.UseHessianVectorProducts) {
    FunctionDecl* pushforwardFD = nullptr;
    FunctionDecl* pullbackFD = nullptr;
    std::tie(pushforwardFD, pullbackFD) = DeriveVectorProductFunctions(args);
    if (!pullbackFD) {
      // LCOV_EXCL_START
      diag(DiagnosticsEngine::Error, m_DiffReq.CallContext->getBeginLoc(),
           "failed to assemble the hessian of '%0' from hessian-vector "
           "products")
          << FD->getNameAsString();
      return {};
      // LCOV_EXCL_STOP
    }
    return BuildHessianFromVectorProducts(
        pushforwardFD, pullbackFD, args, indexIntervalTable,
        IndependentArgsSize, TotalIndependentArgsSize, hessianFuncName, DC,
        hessianFunctionType);
  }

  // Ascertains the independent arguments and differentiates the function
  // in forward and reverse mode by calling ProcessDiffRequest twice each
  // iteration, storing each generated second derivative function
  // (corresponds to columns of Hessian matrix) in a vector for private method
  // merge.
  std::vector<FunctionDecl*> secondDerivativeFuncs;
  for (const ParmVarDecl* PVD : FD->parameters()) {
    const auto* it = std::find(std::begin(args), std::end(args), PVD);
    if (it == args.end())
      continue;
    auto argIndex = it - args.begin();
    if (isArrayOrPointerType(PVD->getType())) {
      // Derive the function w.r.t. to each requested index of the current
      // array in forward mode and then in reverse mode w.r.t to all
      // requested args
      for (auto i = indexIntervalTable[argIndex].Start;
           i < indexIntervalTable[argIndex].Finish; i++) {
        auto independentArgString =
            PVD->getNameAsString() + "[" + std::to_string(i) + "]";
        auto ForwardModeIASL =
            utils::CreateStringLiteral(m_Context, independentArgString);
        FunctionDecl* DFD = nullptr;
        if (m_DiffReq.Mode == DiffMode::hessian_diagonal)
          DFD = DeriveUsingForwardModeTwice(
              m_Sema, m_CladPlugin, m_Builder, m_DiffReq, ForwardModeIASL,
              m_Builder.m_Scheduler.getDerivedFns());
        else
          DFD = DeriveUsingForwardAndReverseMode(
              m_Sema, m_CladPlugin, m_Builder, m_DiffReq, ForwardModeIASL,
              m_DiffReq.Args, m_Builder.m_Scheduler.getDerivedFns());
        secondDerivativeFuncs.push_back(DFD);
      }
    } else {
      // Derive the function w.r.t. to the current arg in forward mode and
      // then in reverse mode w.r.t to all requested args
      auto* ForwardModeIASL =
          utils::CreateStringLiteral(m_Context, PVD->getNameAsString());
      FunctionDecl* DFD = nullptr;
      if (m_DiffReq.Mode == DiffMode::hessian_diagonal)
        DFD = DeriveUsingForwardModeTwice(
            m_Sema, m_CladPlugin, m_Builder, m_DiffReq, ForwardModeIASL,
            m_Builder.m_Scheduler.getDerivedFns());
      else
        DFD = DeriveUsingForwardAndReverseMode(
            m_Sema, m_CladPlugin, m_Builder, m_DiffReq, ForwardModeIASL,
            m_DiffReq.Args, m_Builder.m_Scheduler.getDerivedFns());
      secondDerivativeFuncs.push_back(DFD);
    }
  }

  return Merge(secondDerivativeFuncs, IndependentArgsSize,
               TotalIndependentArgsSize, hessianFuncName, DC,
               hessianFunctionType);
}

  // Combines all generated second derivative functions into a
  // single hessian function by creating CallExprs to each individual
  // secon derivative function in FunctionBody.
  DerivativeAndOverload
  HessianModeVisitor::Merge(std::vector<FunctionDecl*> secDerivFuncs,
                            SmallVector<size_t, 16> IndependentArgsSize,
                            size_t TotalIndependentArgsSize,
                            const std::string& hessianFuncName, DeclContext* DC,
                            QualType hessianFunctionType) {
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

    // `result` owns the namespace Scopes cloneFunction opens; its
    // destructor pops them before SaveScope restores.
    ClonedFunction result = m_Builder.cloneFunction(
        m_DiffReq.Function, *this, DC, noLoc, name, hessianFunctionType);
    FunctionDecl* hessianFD = result.fd;

    beginScope(Scope::FunctionPrototypeScope | Scope::FunctionDeclarationScope |
               Scope::DeclScope);
    m_Sema.PushFunctionScope();
    m_Sema.PushDeclContext(getCurrentScope(), hessianFD);

    llvm::ArrayRef<QualType> paramTypes =
        llvm::cast<FunctionProtoType>(hessianFunctionType)->getParamTypes();
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
    Expr* Result = BuildDeclRef(params.back());
    std::vector<Stmt*> CompStmtSave;

    beginScope(Scope::FnScope | Scope::DeclScope);
    m_DerivativeFnScope = getCurrentScope();

    // Creates callExprs to the second derivative functions genereated
    // and creates maps array elements to input array.
    for (size_t i = 0, e = secDerivFuncs.size(); i < e; ++i) {
      auto size_type = clad_compat::getSizeType(m_Context);
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
          VarDecl* dThisVD = BuildVarDecl(thisObjectType, "_d_this");
          CompStmtSave.push_back(BuildDeclStmt(dThisVD));
          Expr* dThisExpr = BuildDeclRef(dThisVD);
          DeclRefToParams.push_back(
              BuildOp(UnaryOperatorKind::UO_AddrOf, dThisExpr));
        }
      }

      if (m_DiffReq.Mode == DiffMode::hessian_diagonal) {
        const size_t HessianMatrixStartIndex = i;
        // Call the derived function for second derivative.
        Expr* call = BuildCallExprToFunction(secDerivFuncs[i], DeclRefToParams,
                                             /*CUDAExecConfig=*/nullptr,
                                             /*UseRefQualifiedThisObj=*/true);

        // Create the offset argument.
        llvm::APInt offsetValue(size_type_bits, HessianMatrixStartIndex);
        Expr* OffsetArg =
            IntegerLiteral::Create(m_Context, offsetValue, size_type, noLoc);
        // Create a assignment expression to store the value of call expression
        // into the diagonalHessianVector with index HessianMatrixStartIndex.
        Expr* SliceExprLHS = BuildOp(BO_Add, CloneNode(Result), OffsetArg);
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
          Expr* SliceExpr = BuildOp(BO_Add, CloneNode(Result), OffsetArg);

          DeclRefToParams.push_back(SliceExpr);
          columnIndex += indArgSize;
        }
        Expr* call = BuildCallExprToFunction(secDerivFuncs[i], DeclRefToParams,
                                             /*CUDAExecConfig=*/nullptr,
                                             /*UseRefQualifiedThisObj=*/true);
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

    return DerivativeAndOverload{result.fd,
                                 /*OverloadFunctionDecl=*/nullptr};
  }
} // end namespace clad
