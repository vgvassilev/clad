//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "ClangPlugin.h"

#include "clad/Differentiator/DerivativeBuilder.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/Sins.h"
#include "clad/Differentiator/Timers.h"
#include "clad/Differentiator/Version.h"
#include "../lib/Differentiator/DerivativePrinter.h"
#include "../lib/Differentiator/GeneratedCode.h"
#include "../lib/Differentiator/TBRAnalyzer.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/LLVM.h" // isa, dyn_cast
#include "clang/Basic/SourceLocation.h"
// Clang 17 moved the debug info kinds out of clang::codegenoptions and into
// llvm::codegenoptions.
#if CLANG_VERSION_MAJOR < 17
#include "clang/Basic/DebugInfoOptions.h"
#else
#include "llvm/Frontend/Debug/Options.h"
#endif

#ifdef _WIN32
// <windows.h> defines function-like min/max macros that mangle
// std::numeric_limits<>::max() in llvm/ADT/Sequence.h (included below);
// NOMINMAX suppresses them. WIN32_LEAN_AND_MEAN trims unrelated Win32 surface.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Frontend/MultiplexConsumer.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"
#include "clad/Differentiator/DiffMode.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>  // for getenv
#include <iostream> // for std::cerr
#include <memory>
#include <set>
#include <utility>

using namespace clang;

namespace clad {
void InitTimers();

  namespace plugin {
    /// Keeps track if we encountered #pragma clad on/off.
    // FIXME: Figure out how to make it a member of CladPlugin.
    std::vector<clang::SourceRange> CladEnabledRange;
    std::set<clang::SourceLocation> CladLoopCheckpoints;

    // Define a pragma handler for #pragma clad
    class CladPragmaHandler : public PragmaHandler {
    public:
      CladPragmaHandler() : PragmaHandler("clad") {}
      void HandlePragma(Preprocessor& PP, PragmaIntroducer Introducer,
                        Token& PragmaTok) override {
        if (PragmaTok.isNot(tok::identifier)) {
          PP.Diag(PragmaTok, diag::warn_pragma_diagnostic_invalid);
          return;
        }
#ifndef NDEBUG
        IdentifierInfo* II = PragmaTok.getIdentifierInfo();
        assert(II->isStr("clad"));
#endif

        PP.Lex(PragmaTok);
        llvm::StringRef OptionName = PragmaTok.getIdentifierInfo()->getName();
        SourceLocation TokLoc = PragmaTok.getLocation();
        // Handle #pragma clad ON
        if (OptionName == "ON") {
          SourceRange R(TokLoc, /*end*/ SourceLocation());
          // If a second ON is seen, ignore it if the interval is open.
          if (CladEnabledRange.empty() ||
              CladEnabledRange.back().getEnd().isValid())
            CladEnabledRange.push_back(R);
          return;
        }
        // Handle #pragma clad OFF/DEFAULT
        if (OptionName == "OFF" || OptionName == "DEFAULT") {
          if (!CladEnabledRange.empty()) {
            assert(CladEnabledRange.back().getEnd().isInvalid());
            CladEnabledRange.back().setEnd(TokLoc);
          }
          return;
        }
        // Handle #pragma clad checkpoint loop
        if (OptionName == "checkpoint") {
          PP.Lex(PragmaTok);
          // Ensure the next token is `loop`
          if (PragmaTok.isNot(tok::identifier) ||
              PragmaTok.getIdentifierInfo()->getName() != "loop") {
            PP.Diag(PragmaTok.getLocation(),
                    PP.getDiagnostics().getCustomDiagID(
                        DiagnosticsEngine::Error,
                        "expected 'loop' after 'checkpoint' in #pragma clad"));
            return;
          }
          CladLoopCheckpoints.insert(PragmaTok.getLocation());
          return;
        }
        // Diagnose unknown clad pragma option
        PP.Diag(
            TokLoc,
            PP.getDiagnostics().getCustomDiagID(
                DiagnosticsEngine::Error,
                "expected 'ON', 'OFF', 'DEFAULT', or `checkpoint` in pragma"));
      }
    };

    CladPlugin::CladPlugin(CompilerInstance& CI, DifferentiationOptions& DO)
        : m_CI(CI), m_DO(DO), m_HasRuntime(false) {
      CodeGenOptions& CGOpts = m_CI.getCodeGenOpts();
      bool WantTiming = CGOpts.TimePasses;

      if (WantTiming || getenv("CLAD_ENABLE_TIMING"))
        InitTimers();

        // Register clad as a backend pass via the path of clad.so itself,
        // resolved from any symbol we own. Cleaner than iterating
        // CI.getFrontendOpts().Plugins (which depends on how clang was
        // invoked) and keeps the lookup inside this DSO.
#ifdef CLAD_BUILD_STATIC_ONLY
        // Skip registration entirely if clad is statically linked
#elif _WIN32
      HMODULE hm = nullptr;
      if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                 GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(&InitTimers), &hm) &&
          hm) {
        char buf[MAX_PATH];
        if (DWORD n = GetModuleFileNameA(hm, buf, MAX_PATH);
            n > 0 && n < MAX_PATH)
          CGOpts.PassPlugins.emplace_back(buf);
      }
#else
      if (Dl_info info;
          dladdr(reinterpret_cast<void*>(&InitTimers), &info) && info.dli_fname)
        CGOpts.PassPlugins.emplace_back(info.dli_fname);
#endif

      // Add define for __CLAD__, so that CladFunction::CladFunction()
      // doesn't throw an error.
      auto predefines = m_CI.getPreprocessor().getPredefines();
      predefines.append("#define __CLAD__ 1\n");
      m_CI.getPreprocessor().setPredefines(predefines);
    }

    CladPlugin::~CladPlugin() {}

    ALLOW_ACCESS(MultiplexConsumer, Consumers,
                 std::vector<std::unique_ptr<ASTConsumer>>);

    void CladPlugin::Initialize(clang::ASTContext& C) {
      // We know we have a multiplexer. We commit a sin here by stealing it and
      // making the consumer pass-through so that we can delay all operations
      // until clad is happy.

      auto& MultiplexC = cast<MultiplexConsumer>(m_CI.getASTConsumer());
      auto& RobbedCs = ACCESS(MultiplexC, Consumers);
      assert(RobbedCs.back().get() == this && "Clad is not the last consumer");

      const auto& Macros = m_CI.getPreprocessorOpts().Macros;
      const bool IsCling = llvm::any_of(
          Macros, [](const auto& Macro) { return Macro.first == "__CLING__"; });
      if (IsCling && m_CI.getPreprocessor().isIncrementalProcessingEnabled()) {
        std::swap(RobbedCs.front(), RobbedCs.back());
        return;
      }
      std::vector<std::unique_ptr<ASTConsumer>> StolenConsumers;

      // The range-based for loop in MultiplexConsumer::Initialize has
      // dispatched this call. Generally, it is unsafe to delete elements while
      // iterating but we know we are in the end of the loop and ::end() won't
      // be invalidated.
      std::move(RobbedCs.begin(), RobbedCs.end() - 1,
                std::back_inserter(StolenConsumers));
      RobbedCs.erase(RobbedCs.begin(), RobbedCs.end() - 1);
      m_Multiplexer.reset(new MultiplexConsumer(std::move(StolenConsumers)));
    }

    void CladPlugin::HandleTopLevelDeclForClad(DeclGroupRef DGR) {
      if (!CheckBuiltins())
        return;
#if CLANG_VERSION_MAJOR > 16
      // Traverse all constexpr FunctionDecls for the static graph only once to
      // differentiate them immeditely.
      {
        TimedAnalysisRegion R("Rest of constexpr TU");
        for (Decl* D : DGR) {
          if (!isa<FunctionDecl>(D))
            continue;
          auto* FD = cast<FunctionDecl>(D);
          if (FD->isConstexpr() || !m_Multiplexer) {
            getScheduler().Plan(DGR);
            break;
          }
        }
      }

      // This handler can be re-entered while a planning traversal is on the
      // stack: a lookup issued by the traversal makes the ASTReader
      // deserialize pending module decls and pass them to the consumers. The
      // Plan call above then defers the group; processing requests here would
      // interleave with the outer traversal (and clobber its current
      // processing node), so leave them to the outer caller.
      if (!getScheduler().isTraversalInFlight())
        for (DiffRequest& request : getScheduler().getGraph().getNodes()) {
          if (request.ImmediateMode && request.Function->isConstexpr()) {
            getScheduler().getGraph().setCurrentProcessingNode(request);
            ProcessDiffRequest(request);
            getScheduler().getGraph().markCurrentNodeProcessed();
          }
        }
#endif

      // We could not delay the processing of derivatives, act as if each
      // call is final. That would still have vgvassilev/clad#248 unresolved.
      // Not on re-entry (see above): the outer traversal finalizes.
      if (!m_CI.getDiagnostics().hasErrorOccurred() &&
          !getScheduler().isTraversalInFlight() && !m_Multiplexer)
        FinalizeTranslationUnit();
    }

    /// The statements of \p S that the printer gave an offset, paired with it.
    static void collectStmts(const clang::Stmt* S, DerivativePrinter& Printer,
                             const clang::FunctionDecl* FD,
                             llvm::SmallVectorImpl<const clang::Stmt*>& Out) {
      if (!S)
        return;
      if (Printer.locationOf(FD, S).isValid())
        Out.push_back(S);
      for (const clang::Stmt* Child : S->children())
        collectStmts(Child, Printer, FD, Out);
    }

#if CLANG_VERSION_MAJOR < 17
    namespace codegenoptions = clang::codegenoptions;
#else
    namespace codegenoptions = llvm::codegenoptions;
#endif

    /// Every statement under \p S standing at a slot, outermost first.
    static void collectSlotted(const clang::Stmt* S, GeneratedCode& Locs,
                               llvm::SmallVectorImpl<const clang::Stmt*>& Out) {
      if (!S)
        return;
      if (Locs.owns(S->getBeginLoc()))
        Out.push_back(S);
      for (const clang::Stmt* Child : S->children())
        collectSlotted(Child, Locs, Out);
    }

    void CladPlugin::materializeGeneratedCode() {
      if (m_Generated.empty() || !m_DerivativeBuilder)
        return;
      // The line notes are for debug information alone: they decide what
      // CodeGen writes into the line table. A diagnostic reads the printed
      // code directly.
      const bool WantLineNotes =
          m_CI.getCodeGenOpts().getDebugInfo() != codegenoptions::NoDebugInfo;
      if (!WantLineNotes && !m_DO.RemarkTBRAnalysis &&
          !m_DO.DumpGeneratedSource)
        return; // Nobody will read the code, so nobody has to print it.

      // A derivative that gets a forward declaration ahead of its definition
      // is recorded twice. Only the declaration carrying the body has
      // anything of its own to say.
      llvm::erase_if(m_Generated, [](const Generated& G) {
        return !G.Derivative->doesThisDeclarationHaveABody();
      });

      GeneratedCode& Code = m_DerivativeBuilder->getGeneratedCode();
      DerivativePrinter& Printer = getDerivativePrinter();

      // Print first, ask afterwards: a buffer's lines are settled by the
      // first question about it, and a derivative printed after that would
      // sit on lines nobody knows are there.
      for (const Generated& G : m_Generated) {
        clang::FunctionDecl* FD = G.Derivative;
        // Printing it is what lineOf does on the way to answering.
        const unsigned Signature = Printer.lineOf(FD);
        if (!WantLineNotes)
          continue;

        // Outermost first, so that an inner statement assigns over the outer
        // one it sits in and the most specific answer is the one that stays.
        llvm::SmallVector<const clang::Stmt*, 64> Nodes;
        collectSlotted(FD->getBody(), Code, Nodes);
        if (Nodes.empty())
          continue;

        // A note can only name a line of its own file, so a statement whose
        // slots landed in another chunk than the printout is left alone. That
        // is a derivative with more nodes than one chunk holds: it runs on
        // into the next chunk, the printout stays where it started.
        const clang::SourceLocation Printed = Printer.startOf(FD);

        // The brace clad built the body with stands for the signature, so
        // stopping at the function by name lands on its first line.
        const clang::SourceLocation Brace = FD->getBody()->getBeginLoc();
        if (Code.isSameChunk(Brace, Printed))
          Code.assign(Brace, FD->getEndLoc(), Signature);

        for (const clang::Stmt* S : Nodes)
          if (Code.isSameChunk(S->getBeginLoc(), Printed))
            Code.assign(S->getBeginLoc(), S->getEndLoc(),
                        Printer.lineOf(FD, S));
      }

      // Everything is printed. What the SourceManager worked out about these
      // buffers while the derivatives were being built describes an emptier
      // version of them, so drop it before anything asks.
      Code.forgetLineTables();

      if (WantLineNotes) {
        Code.present();
        Code.writeChunksToFiles(m_CI.getDiagnostics());
        adviseOnGeneratedSource();
      }

      for (const Generated& G : m_Generated) {
        if (m_DO.DumpGeneratedSource)
          dumpGeneratedSource(G.Derivative);
        emitAnalysisRemarks(G);
      }

      // All of this has been read now, and incremental compilation comes
      // back with more derivatives to print somewhere else.
      Code.seal();
      m_Generated.clear();
    }

    void CladPlugin::adviseOnGeneratedSource() const {
      // Only when there is nothing to read. A debugger reaches the code clad
      // generated either through the object, which needs -gembed-source and a
      // debugger that understands it, or through a file on disk.
      // Naming the flag at all is an answer, even with no directory after it:
      // it says the generated code being unreadable is known and meant.
      if (m_DO.GeneratedSourceDirGiven)
        return;
      const clang::CodeGenOptions& CGO = m_CI.getCodeGenOpts();
      // Through the enum's own type rather than by name: which header spells
      // it, and in which namespace, has moved between releases.
      const auto Tuning = CGO.getDebuggerTuning();
      const bool ReadsEmbedded = Tuning == decltype(Tuning)::LLDB;
      if (CGO.EmbedSource && ReadsEmbedded)
        return;
      const char* Advice =
          ReadsEmbedded ? "add -gembed-source, or -plugin-arg-clad "
                          "-fgenerated-source-dir=<dir>"
                        : "add -plugin-arg-clad -fgenerated-source-dir=<dir>";
      unsigned ID = m_CI.getDiagnostics().getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "debug information for the code clad generated points at no source a "
          "debugger can open; %0");
      m_CI.getDiagnostics().Report(ID) << Advice;
    }

    void CladPlugin::dumpGeneratedSource(clang::Decl* D) {
      const auto* FD = llvm::dyn_cast<clang::FunctionDecl>(D);
      if (!FD || !FD->getBody())
        return;

      llvm::outs() << "generated-source: " << FD->getNameAsString() << "\n";
      // Every statement the printer announced, in the order its text appears.
      llvm::SmallVector<const clang::Stmt*, 32> Ordered;
      collectStmts(FD->getBody(), getDerivativePrinter(), FD, Ordered);
      clang::SourceManager& SM = m_CI.getSourceManager();
      // Into the printed text, not into the buffer it sits in: every
      // derivative printed after the first starts part-way through its buffer,
      // and indexing its text by a buffer offset runs off the end of it.
      auto offsetOf = [&](const clang::Stmt* S) {
        return getDerivativePrinter().offsetOf(FD, S);
      };
      llvm::stable_sort(Ordered,
                        [&](const clang::Stmt* A, const clang::Stmt* B) {
                          return offsetOf(A) < offsetOf(B);
                        });
      llvm::StringRef Text = getDerivativePrinter().textOf(FD);
      unsigned Last = ~0U;
      for (const clang::Stmt* S : Ordered) {
        // One line per position: the innermost and outermost node starting at
        // the same character would otherwise repeat it.
        unsigned Offset = offsetOf(S);
        if (Offset == Last)
          continue;
        Last = Offset;
        clang::SourceRange R = getDerivativePrinter().rangeOf(FD, S);
        clang::PresumedLoc B = SM.getPresumedLoc(R.getBegin());
        clang::PresumedLoc E = SM.getPresumedLoc(R.getEnd());
        // Begin and end columns, so a test can pin how wide a node is and not
        // just where it starts.
        // A node measured to one line reports its span; one that could not be
        // measured reports just where it starts.
        llvm::outs() << "  " << B.getLine() << ":" << B.getColumn();
        if (R.getEnd().isValid() && E.getLine() == B.getLine() &&
            E.getColumn() >= B.getColumn())
          llvm::outs() << "-" << E.getColumn();
        llvm::outs() << ": " << Text.drop_front(Offset).take_until([](char C) {
          return C == '\n';
        }) << "\n";
      }
    }

    /// Whether \p VD holds a value clad kept for the reverse sweep.
    ///
    /// Clad names its temporaries `_t<N>`, but not all of them are kept
    /// values: a tape is the container for values kept in a loop, and a loop
    /// counter is bookkeeping. Reporting either as a cost the analysis failed
    /// to remove would be wrong, not merely noisy.
    // FIXME: This reads clad's naming convention from the outside. The
    // durable answer is for the visitor to mark a store as it emits one.
    static bool isKeptValue(const clang::VarDecl* VD) {
      if (!VD->hasInit() || !VD->getName().starts_with("_t"))
        return false;
      if (VD->getType().getAsString().find("tape") != std::string::npos)
        return false;
      return !llvm::isa<clang::IntegerLiteral>(
          VD->getInit()->IgnoreParenImpCasts());
    }

    /// The values a derivative keeps for its reverse sweep, in the order they
    /// appear. Clad names them `_t<N>`; a value kept in a loop goes onto a
    /// tape instead, so both shapes count.
    static void collectKeptValues(
        const clang::Stmt* S,
        llvm::SmallVectorImpl<
            std::pair<const clang::Stmt*, const clang::VarDecl*>>& Out) {
      if (!S)
        return;
      if (const auto* DS = llvm::dyn_cast<clang::DeclStmt>(S))
        for (const clang::Decl* D : DS->decls())
          if (const auto* VD = llvm::dyn_cast<clang::VarDecl>(D))
            if (isKeptValue(VD))
              Out.emplace_back(S, VD);
      // A value kept once per iteration goes onto a tape rather than into its
      // own variable. The push is the thing that costs, not the tape.
      if (const auto* CE = llvm::dyn_cast<clang::CallExpr>(S))
        if (const clang::FunctionDecl* Callee = CE->getDirectCallee())
          if (Callee->getName() == "push" && CE->getNumArgs() == 2)
            Out.emplace_back(S, nullptr);
      for (const clang::Stmt* Child : S->children())
        collectKeptValues(Child, Out);
    }

    /// The primal expression itself, so the note underlines all of it. The
    /// clone kept its original range, so this is the user's own code.
    static clang::SourceRange primalRangeOf(const clang::Stmt* K,
                                            const clang::VarDecl* VD) {
      const clang::Expr* Init =
          VD ? VD->getInit() : llvm::cast<clang::CallExpr>(K)->getArg(1);
      return Init ? Init->getSourceRange() : clang::SourceRange();
    }

    /// Where in the user's code the kept value came from.
    ///
    /// The stored initializer is a clone of a primal expression and carries
    /// that expression's location, so a note can point at code the user can
    /// actually change -- which the remark itself cannot, since it points at
    /// code clad wrote. Nodes clad invented outright carry GetValidSLoc, the
    /// start of the main file; a caret on line one would be worse than saying
    /// nothing, so those report no location at all.
    static clang::SourceLocation
    primalLocationOf(const clang::Stmt* K, const clang::VarDecl* VD,
                     const clang::SourceManager& SM) {
      const clang::Expr* Init = nullptr;
      if (VD)
        Init = VD->getInit();
      else if (const auto* CE = llvm::dyn_cast<clang::CallExpr>(K))
        Init = CE->getArg(1); // what the push saves
      if (!Init)
        return {};
      clang::SourceLocation Loc = Init->getBeginLoc();
      if (Loc.isInvalid() || !Loc.isFileID())
        return {};
      if (Loc == SM.getLocForStartOfFile(SM.getMainFileID()))
        return {};
      return Loc;
    }

    void CladPlugin::emitAnalysisRemarks(const Generated& G) {
      if (!m_DO.RemarkTBRAnalysis)
        return;
      const clang::FunctionDecl* FD = G.Derivative;
      if (!FD->getBody())
        return;

      llvm::SmallVector<std::pair<const clang::Stmt*, const clang::VarDecl*>,
                        16>
          Kept;
      collectKeptValues(FD->getBody(), Kept);
      if (Kept.empty())
        return; // Nothing to say, so nothing gets rendered.

      clang::Sema& S = m_CI.getSema();
      // Why the value is still here is a different sentence depending on
      // whether the analysis ran at all; saying "could not prove" when it was
      // switched off would be false.
      const char* Because =
          G.AnalysisRan ? "to-be-recorded analysis could not show it unused"
                        : "to-be-recorded analysis is disabled";
      const clang::SourceManager& SM = m_CI.getSourceManager();
      for (const auto& [K, VD] : Kept) {
        clang::SourceLocation Loc = getDerivativePrinter().locationOf(FD, K);
        if (Loc.isInvalid())
          continue;
        utils::diag(S, clang::DiagnosticsEngine::Remark, Loc,
                    "clad keeps this value for the reverse sweep")
            << getDerivativePrinter().rangeOf(FD, K);
        utils::diag(S, clang::DiagnosticsEngine::Note, Loc, "%0") << Because;
        // Which derivative this is, said outright rather than left to the
        // include stack, which has no caret and no wording.
        if (G.RequestedAt.isValid())
          utils::diag(S, clang::DiagnosticsEngine::Note, G.RequestedAt,
                      "in the derivative of '%0' requested here")
              << G.Original->getNameAsString();
        // The half the user can act on: the expression in their own code
        // whose value this is.
        if (clang::SourceLocation Primal = primalLocationOf(K, VD, SM);
            Primal.isValid())
          utils::diag(S, clang::DiagnosticsEngine::Note, Primal,
                      "the value kept is the one this expression had")
              << primalRangeOf(K, VD);
      }
    }

    static void printDerivative(clang::Decl* D, bool DeclarationOnly,
                                const DifferentiationOptions& DO) {
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      Policy.Bool = true;

      // if enabled, print source code of the derivatives
      if (DO.DumpDerivedFn) {
        D->print(llvm::outs(), Policy);
        if (DeclarationOnly)
          llvm::outs() << ";\n";
      }

      // if enabled, print ASTs of the derivatives
      if (DO.DumpDerivedAST)
        D->dumpColor();

      // if enabled, print the derivatives in a file
      if (DO.GenerateSourceFile) {
        std::error_code err;
        llvm::raw_fd_ostream f("Derivatives.cpp", err,
                               CLAD_COMPAT_llvm_sys_fs_Append);
        D->print(f, Policy);
        if (DeclarationOnly)
          f << ";\n";
        f.flush();
      }
    }

    class AttachedLoopStmtFinder
        : public RecursiveASTVisitor<AttachedLoopStmtFinder> {
      SourceLocation m_PragmaLoc;
      SourceManager& m_SM;
      Stmt* m_AttachedStmt = nullptr;
      SourceLocation m_AttachedLoopLoc;

    public:
      AttachedLoopStmtFinder(SourceLocation pragmaLoc, SourceManager& SM)
          : m_PragmaLoc(pragmaLoc), m_SM(SM) {}

      bool VisitStmt(Stmt* S) {
        SourceLocation beginLoc = S->getBeginLoc();
        if (!beginLoc.isValid() ||
            !m_SM.isBeforeInTranslationUnit(m_PragmaLoc, beginLoc))
          return true;

        if (!m_AttachedStmt || m_SM.isBeforeInTranslationUnit(
                                   beginLoc, m_AttachedStmt->getBeginLoc())) {
          m_AttachedStmt = S;
          m_AttachedLoopLoc = {};
          if (isa<ForStmt>(S) || isa<WhileStmt>(S) || isa<DoStmt>(S))
            m_AttachedLoopLoc = beginLoc;
        }
        return true;
      }

      [[nodiscard]] SourceLocation getAttachedLoopLoc() const {
        return m_AttachedLoopLoc;
      }
    };

    static SourceLocation getAttachedLoopLoc(const FunctionDecl* FD,
                                             SourceLocation pragmaLoc,
                                             SourceManager& SM) {
      Stmt* body = FD->getBody();
      AttachedLoopStmtFinder finder(pragmaLoc, SM);
      finder.TraverseStmt(body);
      return finder.getAttachedLoopLoc();
    }

    static void addCladLoopCheckpoints(ASTContext& C, DiffRequest& request) {
      SourceRange range = request->getSourceRange();
      assert(range.isValid());
      SourceLocation begin = range.getBegin();
      SourceLocation end = range.getEnd();
      clang::SourceManager& SM = C.getSourceManager();
      auto it = CladLoopCheckpoints.upper_bound(begin);
      auto e = CladLoopCheckpoints.end();

      for (; it != e && SM.isBeforeInTranslationUnit(*it, end); ++it)
        request.m_CladLoopCheckpoints.emplace(
            *it, getAttachedLoopLoc(request.Function, *it, SM));
    }

    static void diagnoseUnusedPragma(Sema& S, DiffRequest& request) {
      if (request.Mode != DiffMode::reverse &&
          request.Mode != DiffMode::pullback)
        return;

      static std::set<clang::SourceLocation> DiagnosedCladLoopCheckpoints;
      for (const auto& pair : request.m_CladLoopCheckpoints) {
        if (pair.second.isValid())
          continue;

        if (!DiagnosedCladLoopCheckpoints.insert(pair.first).second)
          continue;

        unsigned diagID = S.Diags.getCustomDiagID(
            DiagnosticsEngine::Error,
            "'#pragma clad checkpoint loop' is only allowed before a loop");
        S.Diag(pair.first, diagID);
      }
    }

    FunctionDecl* CladPlugin::ProcessDiffRequest(DiffRequest& request) {
      Sema& S = m_CI.getSema();
      if (!m_DerivativeBuilder) {
        m_DerivativeBuilder =
            std::make_unique<DerivativeBuilder>(S, *this, getScheduler());
        // Before the first chunk is made: a chunk keeps the name it was made
        // with, and that name is what the line table records.
        if (!m_DO.GeneratedSourceDir.empty())
          m_DerivativeBuilder->getGeneratedCode().setFileBase(
              m_DO.GeneratedSourceDir, m_CI.getCodeGenOpts().MainFileName);
      }

      if (request.Global) {
        auto deriveResult = m_DerivativeBuilder->Derive(request);
        auto* VDDiff = cast_or_null<VarDecl>(deriveResult.derivative);
        ProcessTopLevelDecl(VDDiff);
        // Dump the declaration if requested.
        printDerivative(VDDiff, request.DeclarationOnly, m_DO);
        return nullptr;
      }

      if (request.Function->getDefinition())
        request.Function = request.Function->getDefinition();
      // FIXME: These requests are not fully generated in the diffplanner and we
      // have to update diff params on this stage.
      if (request.CurrentDerivativeOrder > 1 ||
          getScheduler().getDerivedFns().IsCladDerivative(request.Function))
        request.UpdateDiffParamsInfo(m_CI.getSema());
      const FunctionDecl* FD = request.Function;
      ASTContext& C = S.getASTContext();
      clang::PrintingPolicy Policy = C.getPrintingPolicy();
#if CLANG_VERSION_MAJOR > 10
      // Our testsuite expects 'a<b<c> >' rather than 'a<b<c>>'.
      Policy.SplitTemplateClosers = true;
#endif
      // if enabled, print source code of the original functions
      if (m_DO.DumpSourceFn) {
        FD->print(llvm::outs(), Policy);
      }
      // if enabled, print ASTs of the original functions
      if (m_DO.DumpSourceFnAST)
        FD->dumpColor();

      // If enabled, set the proper fields in derivative builder.
      if (m_DO.PrintNumDiffErrorInfo) {
        m_DerivativeBuilder->setNumDiffErrDiag(true);
      }

      // Propagate relevant pragmas to diffrequests
      addCladLoopCheckpoints(C, request);

      FunctionDecl* DerivativeDecl = nullptr;
      bool alreadyDerived = false;
      FunctionDecl* OverloadedDerivativeDecl = nullptr;
      {
        llvm::SaveAndRestore<unsigned> Saved(request.RequestedDerivativeOrder,
                                             1);
        auto DFI = getScheduler().getDerivedFns().Find(request);
        if (DFI.IsValid()) {
          DerivativeDecl = DFI.DerivedFn();
          OverloadedDerivativeDecl = DFI.OverloadedDerivedFn();
          alreadyDerived = true;
        } else {
          auto deriveResult = m_DerivativeBuilder->Derive(request);
          DerivativeDecl = cast_or_null<FunctionDecl>(deriveResult.derivative);
          OverloadedDerivativeDecl = deriveResult.overload;
          // FIXME: Doing this with other function types might lead to
          // accidental numerical diff.
          if (isa<CXXConstructorDecl>(FD) &&
              (request.Mode == DiffMode::pullback) &&
              utils::hasEmptyBody(DerivativeDecl))
            return nullptr;
          if (DerivativeDecl)
            getScheduler().getDerivedFns().Add(DerivedFnInfo(
                request, DerivativeDecl, OverloadedDerivativeDecl));
        }
      }

      // Propagate relevant pragmas to diffrequests
      diagnoseUnusedPragma(S, request);

      if (OverloadedDerivativeDecl) {
        S.MarkFunctionReferenced(SourceLocation(), OverloadedDerivativeDecl);
        DelayedCallInfo DCI{CallKind::HandleTopLevelDecl,
                            OverloadedDerivativeDecl};
        if (!llvm::is_contained(m_DelayedCalls, DCI))
          ProcessTopLevelDecl(OverloadedDerivativeDecl);
      }
      if (DerivativeDecl) {
        if (!alreadyDerived &&
            (!request.CustomDerivative || request.CallUpdateRequired)) {
          // Reported on at the end of the unit rather than here: reading a
          // buffer settles where its lines are, so everything has to be
          // printed into it first.
          Generated G;
          G.Derivative = DerivativeDecl;
          G.AnalysisRan = request.EnableTBRAnalysis;
          // Where a user asked for this derivative, when one did. Clad asks
          // for some itself -- the second derivative a hessian needs -- and
          // those were written on no line at all.
          if (request.CallContext && request.Function) {
            G.Original = request.Function;
            G.RequestedAt = request.CallContext->getBeginLoc();
          }
          m_Generated.push_back(G);
          printDerivative(DerivativeDecl, request.DeclarationOnly, m_DO);

          S.MarkFunctionReferenced(SourceLocation(), DerivativeDecl);
          // We ideally should not call `HandleTopLevelDecl` for declarations
          // inside a namespace. After parsing a namespace that is defined
          // directly in translation unit context , clang calls
          // `BackendConsumer::HandleTopLevelDecl`.
          // `BackendConsumer::HandleTopLevelDecl` emits LLVM IR of each
          // declaration inside the namespace using CodeGen. We need to manually
          // call `HandleTopLevelDecl` for each new declaration added to a
          // namespace because `HandleTopLevelDecl` has already been called for
          // a namespace by Clang when the namespace is parsed.

          // Call CodeGen only if the produced Decl is a top-most
          // decl or is contained in a namespace decl.
          // FIXME: We could get rid of this by prepending the produced
          // derivatives in CladPlugin::HandleTranslationUnitDecl
          DeclContext* derivativeDC = DerivativeDecl->getLexicalDeclContext();
          DelayedCallInfo DCI{CallKind::HandleTopLevelDecl, DerivativeDecl};
          bool isTUorND =
              derivativeDC->isTranslationUnit() || derivativeDC->isNamespace();
          if (isTUorND && !llvm::is_contained(m_DelayedCalls, DCI))
            ProcessTopLevelDecl(DerivativeDecl);
        }
        bool lastDerivativeOrder = (request.CurrentDerivativeOrder ==
                                    request.RequestedDerivativeOrder);
        // If this is the last required derivative order, replace the function
        // inside a call to clad::differentiate/gradient with its derivative.
        if (request.CallUpdateRequired && lastDerivativeOrder)
          request.updateCall(DerivativeDecl, OverloadedDerivativeDecl,
                             m_CI.getSema());

        if (request.DeclarationOnly)
          request.DerivedFDPrototypes.push_back(DerivativeDecl);

        // Last requested order was computed, return the result.
        if (lastDerivativeOrder)
          return DerivativeDecl;
        // If higher order derivatives are required, proceed to compute them
        // recursively.
        request.Function = DerivativeDecl;
        request.CurrentDerivativeOrder += 1;
        return ProcessDiffRequest(request);
      }
      return nullptr;
    }

    void CladPlugin::SendToMultiplexer() {
      if (!m_Multiplexer)
        return;
      for (unsigned i = m_MultiplexerProcessedDelayedCallsIdx;
           i < m_DelayedCalls.size(); ++i) {
        auto DelayedCall = m_DelayedCalls[i];
        DeclGroupRef& D = DelayedCall.m_DGR;
        switch (DelayedCall.m_Kind) {
        case CallKind::HandleCXXStaticMemberVarInstantiation:
          m_Multiplexer->HandleCXXStaticMemberVarInstantiation(
              cast<VarDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleTopLevelDecl:
          m_Multiplexer->HandleTopLevelDecl(D);
          break;
        case CallKind::HandleInlineFunctionDefinition:
          m_Multiplexer->HandleInlineFunctionDefinition(
              cast<FunctionDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleInterestingDecl:
          m_Multiplexer->HandleInterestingDecl(D);
          break;
        case CallKind::HandleTagDeclDefinition:
          m_Multiplexer->HandleTagDeclDefinition(
              cast<TagDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleTagDeclRequiredDefinition:
          m_Multiplexer->HandleTagDeclRequiredDefinition(
              cast<TagDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleCXXImplicitFunctionInstantiation:
          m_Multiplexer->HandleCXXImplicitFunctionInstantiation(
              cast<FunctionDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleTopLevelDeclInObjCContainer:
          m_Multiplexer->HandleTopLevelDeclInObjCContainer(D);
          break;
        case CallKind::HandleImplicitImportDecl:
          m_Multiplexer->HandleImplicitImportDecl(
              cast<ImportDecl>(D.getSingleDecl()));
          break;
        case CallKind::CompleteTentativeDefinition:
          m_Multiplexer->CompleteTentativeDefinition(
              cast<VarDecl>(D.getSingleDecl()));
          break;
        case CallKind::CompleteExternalDeclaration:
          m_Multiplexer->CompleteExternalDeclaration(
              cast<VarDecl>(D.getSingleDecl()));
          break;
        case CallKind::AssignInheritanceModel:
          m_Multiplexer->AssignInheritanceModel(
              cast<CXXRecordDecl>(D.getSingleDecl()));
          break;
        case CallKind::HandleVTable:
          m_Multiplexer->HandleVTable(cast<CXXRecordDecl>(D.getSingleDecl()));
          break;
        case CallKind::InitializeSema:
          m_Multiplexer->InitializeSema(m_CI.getSema());
          break;
        };
      }

      m_MultiplexerProcessedDelayedCallsIdx = m_DelayedCalls.size();
    }

    bool CladPlugin::CheckBuiltins() {
      // If we have included "clad/Differentiator/Differentiator.h" return.
      if (m_HasRuntime)
        return true;

      // The plugin has a lot of different ways to be compiled: in-tree,
      // out-of-tree and hybrid. When we pick up the wrong header files we
      // usually see a problem with C.Idents not being properly initialized.
      // This assert tries to catch such situations heuristically.
      assert(&m_CI.getASTContext().Idents ==
                 &m_CI.getPreprocessor().getIdentifierTable() &&
             "Miscompiled?");
      NamespaceDecl* CladNS =
          utils::LookupNSD(m_CI.getSema(), "clad", /*shouldExist=*/false);
      m_HasRuntime = (CladNS != nullptr);
      return m_HasRuntime;
    }

    void CladPlugin::SetRequestOptions(RequestOptions& opts) const {
      // The last switch that named the analysis decides; otherwise it runs at
      // the default Analyses.def gives it.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                     \
      opts.Enable##Id##Analysis =                                              \
          (m_DO.Id##Switch == AnalysisSwitch::Unset)                           \
              ? (Default)                                                      \
              : (m_DO.Id##Switch == AnalysisSwitch::On);
#include "clad/Differentiator/Analyses.def"
      opts.EmitPortingHints = m_DO.EmitPortingHints;
    }

    DerivativePrinter& CladPlugin::getDerivativePrinter() {
      assert(m_DerivativeBuilder &&
             "asked to print before anything was derived");
      if (!m_DerivativePrinter)
        m_DerivativePrinter = std::make_unique<DerivativePrinter>(
            m_CI.getSema(), m_DerivativeBuilder->getGeneratedCode());
      return *m_DerivativePrinter;
    }

    DiffScheduler& CladPlugin::getScheduler() {
      if (!m_Scheduler) {
        RequestOptions Opts{};
        SetRequestOptions(Opts);
        m_Scheduler = std::make_unique<DiffScheduler>(m_CI.getSema(), Opts,
                                                      CladEnabledRange);
      }
      return *m_Scheduler;
    }

    void CladPlugin::FinalizeTranslationUnit() {
      Sema& S = m_CI.getSema();
      // Restore the TUScope that became a 0 in Sema::ActOnEndOfTranslationUnit.
      if (!m_CI.getPreprocessor().isIncrementalProcessingEnabled())
        S.TUScope = m_StoredTUScope;
      constexpr bool Enabled = true;
      Sema::GlobalEagerInstantiationScope GlobalInstantiations(
          S, Enabled CLAD_COMPAT_CLANG21_AtEndOfTUParam);
      Sema::LocalEagerInstantiationScope LocalInstantiations(
          S CLAD_COMPAT_CLANG21_AtEndOfTUParam);

      if (!getScheduler().getGraph().isProcessingNode()) {
        // This check is to avoid recursive processing of the graph, as
        // HandleTopLevelDecl can be called recursively in non-standard
        // setup for code generation.
        DiffRequest request = getScheduler().getGraph().getNextToProcessNode();
        while (request.Function || request.Global) {
          getScheduler().getGraph().setCurrentProcessingNode(request);
          ProcessDiffRequest(request);
          getScheduler().getGraph().markCurrentNodeProcessed();
          request = getScheduler().getGraph().getNextToProcessNode();
        }
      }

      // Put the TUScope in a consistent state after clad is done.
      if (!m_CI.getPreprocessor().isIncrementalProcessingEnabled())
        S.TUScope = nullptr;

      // Force emission of the produced pending template instantiations.
      LocalInstantiations.perform();
      GlobalInstantiations.perform();
    }

    void CladPlugin::HandleTranslationUnit(ASTContext& C) {
      // In case of diagnostics, don't bother, just let the compiler finish.
      if (!m_CI.getDiagnostics().hasErrorOccurred()) {
        // Traverse all collected DeclGroupRef only once to create the static
        // graph. Planning can trigger implicit instantiations (e.g. clad::Tag
        // when parsing the differentiate-call arguments) whose consumer
        // notifications append to m_DelayedCalls mid-loop; deque::push_back
        // keeps element references valid but invalidates iterators, so index
        // instead of iterating (the appended groups are then planned too).
        // NOLINTNEXTLINE(modernize-loop-convert)
        for (size_t i = 0; i < m_DelayedCalls.size(); ++i) {
          const DelayedCallInfo DCI = m_DelayedCalls[i];
          for (Decl* D : DCI.m_DGR) {
            if (const auto* FD = dyn_cast<FunctionDecl>(D))
              if (FD->isConstexpr())
                continue;
            getScheduler().Plan(DCI.m_DGR);
            break;
          }
        }

        if (m_CI.getFrontendOpts().ShowStats) {
          // Print the graph of the diff requests.
          llvm::errs() << "\n*** INFORMATION ABOUT THE DIFF REQUESTS\n";
          getScheduler().getGraph().dump();
        }

        FinalizeTranslationUnit();
        // Before the multiplexer: this is the last point at which every
        // derivative exists and none has reached code generation.
        materializeGeneratedCode();
        SendToMultiplexer();
      }
      if (m_Multiplexer)
        m_Multiplexer->HandleTranslationUnit(C);
    }

    void CladPlugin::PrintStats() {
      llvm::errs() << "*** INFORMATION ABOUT THE DELAYED CALLS\n";
      for (const DelayedCallInfo& DCI : m_DelayedCalls) {
        llvm::errs() << "   ";
        switch (DCI.m_Kind) {
        case CallKind::HandleCXXStaticMemberVarInstantiation:
          llvm::errs() << "HandleCXXStaticMemberVarInstantiation";
          break;
        case CallKind::HandleTopLevelDecl:
          llvm::errs() << "HandleTopLevelDecl";
          break;
        case CallKind::HandleInlineFunctionDefinition:
          llvm::errs() << "HandleInlineFunctionDefinition";
          break;
        case CallKind::HandleInterestingDecl:
          llvm::errs() << "HandleInterestingDecl";
          break;
        case CallKind::HandleTagDeclDefinition:
          llvm::errs() << "HandleTagDeclDefinition";
          break;
        case CallKind::HandleTagDeclRequiredDefinition:
          llvm::errs() << "HandleTagDeclRequiredDefinition";
          break;
        case CallKind::HandleCXXImplicitFunctionInstantiation:
          llvm::errs() << "HandleCXXImplicitFunctionInstantiation";
          break;
        case CallKind::HandleTopLevelDeclInObjCContainer:
          llvm::errs() << "HandleTopLevelDeclInObjCContainer";
          break;
        case CallKind::HandleImplicitImportDecl:
          llvm::errs() << "HandleImplicitImportDecl";
          break;
        case CallKind::CompleteTentativeDefinition:
          llvm::errs() << "CompleteTentativeDefinition";
          break;
        case CallKind::CompleteExternalDeclaration:
          llvm::errs() << "CompleteExternalDeclaration";
          break;
        case CallKind::AssignInheritanceModel:
          llvm::errs() << "AssignInheritanceModel";
          break;
        case CallKind::HandleVTable:
          llvm::errs() << "HandleVTable";
          break;
        case CallKind::InitializeSema:
          llvm::errs() << "InitializeSema";
          break;
        };
        for (const clang::Decl* D : DCI.m_DGR) {
          llvm::errs() << " " << D;
          if (const auto* ND = dyn_cast<NamedDecl>(D))
            llvm::errs() << " " << ND->getNameAsString();
        }
        llvm::errs() << "\n";
      }

      if (m_Multiplexer)
        m_Multiplexer->PrintStats();
    }

  } // end namespace plugin

  // Routine to check clang version at runtime against the clang version for
  // which clad was built.
  bool checkClangVersion() {
    std::string runtimeVersion = clang::getClangFullCPPVersion();
    std::string builtVersion = CLANG_MAJOR_VERSION;
    if (runtimeVersion.find(builtVersion) == std::string::npos)
      return false;
    else
      return true;
  }
} // end namespace clad

// Attach the frontend plugin.

using namespace clad::plugin;
// register the PluginASTAction in the registry.
static clang::FrontendPluginRegistry::Add<Action<CladPlugin> >
X("clad", "Produces derivatives or arbitrary functions");

static PragmaHandlerRegistry::Add<CladPragmaHandler>
    Y("clad", "Clad pragma directives handler.");

// Attach the backend plugin.
#include "ClangBackendPlugin.h"

#define BACKEND_PLUGIN_NAME "CladBackendPlugin"
// FIXME: Add a proper versioning that's based on CLANG_VERSION_STRING and
// a similar approach for clad (see Version.cpp and VERSION).
#define BACKEND_PLUGIN_VERSION "FIXME"
extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, BACKEND_PLUGIN_NAME, BACKEND_PLUGIN_VERSION,
          clad::ClangBackendPluginPass::registerCallbacks};
}
