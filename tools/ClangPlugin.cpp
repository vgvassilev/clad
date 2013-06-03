//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "autodiff/Differentiator/DerivativeBuilder.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

namespace {

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
    llvm::SmallVector<const FunctionDecl*, 4> m_FunctionsToDiff;
  public:
    llvm::SmallVector<const FunctionDecl*, 4>& getFunctionsToDiff() {
      return m_FunctionsToDiff;
    }

    void collectFunctionsToDiff(FunctionDecl* FD) {
      TraverseDecl(FD);
    }

    bool VisitCallExpr(CallExpr* E) {
      if (const FunctionDecl *FD = E->getDirectCallee()) {
        // Note here that best would be to annotate with, eg:
        //  __attribute__((annotate("This is our diff that must differentiate"))) {
        // However, GCC doesn't support the annotate attribute on a function 
        // definition and clang's version on MacOS chokes up (with clang's trunk
        // everything seems ok 03.06.2013)
        
        //        if (const AnnotateAttr* A = FD->getAttr<AnnotateAttr>())
        if (FD->getName() == "diff")
          // We know that is *our* diff function.
          if (const ImplicitCastExpr* ICE = dyn_cast<ImplicitCastExpr>(E->getArg(0)))
            if (const DeclRefExpr* DRE = dyn_cast<DeclRefExpr>(ICE->getSubExpr())) {
              assert(isa<FunctionDecl>(DRE->getDecl()) && "Must not happen.");
              m_FunctionsToDiff.push_back(cast<const FunctionDecl>(DRE->getDecl()));
            }
      }
      return true;     // return false to abort visiting.
    }
  };
} // end namespace

namespace autodiff {
namespace plugin {
  class AutoDiffPlugin : public ASTConsumer {
  private:
    DiffCollector m_Collector;
    DerivativeBuilder m_DerivativeBuilder;
  public:
    virtual bool HandleTopLevelDecl(DeclGroupRef DGR) {
      for (DeclGroupRef::iterator I = DGR.begin(), E = DGR.end(); I != E; ++I)
        if (FunctionDecl* FD = dyn_cast<FunctionDecl>(*I))
          m_Collector.collectFunctionsToDiff(FD);
      llvm::SmallVector<const FunctionDecl*, 4>& functionsToDerive 
        = m_Collector.getFunctionsToDiff();
      for (size_t i = 0, e = functionsToDerive.size(); i < e; ++i) {
        const FunctionDecl* Derivative 
          = m_DerivativeBuilder.Derive(functionsToDerive[i]);
        Derivative->dump();
      }
      return true;
    }
  };
  
  
  template<typename ConsumerType>
  class Action : public PluginASTAction {
    
  protected:
    ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                   llvm::StringRef InFile) {
      return new ConsumerType();
    }
    
    bool ParseArgs(const CompilerInstance &CI,
                   const std::vector<std::string>& args) {
      return true;
    }
  };
} // end namespace plugin
} // end namespace autodiff

using namespace autodiff::plugin;
// register the PluginASTAction in the registry.
static FrontendPluginRegistry::Add<Action<AutoDiffPlugin> >
X("print-f-g", "prints source code statements in which f or g is referenced from diff");

