//--------------------------------------------------------------------*- C++ -*-
// AutoDiff - the C++ Clang-based Automatic Differentiator
// version: $Id$
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "autodiff/Differentiator/Differentiator.h"

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace {
  class FunctionDeclarationFinder : public ASTConsumer, 
                                    public clang::RecursiveASTVisitor<FunctionDeclarationFinder>  {
    
  public:	  
    virtual void HandleTranslationUnit(ASTContext &ctx) {
      // calls TraverseCallExpr()
      TraverseDecl(ctx.getTranslationUnitDecl());
    }
    
    // returns true to continue parsing, or false to abort parsing.
    bool TraverseCallExpr(CallExpr *E) {
      if (FunctionDecl *FD = E->getDirectCallee()) {
        if (FD->getName() == "diff") {
          std::string argName = getStringHelper(E->getArg(0));
          if (argName == "g" || argName == "f") {
            //print the expression of interest
            llvm::errs() << getStringHelper(E) << ";\n";  
          }
        }
      }
      
      return true;	  
    }
    
  private:		  
    // convert expr to string
    std::string getStringHelper(Expr *E) {
      clang::LangOptions LangOpts;
      LangOpts.CPlusPlus = true;
      clang::PrintingPolicy Policy(LangOpts);
      
      std::string TypeS;
      llvm::raw_string_ostream s(TypeS);
      E->printPretty(s, 0, Policy);
      return s.str();
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
  
  
} // end namespace

// register the PluginASTAction in the registry.
static FrontendPluginRegistry::Add<Action<FunctionDeclarationFinder> >
X("print-f-g", "prints source code statements in which f or g is referenced from diff");

