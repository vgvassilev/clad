//--------------------------------------------------------------------*- C++ -//
// clad - the C++ Clang-based Automatic Differentiator
//
// A constant folding tool, working on AST level
//
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//----------------------------------------------------------------------------//

#ifndef CLAD_CONSTANT_FOLDER_H
#define CLAD_CONSTANT_FOLDER_H

#include "clang/AST/StmtVisitor.h"

namespace clang {
  class ASTContext;
  class BinaryOperator;
  class Expr;
  class QualType;
}

namespace clad {
  class ConstantFolder: public clang::StmtVisitor<ConstantFolder> {
  public:
    clang::Expr* fold(clang::BinaryOperator* BinOp);
    static clang::Expr* synthesizeLiteral(clang::QualType, clang::ASTContext &C,
                                          unsigned val);
  };
} // end namespace clad
#endif // CLAD_CONSTANT_FOLDER_H
