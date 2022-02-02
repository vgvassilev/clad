//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo describing the usage of custom estimation models with the built -in
// error estimator of clad.
//
// author:  Garima Singh
//----------------------------------------------------------------------------//
// For information on how to run this demo, please take a look at the README.

#include "clad/Differentiator/EstimationModel.h"

/// This is our dummy estimation model class.
/// We will be using this to override the virtual function in the
/// FPErrorEstimationModel class.
class CustomModel : public clad::FPErrorEstimationModel {
public:
  CustomModel(clad::DerivativeBuilder& builder)
      : FPErrorEstimationModel(builder) {}
  // Return an expression of the following kind:
  //  dfdx * delta_x
  clang::Expr* AssignError(clad::StmtDiff refExpr) override;
};
