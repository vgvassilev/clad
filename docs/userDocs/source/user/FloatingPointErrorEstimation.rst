Floating Point Error Estimation using CHEF-FP
*********************************************

============
Introduction
============

Data-intensive sciences that work with increasing data volumes and often in
heterogeneous computing environments require floating point stability. Robust
floating-point error detection can help reduce data reprocessing costs and help
develop important new lossy compression algorithms.

How does Automatic Differentiation (AD) fit into this?
======================================================

AD helps evaluate the exact derivative of a function. AD applies the
differential calculus chain rule throughout the semantics of the original
program. In the context of FP error estimation, the implementation relies on:

- **Reverse-Mode AD** (as opposed to Forward-Mode AD), since it provides the
  derivative of the function with respect to all intermediate and input
  variables, and

- **Source Transformation** (as opposed to Operator Overloading), since it does
  most of the work at compile time. The Clad Framework that is used in this
  research also uses source transformation.

**Clad** is implemented as a plugin for the Clang compiler. It inspects the
internal compiler representation of the target function to generate its
derivative. Clad requires little or no code modification, supports a growing
subset of C++ constructs, statements and data types, it enables efficient
gradient computation for large and complex codebases, and is deeply
integrated with the compiler, allowing automatic generation of error
estimation code.

Where does the CHEF-FP logic reside?
====================================

**CHEF-FP** (the tool created for Floating Point Error Estimation) is a
flexible, scalable, and easy-to-use source-code transformation tool based on
Automatic Differentiation (AD) for analyzing approximation errors in HPC
applications. 

  For more details, please view `Fast and Automatic Floating Point Error Analysis with CHEF-FP`_.

The main logic for CHEF-FP resides in the following files:

- `include/clad/Differentiator/ErrorEstimator.h`_

  - uses ``ErrorEstimationHandler`` class to estimate errors in a target
    function.  It keeps track of error expressions, emits error statements, and
    replaces parameter values.

- `include/clad/Differentiator/EstimationModel.h`_

  - contains the information needed to calculate the estimate value of the
    error.


   Above files include a lot of useful documentation in the form of code
   comments. Please view the `Doxygen Documentation`_ and the `Clad Readme`_
   for more details.

How does the FPEE Logic work?
=============================

While parsing the code using Clad, if it encounters a floating point variable,
it needs to be tracked (to accumulate relevant errors against that variable).
Next, the Error Estimation Calculation Formula (Error Model) needs to be built
(using ``EstimationModel.h``).

``EstimationModel.h`` contains the information needed to calculate the estimate 
value of the error. It is highly customizable (e.g., you can plug in your 
own custom formula as well). The default formula multiplies the derivative 
(dfdx) with the value of the variable (delta_x), for which the error estimate 
is required, and the machine epsilon (Em).

``std::abs(dfdx * delta_x * Em)``

  For this formula to work, the value of the variable (delta_x) should be saved
  at the relevant time.

This model will return a formula that is represented using a Clang
expression.This Clang expression can, in turn, be written into the  derivative
code that is generated using Clad.

What else is CHEF-FP capable of?
================================

Sensitivity Analysis
--------------------

A lot of information can be extracted from the intermediate floating point
errors. This includes sensitivity, that is, how sensitive a particular variable
is to floating point errors. The lower the sensitivity, the smaller the
likelihood that the variable will have a large impact on the total floating
point error of the function.

  Note: a more complex model may or may not have a direct relationship with
  sensitivity.

This has important implications on the numerical stability of the algorithms.
It also builds the foundation for the set of type-optimization techniques
called Mixed Precision Tuning.

  **Mixed Precision Tuning** involves demoting certain types to lower
  sensitivity, and subsequently, a lower contribution to the function's final
  error.

How do I create my own Custom model?
====================================

Custom Models may be one of the main reasons that new users may be interested
in adapting the CHEF-FP code to their specific use cases. 

Top define a custom model using Clad:

1. Implement the ``clad::FPErrorEstimationModel`` class, a generic interface 
that provides the error expressions for clad to generate.

2. Override the ``AssignError()`` function. This function is called for all LHS 
of every assignment expression in the target function.

  The function ``AssignError()`` represents the mathematical formula of an
  error model in a form that Clang can understand and convert to code. It
  provides users with a reference to the variable of interest and its
  derivative. The user, in turn, must return an expression that will be used to
  accumulate the error.

  Note: Creating these functions requires knowledge of the Clang APIs.

Demo customization examples can be found here:

- `demos/ErrorEstimation`_

The `CustomModel`_ and `PrintModel`_ demos are useful for users who would like 
to write their own models.

Further Reading
===============

For more technical details, please view: 

- `Fast and Automatic Floating Point Error Analysis with CHEF-FP`_ - (published paper)

- `How to Estimate Floating Point Errors Using AD`_ - (tutorial)

- `CHEF-FP Examples Repo`_ - (includes benchmarks)

- `Estimating Floating-Point Errors Using Automatic Differentiation`_ - (presentation, slides and video)

- `Floating-Point Error Estimation Proposal`_ - (PDF, slightly outdated, useful for background information)


Appendix - Notable Classes
==========================

clad::ErrorEstimator::ErrorEstimationHandler
--------------------------------------------

The ``ErrorEstimationHandler`` class is used to estimate errors in a target
function. When you use Clad in Error Estimation mode, the
``ErrorEstimationHandler`` class is responsible for handling the derivative and
error information exchange between Clad and the Error Estimation module
(CHEF-FP). This class is responsible for a lot of the housekeeping tasks as
well.

clad::ErrorEstimator::EmitFinalErrorStmts
-----------------------------------------

This function adds the final error and the other parameter errors to the
forward block.


.. _include/clad/Differentiator/ErrorEstimator.h: https://github.com/vgvassilev/clad/blob/master/include/clad/Differentiator/ErrorEstimator.h

.. _include/clad/Differentiator/EstimationModel.h: https://github.com/vgvassilev/clad/blob/master/include/clad/Differentiator/EstimationModel.h

.. _demos/ErrorEstimation: https://github.com/vgvassilev/clad/tree/master/demos/ErrorEstimation

.. _Fast and Automatic Floating Point Error Analysis with CHEF-FP: https://arxiv.org/pdf/2304.06441.pdf

.. _CustomModel: https://github.com/vgvassilev/clad/blob/master/demos/ErrorEstimation/CustomModel/README.md

.. _PrintModel: https://github.com/vgvassilev/clad/blob/master/demos/ErrorEstimation/PrintModel/README.md

.. _How to Estimate Floating Point Errors Using AD: https://compiler-research.org/tutorials/fp_error_estimation_clad_tutorial/

.. _Estimating Floating-Point Errors Using Automatic Differentiation: https://compiler-research.org/presentations/#FPErrorEstADSIAMUQ2022

.. _Floating-Point Error Estimation Proposal: https://compiler-research.org/assets/docs/Garima_Singh_Proposal_2020.pdf

.. _CHEF-FP Examples Repo: https://github.com/grimmmyshini/chef-fp-examples

.. _Clad Readme: https://github.com/vgvassilev/clad#floating-point-error-estimation---cladestimate_error

.. _Doxygen Documentation: https://clad.readthedocs.io/en/latest/internalDocs/html/index.html