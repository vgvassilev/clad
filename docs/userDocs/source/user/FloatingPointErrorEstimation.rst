Floating-point error estimation
*******************************

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

  - it also holds the default error model, and looks up a custom one when the
    program supplies it.


   Above files include a lot of useful documentation in the form of code
   comments. Please view the `Doxygen Documentation`_ and the `Clad Readme`_
   for more details.

How does the FPEE Logic work?
=============================

While parsing the code using Clad, if it encounters a floating point variable,
it needs to be tracked (to accumulate relevant errors against that variable).
Next, the Error Estimation Calculation Formula (Error Model) needs to be built.

The error model decides what each write's error is taken to be, and it is
replaceable. The built-in one is the first-order Taylor expansion of the
function about the computed values. Writing :math:`\Delta v_i` for the rounding
error the :math:`i`-th write commits,

.. math::

   f(v + \Delta v) - f(v) \approx \sum_{i=1}^{k} \pdv{f}{v_i} \, \Delta v_i
   = \sum_{i=1}^{k} \bar{v}_i \, \Delta v_i

Bounding each write's rounding error relatively, :math:`\Delta v_i = v_i
\varepsilon`, and taking each term's magnitude gives what Clad emits:

.. math::

   E = \sum_{i=1}^{k} \left| \bar{v}_i \, v_i \, \varepsilon \right|

The built-in model uses the ``float`` machine epsilon,
:math:`\varepsilon = 2^{-23} \approx 1.19 \times 10^{-7}`, for every variable
it estimates, whatever that variable's own type. A ``double`` computation is
therefore charged the rounding of a ``float`` one, so the built-in model errs
high by a wide margin and is a starting point rather than a final answer. A
program that needs a closer estimate supplies its own model.

For the formula to work, the value of the variable has to be saved at the
relevant time.

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

To define a custom model, declare one function in namespace ``clad``::

  namespace clad {
  double getErrorVal(double dx, double x, const char* name);
  }

Clad looks it up by name when it generates an ``estimate_error`` derivative. If
it finds one, it calls it in place of the built-in model at every write the
reverse sweep passes, and accumulates what it returns. ``dx`` is the adjoint of
the value written, ``x`` is the value itself, and ``name`` is the variable's
name, which a model can use to treat some variables differently or to report on
them.

The signature has to match exactly. A ``getErrorVal`` whose signature differs is
reported as an error naming the expected one, rather than silently ignored.

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