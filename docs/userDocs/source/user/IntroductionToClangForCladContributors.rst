Clang for contributors
**********************

Since there’s a lack of official documentation for the LLVM Clang, while the
existing docs are not meant to explain how to modify the Clang AST, we are 
writing this text to help the newcomers get into the technical aspects of Clang
useful for contributing to Clad and manipulating the Clang AST in general.

The big picture
=================
So, as you probably know, Clang is a compiler. More precisely, it is a compiler
front end that operates with the LLVM compiler backend. This means that the 
code you're trying to compile will first be processed by Clang, which parses 
the source code and performs initial processing before passing it to the LLVM 
backend for optimisation and code generation. In this picture, Clad is an 
extension for Clang to generate derivatives of C++ functions along the way. 
That is, **Clad synthesises the code for the derivatives** while Clang is 
processing the actual code you have written.

The AST
=========
.. _clang::Expr: https://clang.llvm.org/doxygen/classclang_1_1ValueStmt__inherit__graph.png
.. |clang::Expr| replace:: ``clang::Expr`` 
.. _clang::Decl: https://clang.llvm.org/doxygen/classclang_1_1Decl.html
.. |clang::Decl| replace:: ``clang::Decl`` 
.. _clang::DeclStmt: https://clang.llvm.org/doxygen/classclang_1_1DeclStmt.html
.. |clang::DeclStmt| replace:: ``clang::DeclStmt``
.. _clang::Stmt: https://clang.llvm.org/doxygen/classclang_1_1Stmt.html
.. |clang::Stmt| replace:: ``clang::Stmt``
.. _clang::ValueStmt: https://clang.llvm.org/doxygen/classclang_1_1ValueStmt.html
.. |clang::ValueStmt| replace:: ``clang::ValueStmt`` 
.. _clang::CompoundStmt: https://clang.llvm.org/doxygen/classclang_1_1CompoundStmt.html
.. |clang::CompoundStmt| replace:: ``clang::CompoundStmt``

Clang, parses the input code into an **Abstract Syntax Tree (AST)**. It is at 
this stage that Clad operates as well. The nodes in the Clang AST represent 
various constructs of the source code, including: **declarations, statements, 
expressions**, etc. As one can guess from the name, **declarations (** 
|clang::Decl|_ **)** represent places in the source code where something is 
declared (a variable (clang::VarDecl), a function, etc.); **expressions (** 
|clang::Expr|_ **)** are basically something that can be evaluated to a 
meaningful value, for instance binary operations (``clang::BinaryOperator``) 
like addition or comparison; **statements (** |clang::Stmt|_ **)** could be a 
lot of different things, like loops (``clang::WhileStmt``), conditions 
(``clang::IfStmt``), breaks, etc – pretty much anything you have in your code. 
An important note is that declarations (``clang::Decl``) and statements 
(``clang::Stmt``) are completely different classes (so, **a declaration is not 
a statement**), but there’s a separate declaration statement class, 
|clang::DeclStmt|_, which wraps a declaration so it can appear among
statements. An expression (``clang::Expr``), on the other
hand, is always a statement, since ``clang::Expr`` inherits from 
|clang::ValueStmt|_ which, in turn, inherits from ``clang::Stmt`` and is a 
statement.
Statements can be grouped into one with a compound statement 
(|clang::CompoundStmt|_).

Flow graph of Clad’s compilation
==================================
Clang hands an AST consumer a translation unit through a few entry points, and
Clad is such a consumer.
`This source <https://clang.llvm.org/docs/RAVFrontendAction.html>`__ describes
those entry points and how they are connected; the ones Clad uses are explained
in better detail in later sections.

Clad does its work in two passes over the unit, which the graph below follows.
It first *plans*: a walk of the parsed declarations finds every
``clad::differentiate``, ``clad::gradient``, ``clad::hessian``,
``clad::jacobian`` and ``clad::estimate_error`` call, records a request for
each, and walks into what those functions call to record the sub-requests they
will need. It then *derives*: it drains those requests, generating a derivative
for each and pointing the call at it. The two are separate because deriving one
function discovers more requests still, so the second pass adds to the same
graph it is draining.

.. mermaid::

   flowchart TD
     CAC["Action::CreateASTConsumer<br/>builds the CladPlugin consumer"]
     HTLD["HandleTopLevelDecl<br/>defers each declaration group"]
     HTU["HandleTranslationUnit<br/>the unit is parsed;<br/>both passes run here"]
     PLAN["DiffScheduler::Plan<br/>DiffCollector walks the groups"]
     VCE["DiffCollector::VisitCallExpr<br/>spots the clad:: calls"]
     GRAPH[("DynamicGraph of DiffRequest<br/>one node per derivative")]
     FIN["FinalizeTranslationUnit<br/>takes the next node"]
     PDR["ProcessDiffRequest<br/>one request at a time"]
     DERIVE["DerivativeBuilder::Derive<br/>picks the visitor for the mode"]
     VISIT["Visitor::Derive<br/>builds the derivative's AST"]
     UPDATE["DiffRequest::updateCall<br/>points the call at the derivative"]
     MAT["materializeGeneratedCode<br/>prints the generated source"]
     MUX["SendToMultiplexer<br/>replays the decls to CodeGen"]

     CAC --> HTLD --> HTU
     HTU -- "pass 1: plan" --> PLAN --> VCE
     VCE -- "records a request" --> GRAPH
     GRAPH -- "pass 2: derive" --> FIN
     FIN -- "for each node" --> PDR --> DERIVE --> VISIT --> UPDATE
     VISIT -. "a nested call" .-> GRAPH
     UPDATE -- "once the graph is drained" --> MAT --> MUX

Identifying functions to derive
=================================
``HandleTopLevelDecl()`` sets each declaration group aside for later. The one
exception is ``clad::differentiate<clad::immediate_mode>`` on a ``constexpr``
function, which is planned and derived as its group arrives. The walk that finds
every other request runs once the whole unit is parsed, from
``HandleTranslationUnit()``. It uses ``DiffCollector``, a
``RecursiveASTVisitor``, which reaches nodes through the corresponding
``Visit()`` functions -- a call expression, which corresponds to a function call
node, through ``VisitCallExpr()``, and so on. Clad distinguishes the functions
related to a differentiation request by annotating a compiler attribute that
matches the first letter of the differentiation method to be used (e.g. “G” for
gradient function). This way, when ``VisitCallExpr()`` is called upon those
nodes, the differentiation request is identified, initialised and added to the
request graph.
Similarly, any statement supported by Clad has a corresponding ``Visit()`` 
method. An if statement would be visited by ``VisitIfStmt()``, a return 
statement by ``VisitReturnStmt()``, etc. The general rule is that the ``Visit``
method is called ``Visit+the name of the statement in Clang``.

Code generation and insertion
===============================
Once every request has been identified, ``FinalizeTranslationUnit()`` takes them
off the graph one at a time and hands each to the translation unit’s
``DerivativeBuilder``, which is built on the first request and reused.
``DerivativeBuilder::Derive()`` picks the ``Visitor`` for the request’s mode
(forward, reverse, etc.) and invokes its ``Derive()`` method. That walks the
original body, which triggers a secondary processing of the nodes in the
original function. In reverse mode the walk is driven by
``ReverseModeVisitor::DifferentiateWithClad()``.
During this traversal, for each node encountered, the corresponding node in the
derived function is created and returned to the parent node. The parent node 
then emplaces this newly created node into the current code block of the 
derived function. This process repeats until the traversal reaches the top 
level of the original function, resulting in the complete computation of the 
derived function.
Once the derived function is fully computed, its declaration is created, 
including the appropriate body and parameters. A reference to this derived 
function is then added in ``updateCall()``, where the function to be returned 
by ``clad::differentiate()`` / ``clad::gradient()`` is replaced with its 
derivative.

Statement differentiation
===========================
Visit methods that we have just talked about earlier typically return a 
``clad::StmtDiff`` object. This is a Clad’s structure, so you can find its 
definition in the code and explore its methods. It is basically a type that 
contains 4 values, each one being a pointer to a ``clang::Stmt``. Of these, the
following two values are crucial to understand:

- ``clad::StmtDiff::getStmt()`` - this method gives you the statement contained
  in this StmtDiff object.

- ``clad::StmtDiff::getStmt_dx()`` - this method gives you the derivative of 
  the statement contained by this object.

Let’s see how ``clang::NullStmt`` is handled in the reverse mode (reverse mode
differentiation is called by calling ``clad::gradient``) as an example. Since 
the null statement (which you can have by writing ``;;`` at the end of a line 
in the source code) represents an empty statement in Clang, it doesn’t really 
affect the derivative in any way. So if this happens to be in the source code, 
we just want to ignore it. In the reverse mode, Clad generates derivative 
functions that consist of two parts: a forward pass and a reverse pass. Clad 
produces a ``clad::StmtDiff`` object for each statement of the original 
function where the object’s first part is what will be put in the forward pass
part of the produced derived function and should basically do the same thing
as the original statement, whereas the object’s second part is basically the 
derivative, which is put into the reverse pass. So, in the case of the null 
statement, both of these should be nothing! Which is expressed by returning an 
empty ``clad::StmtDiff`` from the visitor.

.. code-block:: c++

    StmtDiff VisitNullStmt(const clang::NullStmt* NS) { 
        return StmtDiff{}; 
    };

Less relevant notes
=====================
This section is not that important, but it’s just a bunch of facts that might 
confuse a newcomer.

- Clang AST might not seem like a tree (:D)! In a mathematical sense, a tree is
  a graph without cycles. While the AST itself is intended to be a tree and 
  thus acyclic, practical considerations and additional structures can 
  introduce references that might resemble cycles. So, some recursive 
  constructs or templates  in the source code might confuse you.

- Clang can compile many C-like languages, specifically: C, C++, Objective-C, 
  and Objective-C++. So, if you look at the documentation, not only are you 
  going to see some general Clang classes and classes designed specifically for
  C++ compilation (those mostly start with a CXX- prefix), but also things that
  have an ObjC- prefix like ``clang::ObjCAtTryStmt`` that represent statements 
  of the Objective-C programming language that you likely don’t need.

Further Reading
===============

- `“Changing Everything With Clang Plugins” — 2020 LLVM Developers’ Meeting: H. Finkel <https://www.youtube.com/watch?v=A9COzFs-gEg>`__

- `Introduction to the Clang AST <https://clang.llvm.org/docs/IntroductionToTheClangAST.html>`__ --
  the shortest path to reading what ``-ast-dump`` prints.

- `How to write RecursiveASTVisitor based ASTFrontendActions <https://clang.llvm.org/docs/RAVFrontendAction.html>`__ --
  a walkthrough of the traversal Clad's visitors are built on.

- `Clang Internals Manual <https://clang.llvm.org/docs/InternalsManual.html>`__ --
  the reference for the AST, Sema and the diagnostics subsystem.

- `Clang Plugins <https://clang.llvm.org/docs/ClangPlugins.html>`__ --
  how a plugin like Clad is attached to the compiler.

- `LLVM Programmer's Manual <https://llvm.org/docs/ProgrammersManual.html>`__ --
  the data structures and idioms LLVM code, Clad included, is written in.

For the mathematics rather than the compiler, see
:ref:`Further Reading <ad-further-reading>` in Core Concepts.
