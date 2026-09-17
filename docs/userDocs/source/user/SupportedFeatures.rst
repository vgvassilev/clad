Supported C++ Features
***********************

The short answer is that Clad differentiates ordinary numerical C++. Arithmetic
and comparisons, calls to your own functions and to ``<cmath>``, ``if`` and
``switch``, every loop form, arrays and pointers, references, structs and
classes with their constructors and member functions, lambdas, function
templates and overloaded operators are all exercised by the test suite.
:doc:`Using Clad <UsingClad>` works through the ones you drive directly.

One caveat on that list: a class holding a reference *member* is a different
matter from a reference variable, and currently gives a wrong gradient. See
`issue #2082 <https://github.com/vgvassilev/clad/issues/2082>`__.

The rest of this page covers the edges: what Clad refuses and what it warns
about.

If Clad cannot differentiate something
=======================================

You will usually be told. Clad reports the construct by name:

.. code-block:: text

  warning: statement kind 'CXXThrowExpr' is not supported

The name in that warning is the construct Clad has no rule for, and it is the
term to search the `issue tracker <https://github.com/vgvassilev/clad/issues>`__
with: gaps are tracked there, where the state is current. If nothing matches,
please open an issue.

Four things can happen:

It works
  The examples in this documentation are compiled and run by Clad's test suite,
  so what they claim is what they do.

Clad tells you, and the build finishes
  You get the warning above. The derivative is wrong or missing for that part of
  the function, but nothing is hidden from you.

Clad tells you, and then crashes
  For a few constructs the warning is followed by a compiler crash. That is a
  bug rather than a deliberate limit, and is tracked as one.

Clad is silently wrong
  No diagnostic, and an incorrect derivative. This is a bug; please report it.

Support is generally consistent between modes: what Clad handles, it handles in
both. A few constructs are the exception, handled in one mode and not the other,
so if one mode refuses your function the other is worth trying.

Templates
==========

Clad differentiates an instantiation, so constructs that exist only in an
uninstantiated template -- dependent names, unresolved overload sets, fold
expressions, parameter packs, concept checks -- never reach it. Name the
instantiation you want, as described in
:ref:`Differentiating Templates and Overloaded Functions
<differentiating-templates-and-overloads>`.
