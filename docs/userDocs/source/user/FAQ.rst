FAQ 
*****

The questions below come up repeatedly in Clad's
`issues <https://github.com/vgvassilev/clad/issues>`__ and
`discussions <https://github.com/vgvassilev/clad/discussions>`__.

Clang says "clad doesn't appear to be loaded"
==============================================

The full message is:

.. code-block:: text

   static assertion failed: clad doesn't appear to be loaded; make sure that
   you pass clad.so to clang.

``clad/Differentiator/Differentiator.h`` is an ordinary header and compiles on
its own, but the derivative is produced by the plugin, not by the header. If
the compile command does not load the plugin, there is nothing to produce it,
and Clad says so instead of letting you link a program whose derivatives are
all missing.

Add the plugin to the command line:

.. code-block:: bash

   clang++ -std=c++17 -I /full/path/to/include/ \
           -fplugin=/full/path/to/lib/clad.so SourceFile.cpp

:doc:`Clad Installation <InstallationAndUsage>` has the details, including the
spelling older Clang releases need.

Clang cannot load clad.so, or reports an undefined symbol
==========================================================

Clad is a Clang plugin, so it is built against one particular Clang and has to
be loaded into that same one. Loading it into a different build gives an
undefined symbol at load time, and loading it into a different version makes
Clad refuse to run: it compares the version of the Clang loading it against the
version it was built against.

Point the compile command at the ``clang++`` Clad was built against. Note that
the Apple releases of Clang do not load Clad at all -- install an upstream LLVM.

Should I use forward mode or reverse mode?
===========================================

Count the inputs you need derivatives for. One run of forward mode
(``clad::differentiate``) gives you the derivative with respect to one input;
one run of reverse mode (``clad::gradient``) gives you the derivatives with
respect to every input at once. Both cost a small multiple of the original
function.

So a function with many inputs and one output -- the usual case, a scalar cost
or likelihood -- wants reverse mode. A function with one input and many outputs
wants forward mode. :doc:`Core Concepts <CoreConcepts>` explains why.

How do I differentiate with respect to an array?
=================================================

If the parameter has a known size, pass an array of the same size for its
derivative:

.. literalinclude:: ../../../../test/Documentation/FAQ/FixedSizeArray.cpp
   :language: cpp
   :start-after: docs-begin-fixed-size-array
   :end-before: docs-end-fixed-size-array

If it is a pointer whose length is only known at run time, wrap the derivative
in a ``clad::array_ref`` so Clad knows how far it may write:

.. literalinclude:: ../../../../test/Documentation/FAQ/PointerArray.cpp
   :language: cpp
   :start-after: docs-begin-pointer-array
   :end-before: docs-end-pointer-array

Clad warns that a function has no definition
=============================================

The warning reads:

.. code-block:: text

   attempted differentiation of function 'f' without definition and no
   suitable overload was found in namespace 'custom_derivatives'

Clad differentiates source code, so it needs the body of every function it
reaches. A function declared in a header but defined in another translation
unit, or in a library you only link against, has no body to differentiate.

There are three ways forward. Make the definition visible, by moving it into a
header or into the same translation unit. Write a custom derivative that tells
Clad what the derivative of that function is -- see
:doc:`Custom derivatives <CustomDerivatives>`; the ``-fclad-porting-hints``
flag makes Clad name the signature it is looking for. Or leave it, and let
Clad fall back to numerical differentiation, which is what it does by default;
compiling with ``-DCLAD_NO_NUM_DIFF`` turns the fallback off and makes the
missing definition an error.

How do I see the code Clad generated?
======================================

``CladFunction::dump()`` prints it, and ``CladFunction::getCode()`` returns it
as a string:

.. code-block:: cpp

   auto df = clad::differentiate(f, "x");
   df.dump();

That shows the one derivative you asked for. To see everything Clad generated
for a translation unit, including the derivatives it produced for nested calls,
or to get a file you can compile and step through in a debugger, see
:ref:`Inspecting the generated code <inspecting-the-generated-code>`.

Can Clad differentiate templates and overloaded functions?
===========================================================

Yes, but you have to say which function you mean, because the name alone stands
for several. See
:ref:`Differentiating Templates and Overloaded Functions <differentiating-templates-and-overloads>`.

I think a derivative is wrong. What should I report?
=====================================================

Please open an `issue <https://github.com/vgvassilev/clad/issues>`__ with:

* the smallest function that still shows the problem, in one file;
* what Clad generated for it, from ``dump()``;
* the value you expected and the value you got;
* the Clang version and the Clad version or commit.

A derivative that disagrees with a finite difference of the original function
is good evidence, and the generated code usually shows where the derivative
went wrong, so those two together are the most useful thing to send.
