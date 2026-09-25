Demos
*****

The examples elsewhere in this guide are short and meant to be read. The
programs under ``demos/`` are meant to be run. Each is a whole program you can
build, execute and change, and most print something worth looking at.

Each entry below says what the demo is for, then shows the few lines where it
actually calls clad. Those lines are pulled from the demo itself, so they are
the real code and not a copy that can go out of date. The name links to the
whole program in the repository, which is where to go once the excerpt has
told you whether it is the one you want.

Building one
============

Nothing in the build system builds the demos. A demo is compiled the way any
program using clad is, by loading the plugin:

.. code-block:: bash

   clang++ -std=c++17 -I /path/to/clad/include \
           -fplugin=/path/to/clad.so demos/GradientDescent.cpp -o demo

:doc:`Installation and usage <InstallationAndUsage>` explains the flags and
the :doc:`options reference <Options>` lists what clad accepts. Demos needing
more than that say so.

Learning from a derivative
==========================

:demo:`GradientDescent.cpp`
   Fits a straight line to data. To know which way to nudge the line, you need
   the slope of the error with respect to each parameter, and clad supplies it.
   This is the shortest answer to why reverse mode exists: many parameters, one
   number to make smaller.

   .. literalinclude:: ../../../../demos/GradientDescent.cpp
      :language: cpp
      :start-after: docs-begin-fit
      :end-before: docs-end-fit

   .. literalinclude:: ../../../../demos/GradientDescent.cpp
      :language: cpp
      :start-after: docs-begin-fit-call
      :end-before: docs-end-fit-call

:demo:`cladtorch/`
   The same idea, grown up: a GPT-2 that trains and writes text. One call asks
   for the gradient of the loss, and the training loop does the rest. Needs
   libtorch; ``download_training_data.sh`` fetches the text it learns from.

   .. literalinclude:: ../../../../demos/cladtorch/train_llm.cpp
      :language: cpp
      :start-after: docs-begin-llm-loss
      :end-before: docs-end-llm-loss

   .. literalinclude:: ../../../../demos/cladtorch/train_llm.cpp
      :language: cpp
      :start-after: docs-begin-llm
      :end-before: docs-end-llm

Differentiating a whole program
===============================

:demo:`ODESolverSensitivity.cpp`
   Asks how much the answer of a differential equation would move if you
   changed the numbers you started with. Clad differentiates the solver itself,
   loops and all, so changing the equation or swapping the integrator gives new
   derivatives without deriving anything by hand.

   .. literalinclude:: ../../../../demos/ODESolverSensitivity.cpp
      :language: cpp
      :start-after: docs-begin-sensitivity
      :end-before: docs-end-sensitivity

   .. literalinclude:: ../../../../demos/ODESolverSensitivity.cpp
      :language: cpp
      :start-after: docs-begin-sensitivity-call
      :end-before: docs-end-sensitivity-call

:demo:`ComputerGraphics/smallpt/`
   A path tracer. Its shapes are described by a function that says how far away
   a surface is, and the direction a surface faces is the derivative of that
   function, so clad works out the shading rather than a formula written by hand
   for every shape.

   .. literalinclude:: ../../../../demos/ComputerGraphics/smallpt/SmallPT.cpp
      :language: cpp
      :start-after: docs-begin-smallpt
      :end-before: docs-end-smallpt

   .. literalinclude:: ../../../../demos/ComputerGraphics/smallpt/SmallPT.cpp
      :language: cpp
      :start-after: docs-begin-smallpt-call
      :end-before: docs-end-smallpt-call

:demo:`Gradient.cpp`
   The same trick in miniature and easier to read first: the direction a sphere
   faces at a point, from three derivatives of the sphere's equation.

   .. literalinclude:: ../../../../demos/Gradient.cpp
      :language: cpp
      :start-after: docs-begin-normal
      :end-before: docs-end-normal

Trusting the floating point
===========================

:demo:`ErrorEstimation/FloatSum.cpp`
   Adds the same numbers twice, once plainly and once with a trick that
   compensates for rounding, and has clad estimate how much error each one
   built up. The gnuplot lines in the file plot the two against each other.

   .. literalinclude:: ../../../../demos/ErrorEstimation/FloatSum.cpp
      :language: cpp
      :start-after: docs-begin-esterror
      :end-before: docs-end-esterror

   .. literalinclude:: ../../../../demos/ErrorEstimation/FloatSum.cpp
      :language: cpp
      :start-after: docs-begin-esterror-call
      :end-before: docs-end-esterror-call

:demo:`ErrorEstimation/CustomModel/`
   Clad's built-in guess at the error is deliberately pessimistic: it reports
   the worst case. If you know more about your numbers you can say so by
   supplying your own model, which is what this does. It has a README.

   .. literalinclude:: ../../../../demos/ErrorEstimation/CustomModel/test.cpp
      :language: cpp
      :start-after: docs-begin-custommodel
      :end-before: docs-end-custommodel

:demo:`ErrorEstimation/PrintModel/`
   A model that estimates nothing and simply reports every place clad would
   have accounted for error, which is the easiest way to see what the machinery
   is doing. It has a README.

   .. literalinclude:: ../../../../demos/ErrorEstimation/PrintModel/test.h
      :language: cpp
      :start-after: docs-begin-printmodel
      :end-before: docs-end-printmodel

Running on a GPU
================

:demo:`CUDA/`
   Six programs: ``VectorAddition``, ``ParticleSimulation``,
   ``TensorContraction``, ``LinearRegression``, ``BoWLogisticRegression``, and a
   port of NVIDIA's ``BlackScholes`` that keeps the original CPU version and
   checks the gradients against it. Most differentiate host code that drives
   the GPU through Thrust; ``TensorContraction`` differentiates a function that
   launches a kernel, which is what the excerpt shows. Needs a CUDA toolkit.

   .. literalinclude:: ../../../../demos/CUDA/TensorContraction.cu
      :language: cpp
      :start-after: docs-begin-cuda
      :end-before: docs-end-cuda

Reaching past what clad can differentiate
=========================================

:demo:`CustomTypeNumDiff.cpp`
   Some types clad cannot take apart, like a number stored as a scaled integer.
   It falls back to measuring the derivative instead of deriving it, by
   evaluating the function at nearby points.

   .. literalinclude:: ../../../../demos/CustomTypeNumDiff.cpp
      :language: cpp
      :start-after: docs-begin-numdiff
      :end-before: docs-end-numdiff

:demo:`Templates.cpp`
   A template whose ``long double`` version computes a different formula from
   the general one. Clad differentiates whichever version the compiler actually
   picked, not the one you wrote first.

   .. literalinclude:: ../../../../demos/Templates.cpp
      :language: cpp
      :start-after: docs-begin-specialisation
      :end-before: docs-end-specialisation

Doing it in one pass
====================

:demo:`VectorForwardMode.cpp`
   Forward mode normally costs one pass per input you ask about. Vector mode
   does them together, which matters here because the inputs are two arrays
   whose length is only known while the program runs.

   .. literalinclude:: ../../../../demos/VectorForwardMode.cpp
      :language: cpp
      :start-after: docs-begin-vectormode
      :end-before: docs-end-vectormode

:demo:`RosenbrockFunction.cpp`
   The Rosenbrock function, a long narrow valley that optimisers are measured
   against, with its derivatives taken in forward mode.

   .. literalinclude:: ../../../../demos/RosenbrockFunction.cpp
      :language: cpp
      :start-after: docs-begin-rosenbrock
      :end-before: docs-end-rosenbrock

Without installing anything
===========================

Clad is on `Compiler Explorer <https://godbolt.org/z/3KWhY4j8M>`_, which is the
quickest way to see what it does to a function: write one, ask for its
derivative, and read the code clad generated beside it. The build there comes
from clad's own master, so what you try is what is current.

:demo:`Jupyter/Intro.ipynb`
   Clad in a notebook, a cell at a time. `Binder
   <https://mybinder.org/v2/gh/vgvassilev/clad/master?labpath=%2Fdemos%2FJupyter%2FIntro.ipynb>`_
   runs this one in a browser; locally it needs a C++ Jupyter kernel, and the
   notebook uses xeus-cpp.
