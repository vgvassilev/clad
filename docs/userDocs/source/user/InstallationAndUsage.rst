Clad Installation
******************

This page covers both installation and usage details for Clad.

At the moment, LLVM/Clang 7.0.x - 17.0.x are supported.

Conda Installation
====================

Clad is available using conda <https://anaconda.org/conda-forge/clad>:

.. code-block:: bash

   conda install -c conda-forge clad


If you have already added ``conda-forge`` as a channel, the ``-c conda-forge`` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions:

.. code-block:: bash

   conda config --add channels conda-forge
   conda update --all


Building from source
======================

Building from source (example was tested on Ubuntu 20.04 LTS)
-----------------------------------------------------------------------------------

.. code-block:: bash

   #sudo apt install clang-11 libclang-11-dev llvm-11-tools llvm-11-dev
   sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" 
   sudo -H pip install lit
   git clone https://github.com/vgvassilev/clad.git
   cd clad
   mkdir build && cd build
   cmake ../ -DLLVM_DIR=/usr/lib/llvm-11 -DLLVM_EXTERNAL_LIT="``which lit``"
   make && sudo make install
   
Building from source (example was tested on macOS Catalina 10.15.7)
--------------------------------------------------------------------

.. code-block:: bash

   brew install llvm@12
   brew install python
   python -m pip install lit
   git clone https://github.com/vgvassilev/clad.git clad
   mkdir build; cd build
   cmake ../clad -DLLVM_DIR=/usr/local/Cellar/llvm/12.0.0_1/lib/cmake/llvm -DClang_DIR=/usr/local/Cellar/llvm/12.0.0_1/lib/cmake/clang -DLLVM_EXTERNAL_LIT="``which lit``"
   make && make install
   make check-clad

Run the Clad tests:

.. code-block:: bash

   make -j8 check-clad

How to use Clad
=================

With Jupyter Notebooks
------------------------

xeus-cpp <https://github.com/compiler-research/xeus-cpp> provides a Jupyter kernel for C++ with the help of the C++ interpreter clang-repl and the native implementation of the Jupyter protocol xeus. Within the xeus-cpp framework, Clad can enable automatic differentiation (AD) such that users can automatically generate C++ code for their computation of derivatives of their functions.

To set up your environment, use:

.. code-block:: bash

   mamba create -n xeus-clad -c conda-forge clad xeus-cpp clangdev=20 jupyterlab
   conda activate xeus-clad
   jupyter notebook

The above will launch Jupyter with 3 Clad attached kernels for C++ 11/14/17.

Try out a Clad tutorial interactively in your browser through binder, here <https://mybinder.org/v2/gh/vgvassilev/clad/master?labpath=%2Fdemos%2FJupyter%2FIntro.ipynb>. 

As a plugin for Clang
-----------------------

Since Clad is a Clang plugin, it must be properly attached when the Clang compiler is invoked. First, the plugin must be built to get libclad.so (or .dylib). Thus, to compile SourceFile.cpp with Clad enabled use:

.. code-block:: bash

   clang -cc1 -x c++ -std=c++11 -load /full/path/to/lib/clad.so -plugin clad SourceFile.cpp

To compile using Clang < 10 , for example with clang-9, use:

.. code-block:: bash

   clang-9 -I /full/path/to/include/  -x c++ -std=c++11 -fplugin=/full/path/to/lib/clad.so SourceFile.cpp -o sourcefile -lstdc++ -lm

To save the Clad generated derivative code to `Derivatives.cpp` add:

.. code-block:: bash

   -Xclang -plugin-arg-clad -Xclang -fgenerate-source-file

To print the Clad generated derivative add:

.. code-block:: bash

   -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn


Note: Clad does not work with the Apple releases of Clang
