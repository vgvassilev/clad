Clad Installation
==================

This page covers how to properly install Clad.

At the moment, LLVM/Clang 5.0.x - 13.0.0 are supported.

Conda Installation
--------------------

Clad is available using conda <https://anaconda.org/conda-forge/clad>:

.. code-block:: bash

  conda install -c conda-forge clad


If you have already added ``conda-forge`` as a channel, the ``-c conda-forge`` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions:

  .. code-block:: bash

    conda config --add channels conda-forge
    conda update --all


Building from source (example was tested on Ubuntu 18.04 LTS and Ubuntu 20.04 LTS)
-----------------------------------------------------------------------------------

  .. code-block:: bash
  
    #sudo apt install clang-9 libclang-9-dev llvm-9-tools llvm-9-dev
    sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
    sudo -H pip install lit
    git clone https://github.com/vgvassilev/clad.git clad
    mkdir build_dir inst; cd build_dir
    cmake ../clad -DClang_DIR=/usr/lib/llvm-9 -DLLVM_DIR=/usr/lib/llvm-9 -DCMAKE_INSTALL_PREFIX=../inst -DLLVM_EXTERNAL_LIT="``which lit``"
    make && make install
  
Building from source (example was tested on macOS Catalina 10.15.7)
--------------------------------------------------------------------

.. code-block:: bash

  brew install llvm@12
  brew install python
  python -m pip install lit
  git clone https://github.com/vgvassilev/clad.git clad
  mkdir build_dir inst; cd build_dir
  cmake ../clad -DLLVM_DIR=/usr/local/Cellar/llvm/12.0.0_1/lib/cmake/llvm -DClang_DIR=/usr/local/Cellar/llvm/12.0.0_1/lib/cmake/clang -DCMAKE_INSTALL_PREFIX=../inst -DLLVM_EXTERNAL_LIT="``which lit``"
  make && make install
  make check-clad
  
Building from source LLVM, Clang and clad (development environment)
--------------------------------------------------------------------

  .. code-block:: bash

    sudo -H pip install lit
    LAST_KNOWN_GOOD_LLVM=$(wget https://raw.githubusercontent.com/vgvassilev/clad/master/LastKnownGoodLLVMRevision.txt -O - -q --no-check-certificate)
    LAST_KNOWN_GOOD_CLANG=$(wget https://raw.githubusercontent.com/vgvassilev/clad/master/LastKnownGoodClangRevision.txt -O - -q --no-check-certificate)
    git clone https://github.com/llvm-mirror/llvm.git src
    cd src; git checkout $LAST_KNOWN_GOOD_LLVM
    cd tools
    git clone https://github.com/llvm-mirror/clang.git clang
    cd clang ; git checkout $LAST_KNOWN_GOOD_CLANG
    cd ../
    git clone https://github.com/vgvassilev/clad.git clad
    cd ../..
    mkdir obj inst
    cd obj
    cmake -DCMAKE_BUILD_TYPE=Debug -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_INSTALL_PREFIX=../inst -DLLVM_EXTERNAL_LIT="``which lit``" ../src/
    make && make install
  