#!/bin/bash

build_directory=build
install_directory=install
llvm_version=18
build_type=Release
displayed_help=0

# Determine information about the system
determine_system_info () {
    export os=$(uname -s)
    if [[ "${os}" == "Darwin" ]]; then
      export os_version=$(sw_vers -productVersion)
      export ncpus=$(sysctl -n hw.ncpu)
    else
      export ncpus=$(nproc --all)
    fi
}

# Install getopt so bash script options can be processed
install_getopt () {
  if [[ "$os" == "Darwin" ]]; then
      brew install util-linux
      export PATH="$(brew --prefix util-linux)/bin:$PATH"
      export PATH="$(brew --prefix util-linux)/sbin:$PATH"
  else
      sudo apt install util-linux
  fi
}

# Display help information of various options of this script
display_help () {
    echo "This build script comes with many options"
    echo "These options are"
    echo "-h/--help Display this help information"
    echo "-b/--build_dir Set directory to build clad into (Default build)"
    echo "-i/--install_dir Set directory to install clad into (Default install)"
    echo "-l/--llvm Set which to build clad against (Default 18)"
    echo "-t/--build_type Set build type for clad (Default Release)"
}

# Install llvm which Clad will be built against
install_compiler () {
  if [[ "$os" == "Darwin" ]]; then
    echo 'Installing llvm '$llvm_version' using brew to build Clad against'
    brew update
    brew upgrade  
    brew install llvm@${llvm_version}
    path_to_llvm_build=$(brew --prefix llvm@${llvm_version})
    export CC=$path_to_llvm_build/bin/clang
    export CXX=$path_to_llvm_build/bin/clang++
  else
    echo 'Install llvm '$llvm_version' using apt to build Clad against'
    os_codename="`cat /etc/os-release | grep UBUNTU_CODENAME | cut -d = -f 2`"
    sudo apt update
    curl https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
    echo "deb https://apt.llvm.org/${os_codename}/ llvm-toolchain-${os_codename}-${llvm_version} main" | sudo tee -a /etc/apt/sources.list
    sudo apt update
    sudo apt install -y clang-${llvm_version} libclang-${llvm_version}-dev libc++-${llvm_version}-dev
    path_to_llvm_build=/usr/lib/llvm-${llvm_version}/
    
    # llvm-8-tools is broken as it depends on python2.7 we use FileCheck from llvm-9-tools
    if [[ "$llvm_version" == "8" ]]; then
        sudo apt install llvm-9-dev llvm-9-tools
        sudo ln -s /usr/lib/llvm-9/bin/FileCheck $path_to_llvm_build/bin/FileCheck
    fi
    
    export CC=clang-${llvm_version}
    export CXX=clang++-${llvm_version}
  fi
}

# Install dependencies needed to build and test Clad
install_other_dependencies () {
  if [[ "$os" == "Darwin" ]]; then
    brew install python cmake
    # workaround for https://github.com/actions/setup-python/issues/577
    for pkg in $(brew list | grep '^python@'); do
      brew unlink "$pkg"
      brew link --overwrite "$pkg"
    done
    brew install lit
    export lit_directory="$(brew --prefix)/bin/lit"
  else
    sudo apt install python3 cmake python3-pip 
    pip3 install lit
    export lit_directory="$(python3 -m site --user-base)/bin/lit"
  fi
}

# Builds Clad
build_clad () {
  mkdir $build_directory
  cd $build_directory
  echo $path_to_llvm_build/lib/cmake/llvm
  echo $path_to_llvm_build/lib/cmake/clang
  cmake -DCMAKE_INSTALL_PREFIX=$install_directory \
  	-DClang_DIR="$path_to_llvm_build/lib/cmake/" \
	-DLLVM_DIR="$path_to_llvm_build/lib/cmake/"  \
	-DCMAKE_BUILD_TYPE=$build_type \
	-DLLVM_EXTERNAL_LIT="$lit_directory" \
	-DLLVM_ENABLE_WERROR=On           \
	-DCMAKE_CXX_FLAGS="-stdlib=libc++" \
	..
  cmake --build . -j $ncpus
}

# Tests Clad
test_clad () {
  cmake --build . --target check-clad -j $ncpus
}

# Install Clad
install_clad () {
  cmake --build . --target install -j $ncpus
}

# Parse options provided by the user
parse_options () {
eval set -- "$options"

while true
do
    # Consume next (1st) argument
    case "$1" in
	-h|--help)
	    display_help
	    exit 1
	    ;;
	-b|--build_dir)
	    shift
	    export build_directory="$1"
	    ;;
	-i|--install_dir)
	    shift
	    export install_directory="$1"
	    ;;
	-l|--llvm)
	    shift
	    export llvm_version="$1"
	    ;;
	-t|--build_type)
	    type
	    export build_type="$1"
	    ;;
	(--)
	    shift
	    break
      ;;
    esac
    # Fetch next argument as 1st
    shift
done 

}

# Display diagnostic information when building Clad 
display_diagnostic_info () {
    if [[ "${os}" == "Darwin" ]]; then
      echo 'Operating System=MacOS '$os_version
      echo 'Clad will be built with '$ncpus' parallel jobs'
    else
      os_codename="`cat /etc/os-release | grep UBUNTU_CODENAME | cut -d = -f 2`"
      echo 'Operating System= '$os_codename
      echo 'Clad will be built with '$ncpus' parallel jobs'
    fi
}

determine_system_info
install_getopt
# Parse options. Note that options may be followed by one colon to indicate 
# they have a required argument
options=$(getopt -l "help,build_dir:,install_dir:,llvm:,build_type:" -o "hb:i:l:t:" -a -- "$@")
parse_options
display_diagnostic_info
install_compiler
install_other_dependencies
build_clad
test_clad
install_clad
