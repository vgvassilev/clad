##===- tools/bugpoint-passes/Makefile -- -------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLAD_LEVEL := ..
CLANG_LEVEL := ../../clang
LIBRARYNAME := clad

EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/clad.exports

LINK_LIBS_IN_SHARED = 1
#LOADABLE_MODULE := 1
SHARED_LIBRARY = 1

# Set common clad build flags.
CPP.Flags += -I$(PROJ_SRC_DIR)/$(CLAD_LEVEL)/include \
	 -I$(PROJ_OBJ_DIR)/$(CLAD_LEVEL)/include \
	 -I$(PROJ_SRC_DIR)/$(CLAD_LEVEL)/tools # FIXME: For RequiredSymbols.cpp
ifdef CLAD_VENDOR
CPP.Flags += -DCLAD_VENDOR='"$(CLAD_VENDOR) "'
endif


include $(CLANG_LEVEL)/../../Makefile.config

#LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc

USEDLIBS := cladDifferentiator.a \
	    clangSerialization.a LLVMOption.a clangFrontend.a clangDriver.a \
            clangParse.a clangSema.a clangAST.a clangLex.a clangAnalysis.a \
	    clangEdit.a clangBasic.a



include $(CLANG_LEVEL)/Makefile
