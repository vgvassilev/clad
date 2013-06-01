##===- tools/bugpoint-passes/Makefile -- -------------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===----------------------------------------------------------------------===##

CLANG_LEVEL := ../../clang
LIBRARYNAME := AutoDiff

EXPORTED_SYMBOL_FILE = $(PROJ_SRC_DIR)/AutoDiff.exports

LINK_LIBS_IN_SHARED = 1
#LOADABLE_MODULE := 1
SHARED_LIBRARY = 1

include $(CLANG_LEVEL)/../../Makefile.config

LINK_COMPONENTS := $(TARGETS_TO_BUILD) asmparser bitreader support mc

USEDLIBS := autodiffDifferentiator.a \
	    clangSerialization.a clangFrontend.a clangDriver.a clangParse.a \
	    clangSema.a clangAST.a clangLex.a clangAnalysis.a clangEdit.a \
	    clangBasic.a



include $(CLANG_LEVEL)/Makefile
