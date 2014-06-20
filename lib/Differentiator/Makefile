##===------clad/lib/Differentiator/Makefile-----------------*- Makefile -*-===##
# 
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
# 
##===----------------------------------------------------------------------===##
#
#  This implements the a C++ differentiator backend.
#
##===----------------------------------------------------------------------===##

CLAD_LEVEL = ../..
LIBRARYNAME := cladDifferentiator
BUILD_ARCHIVE = 1
CXXFLAGS = -fno-rtti

include $(CLAD_LEVEL)/Makefile

CLAD_GIT_REVISION := $(strip \
	   $(shell $(LLVM_SRC_ROOT)/utils/GetSourceVersion $(PROJ_SRC_DIR)/$(CLAD_LEVEL)))
CLAD_CLANG_COMPAT_REVISION := $(strip \
	$(shell cat $(PROJ_SRC_DIR)/$(CLAD_LEVEL)/LastKnownGoodLLVMRevision.txt))

CPP.Defines += -DCLAD_GIT_REVISION='"$(CLAD_GIT_REVISION)"' \
	-DCLAD_CLANG_COMPAT_REVISION='"$(CLAD_CLANG_COMPAT_REVISION)"'
# Inform the sources about their install location
CPPFLAGS += -I$(PROJ_SRC_DIR)/../../include -I$(PROJ_SRC_DIR)/../../../clang/include \
-I$(PROJ_SRC_DIR)/../../../clang/lib -I$(PROJ_OBJ_DIR)/../../include \
-I$(PROJ_OBJ_DIR)/../../../clang/include \
'-DCLANG_SYS_HEADERS="-I$(PROJ_SRC_DIR)../../../clang/lib/Headers"' \
'-DCLAD_INSTDIR_INCL="$(DESTDIR)$(PROJ_includedir)"' \
'-DCLAD_SRCDIR_INCL="$(LLVM_SRC_ROOT)/tools/clad/include"'\


