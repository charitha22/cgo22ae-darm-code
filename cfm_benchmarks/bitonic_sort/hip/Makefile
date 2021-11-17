
include ../../config.mk

# clang
HIPCC := $(ROCM_HOME)/bin/hipcc
HIPCC_FLAGS := -O3 


SRC = bitonic_sort.cpp
HIP_EXECUTABLE = bitonic_sort.hip
CFM_EXECUTABLE = bitonic_sort.cfm

all : $(HIP_EXECUTABLE) $(CFM_EXECUTABLE)

$(CFM_EXECUTABLE) : $(SRC)
	mkdir -p tmp
	$(HIPCC) -### $(HIPCC_FLAGS)  $< -o $@ 2>&1 | python3 $(COMMAND_GEN_SCRIPT) --llvm-home=$(LLVM_HOME) --output-loc=./tmp \
				--pass-loc=$(PASS_LOC) --ptx-opt-level=O0 > ./tmp/compile_command_cfm.sh
	. ./tmp/compile_command_cfm.sh

$(HIP_EXECUTABLE) : $(SRC)
	mkdir -p tmp
	$(HIPCC) -### $(HIPCC_FLAGS)  $< -o $@ 2>&1 | python3 $(COMMAND_GEN_SCRIPT) --llvm-home=$(LLVM_HOME) --output-loc=./tmp \
				--pass-loc=$(PASS_LOC) --ptx-opt-level=O0 --disable-pass > ./tmp/compile_command.sh
	. ./tmp/compile_command.sh


clean :
	rm -rf $(HIP_EXECUTABLE) $(CFM_EXECUTABLE) ./tmp *.o

rocprof_clean:
	rm -f *.db *.csv *.sysinfo.txt *.json ./tmp_metrics/*

