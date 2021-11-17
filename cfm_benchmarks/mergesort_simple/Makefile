include ../config.mk

# clang
HIP_CC = $(ROCM_HOME)/bin/hipcc
HIP_FLAGS = -O3 


SRC = mergesort.cpp
HIP_EXECUTABLE = mergesort.hipcc
CFM_EXECUTABLE = mergesort.cfm

all : $(HIP_EXECUTABLE) $(CFM_EXECUTABLE)

$(CFM_EXECUTABLE) : $(SRC)
	mkdir -p tmp
	$(HIP_CC) -### $(HIP_FLAGS)  $^ -o $@ 2>&1 | python3 $(COMMAND_GEN_SCRIPT) --llvm-home=$(LLVM_HOME) --output-loc=./tmp \
				--pass-loc=$(PASS_LOC) --ptx-opt-level=O0  > ./tmp/compile_command_cfm.sh
	. ./tmp/compile_command_cfm.sh

$(HIP_EXECUTABLE) : $(SRC)

	mkdir -p tmp
	$(HIP_CC) -### $(HIP_FLAGS)  $^ -o $@ 2>&1 | python3 $(COMMAND_GEN_SCRIPT) --llvm-home=$(LLVM_HOME) --output-loc=./tmp \
				--pass-loc=$(PASS_LOC) --ptx-opt-level=O0 --disable-pass > ./tmp/compile_command.sh
	. ./tmp/compile_command.sh


rocprof_clean :
	rm -f *.csv *.db *.txt *.json

clean :
	rm -rf $(HIP_EXECUTABLE) $(CFM_EXECUTABLE) ./tmp *.o *.csv *.db *.txt *.json