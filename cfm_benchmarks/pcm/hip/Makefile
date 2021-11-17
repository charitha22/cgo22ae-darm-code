
ROCM_HOME := /opt/rocm/
LLVM_HOME := /home/cgusthin/workspace/llvm-project-rocm/build_install
COMMAND_GEN_SCRIPT := $(LLVM_HOME)/../scripts/gen_compile_command.py

# clang
HIPCC := $(ROCM_HOME)/bin/hipcc
HIPCC_FLAGS := -O3 


SRC = pcm.hip.cpp
HIP_EXECUTABLE = pcm.hipcc
CFM_EXECUTABLE = pcm.cfm

all : $(HIP_EXECUTABLE) $(CFM_EXECUTABLE)

$(CFM_EXECUTABLE) : $(SRC)
	mkdir -p tmp
	$(HIPCC) -### $(HIPCC_FLAGS)  $< -o $@ 2>&1 | python3 $(COMMAND_GEN_SCRIPT) --llvm-home=$(LLVM_HOME) --output-loc=./tmp \
				--pass-loc=$(LLVM_HOME)/../build/lib/LLVMCFMerger.so --ptx-opt-level=O0 > ./tmp/compile_command_cfm.sh
	. ./tmp/compile_command_cfm.sh

$(HIP_EXECUTABLE) : $(SRC)
	mkdir -p tmp
	$(HIPCC) -### $(HIPCC_FLAGS)  $< -o $@ 2>&1 | python3 $(COMMAND_GEN_SCRIPT) --llvm-home=$(LLVM_HOME) --output-loc=./tmp \
				--pass-loc=$(LLVM_HOME)/../build/lib/LLVMCFMerger.so --ptx-opt-level=O0 --disable-pass > ./tmp/compile_command.sh
	. ./tmp/compile_command.sh


clean :
	rm -rf $(HIP_EXECUTABLE) $(CFM_EXECUTABLE) ./tmp *.o

rocprof_clean:
	rm -f *.db *.csv *.sysinfo.txt *.json ./tmp_metrics/*

