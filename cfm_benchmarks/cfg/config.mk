LLVM_HOME=/home/cgusthin/workspace/llvm-project-rocm/build/
clang=$(LLVM_HOME)/bin/clang
opt=$(LLVM_HOME)/bin/opt
lli=$(LLVM_HOME)/bin/lli
#pass_location=$(LLVM_HOME)/lib/LLVMCFMelderPlugin.so
cfmerger_flags=-cfmelder  -S -force-cf-merging 
