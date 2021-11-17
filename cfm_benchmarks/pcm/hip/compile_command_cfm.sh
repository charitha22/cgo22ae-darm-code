 /opt/rocm-4.0.0/llvm/bin/clang-12 -cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -S -emit-llvm --mrelax-relocations -disable-free -disable-llvm-verifier -discard-value-names -main-file-name pcm.hip.cpp -mrelocation-model pic -pic-level 1 -mframe-pointer=none -fno-rounding-math -mconstructor-aliases -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_isa_version_900.bc -target-cpu gfx900 -fno-split-dwarf-inlining -debugger-tuning=gdb -resource-dir /opt/rocm-4.0.0/llvm/lib/clang/12.0.0 -internal-isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include/cuda_wrappers -internal-isystem /opt/rocm/include -include __clang_hip_runtime_wrapper.h -isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include/.. -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hip/include -D __HIP_ROCclr__ -D __HIP_ROCclr__ -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/local/include -internal-isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -O0 -disable-O0-optnone -std=c++11 -fdeprecated-macro -fno-autolink -fdebug-compilation-dir /home/cgusthin/workspace/llvm-project-rocm/tests/pcm/hip -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -vectorize-loops -fno-unroll-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -fcuda-allow-variadic-functions -faddrsig -o ./tmp/pcm-gfx900-eecc67.ll -x hip pcm.hip.cpp
/home/cgusthin/workspace/llvm-project-rocm/build_install/bin/opt -load /home/cgusthin/workspace/llvm-project-rocm/build_install/../build/lib/LLVMCFMerger.so -gvn -mem2reg -cfmerger -cfmerger -cfmerger -cfmerger -cfmerger  -cfmerger -cfmerger -cfmerger -simplifycfg -S < ./tmp/pcm-gfx900-eecc67.ll  > ./tmp/after_pass.ll
/home/cgusthin/workspace/llvm-project-rocm/build_install/bin/llc  -O3  -mtriple amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -mattr=-code-object-v3  -amdgpu-function-calls=0 ./tmp/after_pass.ll -o ./tmp/pcm-gfx900-eecc67.o 
 /opt/rocm/llvm/bin/lld -flavor gnu --no-undefined -shared -plugin-opt=-amdgpu-internalize-symbols -plugin-opt=mcpu=gfx900 -plugin-opt=O3 -plugin-opt=-amdgpu-early-inline-all=true -plugin-opt=-amdgpu-function-calls=false -o ./tmp/pcm-gfx900-604e99.out ./tmp/pcm-gfx900-eecc67.o

 /opt/rocm/llvm/bin/clang-offload-bundler -type=o -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx900 -inputs=/dev/null,./tmp/pcm-gfx900-604e99.out -outputs=./tmp/pcm-c332ef.hipfb

 /opt/rocm-4.0.0/llvm/bin/clang-12 -cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -emit-obj --mrelax-relocations -disable-free -disable-llvm-verifier -discard-value-names -main-file-name pcm.hip.cpp -mrelocation-model static -mframe-pointer=none -fmath-errno -fno-rounding-math -mconstructor-aliases -munwind-tables -target-cpu x86-64 -fno-split-dwarf-inlining -debugger-tuning=gdb -resource-dir /opt/rocm-4.0.0/llvm/lib/clang/12.0.0 -internal-isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include/cuda_wrappers -internal-isystem /opt/rocm/include -include __clang_hip_runtime_wrapper.h -isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include/.. -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hip/include -D __HIP_ROCclr__ -D __HIP_ROCclr__ -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/local/include -internal-isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /opt/rocm-4.0.0/llvm/lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -O3 -std=c++11 -fdeprecated-macro -fdebug-compilation-dir /home/cgusthin/workspace/llvm-project-rocm/tests/pcm/hip -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -fcuda-include-gpubinary ./tmp/pcm-c332ef.hipfb -fcuda-allow-variadic-functions -faddrsig -o ./tmp/pcm-945815.o -x hip pcm.hip.cpp

 /usr/bin/ld -z relro --hash-style=gnu --eh-frame-hdr -m elf_x86_64 -dynamic-linker /lib64/ld-linux-x86-64.so.2 -o pcm.cfm /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../x86_64-linux-gnu/crt1.o /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../x86_64-linux-gnu/crti.o /usr/lib/gcc/x86_64-linux-gnu/7.5.0/crtbegin.o -L/opt/rocm/hip/lib -L/opt/rocm/llvm/bin/../lib/clang/12.0.0/lib/linux -L/usr/lib/gcc/x86_64-linux-gnu/7.5.0 -L/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../x86_64-linux-gnu -L/lib/x86_64-linux-gnu -L/lib/../lib64 -L/usr/lib/x86_64-linux-gnu -L/usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../.. -L/opt/rocm-4.0.0/llvm/bin/../lib -L/lib -L/usr/lib -lgcc_s -lgcc -lpthread -lm ./tmp/pcm-945815.o --enable-new-dtags --rpath=/opt/rocm/hip/lib:/opt/rocm/lib -lamdhip64 -lclang_rt.builtins-x86_64 -lstdc++ -lm -lgcc_s -lgcc -lc -lgcc_s -lgcc /usr/lib/gcc/x86_64-linux-gnu/7.5.0/crtend.o /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../x86_64-linux-gnu/crtn.o
