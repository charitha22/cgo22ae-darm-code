; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=x86_64-apple-darwin -mattr=avx512f,avx512bw,avx512vl < %s | FileCheck %s

; Skylake-avx512 target supports masked load/store for i8 and i16 vectors

define <16 x i8> @test_mask_load_16xi8(<16 x i1> %mask, <16 x i8>* %addr, <16 x i8> %val) {
; CHECK-LABEL: test_mask_load_16xi8:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %xmm0, %xmm0
; CHECK-NEXT:    vpmovb2m %xmm0, %k1
; CHECK-NEXT:    vmovdqu8 (%rdi), %xmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %res = call <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>* %addr, i32 4, <16 x i1>%mask, <16 x i8> undef)
  ret <16 x i8> %res
}
declare <16 x i8> @llvm.masked.load.v16i8.p0v16i8(<16 x i8>*, i32, <16 x i1>, <16 x i8>)

define <32 x i8> @test_mask_load_32xi8(<32 x i1> %mask, <32 x i8>* %addr, <32 x i8> %val) {
; CHECK-LABEL: test_mask_load_32xi8:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %ymm0, %ymm0
; CHECK-NEXT:    vpmovb2m %ymm0, %k1
; CHECK-NEXT:    vpblendmb (%rdi), %ymm1, %ymm0 {%k1}
; CHECK-NEXT:    retq
  %res = call <32 x i8> @llvm.masked.load.v32i8.p0v32i8(<32 x i8>* %addr, i32 4, <32 x i1>%mask, <32 x i8> %val)
  ret <32 x i8> %res
}
declare <32 x i8> @llvm.masked.load.v32i8.p0v32i8(<32 x i8>*, i32, <32 x i1>, <32 x i8>)

define <64 x i8> @test_mask_load_64xi8(<64 x i1> %mask, <64 x i8>* %addr, <64 x i8> %val) {
; CHECK-LABEL: test_mask_load_64xi8:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %zmm0, %zmm0
; CHECK-NEXT:    vpmovb2m %zmm0, %k1
; CHECK-NEXT:    vpblendmb (%rdi), %zmm1, %zmm0 {%k1}
; CHECK-NEXT:    retq
  %res = call <64 x i8> @llvm.masked.load.v64i8.p0v64i8(<64 x i8>* %addr, i32 4, <64 x i1>%mask, <64 x i8> %val)
  ret <64 x i8> %res
}
declare <64 x i8> @llvm.masked.load.v64i8.p0v64i8(<64 x i8>*, i32, <64 x i1>, <64 x i8>)

define <8 x i16> @test_mask_load_8xi16(<8 x i1> %mask, <8 x i16>* %addr, <8 x i16> %val) {
; CHECK-LABEL: test_mask_load_8xi16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $15, %xmm0, %xmm0
; CHECK-NEXT:    vpmovw2m %xmm0, %k1
; CHECK-NEXT:    vmovdqu16 (%rdi), %xmm0 {%k1} {z}
; CHECK-NEXT:    retq
  %res = call <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>* %addr, i32 4, <8 x i1>%mask, <8 x i16> undef)
  ret <8 x i16> %res
}
declare <8 x i16> @llvm.masked.load.v8i16.p0v8i16(<8 x i16>*, i32, <8 x i1>, <8 x i16>)

define <16 x i16> @test_mask_load_16xi16(<16 x i1> %mask, <16 x i16>* %addr, <16 x i16> %val) {
; CHECK-LABEL: test_mask_load_16xi16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %xmm0, %xmm0
; CHECK-NEXT:    vpmovb2m %xmm0, %k1
; CHECK-NEXT:    vmovdqu16 (%rdi), %ymm0 {%k1} {z}
; CHECK-NEXT:    retq
  %res = call <16 x i16> @llvm.masked.load.v16i16.p0v16i16(<16 x i16>* %addr, i32 4, <16 x i1>%mask, <16 x i16> zeroinitializer)
  ret <16 x i16> %res
}
declare <16 x i16> @llvm.masked.load.v16i16.p0v16i16(<16 x i16>*, i32, <16 x i1>, <16 x i16>)

define <32 x i16> @test_mask_load_32xi16(<32 x i1> %mask, <32 x i16>* %addr, <32 x i16> %val) {
; CHECK-LABEL: test_mask_load_32xi16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %ymm0, %ymm0
; CHECK-NEXT:    vpmovb2m %ymm0, %k1
; CHECK-NEXT:    vpblendmw (%rdi), %zmm1, %zmm0 {%k1}
; CHECK-NEXT:    retq
  %res = call <32 x i16> @llvm.masked.load.v32i16.p0v32i16(<32 x i16>* %addr, i32 4, <32 x i1>%mask, <32 x i16> %val)
  ret <32 x i16> %res
}
declare <32 x i16> @llvm.masked.load.v32i16.p0v32i16(<32 x i16>*, i32, <32 x i1>, <32 x i16>)

define void @test_mask_store_16xi8(<16 x i1> %mask, <16 x i8>* %addr, <16 x i8> %val) {
; CHECK-LABEL: test_mask_store_16xi8:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %xmm0, %xmm0
; CHECK-NEXT:    vpmovb2m %xmm0, %k1
; CHECK-NEXT:    vmovdqu8 %xmm1, (%rdi) {%k1}
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v16i8.p0v16i8(<16 x i8> %val, <16 x i8>* %addr, i32 4, <16 x i1>%mask)
  ret void
}
declare void @llvm.masked.store.v16i8.p0v16i8(<16 x i8>, <16 x i8>*, i32, <16 x i1>)

define void @test_mask_store_32xi8(<32 x i1> %mask, <32 x i8>* %addr, <32 x i8> %val) {
; CHECK-LABEL: test_mask_store_32xi8:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %ymm0, %ymm0
; CHECK-NEXT:    vpmovb2m %ymm0, %k1
; CHECK-NEXT:    vmovdqu8 %ymm1, (%rdi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v32i8.p0v32i8(<32 x i8> %val, <32 x i8>* %addr, i32 4, <32 x i1>%mask)
  ret void
}
declare void @llvm.masked.store.v32i8.p0v32i8(<32 x i8>, <32 x i8>*, i32, <32 x i1>)

define void @test_mask_store_64xi8(<64 x i1> %mask, <64 x i8>* %addr, <64 x i8> %val) {
; CHECK-LABEL: test_mask_store_64xi8:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %zmm0, %zmm0
; CHECK-NEXT:    vpmovb2m %zmm0, %k1
; CHECK-NEXT:    vmovdqu8 %zmm1, (%rdi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v64i8.p0v64i8(<64 x i8> %val, <64 x i8>* %addr, i32 4, <64 x i1>%mask)
  ret void
}
declare void @llvm.masked.store.v64i8.p0v64i8(<64 x i8>, <64 x i8>*, i32, <64 x i1>)

define void @test_mask_store_8xi16(<8 x i1> %mask, <8 x i16>* %addr, <8 x i16> %val) {
; CHECK-LABEL: test_mask_store_8xi16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $15, %xmm0, %xmm0
; CHECK-NEXT:    vpmovw2m %xmm0, %k1
; CHECK-NEXT:    vmovdqu16 %xmm1, (%rdi) {%k1}
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v8i16.p0v8i16(<8 x i16> %val, <8 x i16>* %addr, i32 4, <8 x i1>%mask)
  ret void
}
declare void @llvm.masked.store.v8i16.p0v8i16(<8 x i16>, <8 x i16>*, i32, <8 x i1>)

define void @test_mask_store_16xi16(<16 x i1> %mask, <16 x i16>* %addr, <16 x i16> %val) {
; CHECK-LABEL: test_mask_store_16xi16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %xmm0, %xmm0
; CHECK-NEXT:    vpmovb2m %xmm0, %k1
; CHECK-NEXT:    vmovdqu16 %ymm1, (%rdi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v16i16.p0v16i16(<16 x i16> %val, <16 x i16>* %addr, i32 4, <16 x i1>%mask)
  ret void
}
declare void @llvm.masked.store.v16i16.p0v16i16(<16 x i16>, <16 x i16>*, i32, <16 x i1>)

define void @test_mask_store_32xi16(<32 x i1> %mask, <32 x i16>* %addr, <32 x i16> %val) {
; CHECK-LABEL: test_mask_store_32xi16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %ymm0, %ymm0
; CHECK-NEXT:    vpmovb2m %ymm0, %k1
; CHECK-NEXT:    vmovdqu16 %zmm1, (%rdi) {%k1}
; CHECK-NEXT:    vzeroupper
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v32i16.p0v32i16(<32 x i16> %val, <32 x i16>* %addr, i32 4, <32 x i1>%mask)
  ret void
}

declare void @llvm.masked.store.v32i16.p0v32i16(<32 x i16>, <32 x i16>*, i32, <32 x i1>)

; Make sure we scalarize masked loads of f16.
define <16 x half> @test_mask_load_16xf16(<16 x i1> %mask, <16 x half>* %addr, <16 x half> %val) {
; CHECK-LABEL: test_mask_load_16xf16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    pushq %rbp
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    pushq %r15
; CHECK-NEXT:    .cfi_def_cfa_offset 24
; CHECK-NEXT:    pushq %r14
; CHECK-NEXT:    .cfi_def_cfa_offset 32
; CHECK-NEXT:    pushq %r13
; CHECK-NEXT:    .cfi_def_cfa_offset 40
; CHECK-NEXT:    pushq %r12
; CHECK-NEXT:    .cfi_def_cfa_offset 48
; CHECK-NEXT:    pushq %rbx
; CHECK-NEXT:    .cfi_def_cfa_offset 56
; CHECK-NEXT:    .cfi_offset %rbx, -56
; CHECK-NEXT:    .cfi_offset %r12, -48
; CHECK-NEXT:    .cfi_offset %r13, -40
; CHECK-NEXT:    .cfi_offset %r14, -32
; CHECK-NEXT:    .cfi_offset %r15, -24
; CHECK-NEXT:    .cfi_offset %rbp, -16
; CHECK-NEXT:    movq %rdi, %rax
; CHECK-NEXT:    vpsllw $7, %xmm0, %xmm0
; CHECK-NEXT:    vpmovmskb %xmm0, %r11d
; CHECK-NEXT:    testb $1, %r11b
; CHECK-NEXT:    je LBB12_1
; CHECK-NEXT:  ## %bb.2: ## %cond.load
; CHECK-NEXT:    movzwl (%rsi), %ecx
; CHECK-NEXT:    movl %ecx, {{[-0-9]+}}(%r{{[sb]}}p) ## 4-byte Spill
; CHECK-NEXT:    jmp LBB12_3
; CHECK-NEXT:  LBB12_1:
; CHECK-NEXT:    movl $0, {{[-0-9]+}}(%r{{[sb]}}p) ## 4-byte Folded Spill
; CHECK-NEXT:  LBB12_3: ## %else
; CHECK-NEXT:    xorl %edi, %edi
; CHECK-NEXT:    movl $0, {{[-0-9]+}}(%r{{[sb]}}p) ## 4-byte Folded Spill
; CHECK-NEXT:    movl %edi, %ecx
; CHECK-NEXT:    testb $2, %r11b
; CHECK-NEXT:    je LBB12_4
; CHECK-NEXT:  ## %bb.5: ## %cond.load1
; CHECK-NEXT:    movw %di, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    movl %edi, %r12d
; CHECK-NEXT:    movl %edi, %ebx
; CHECK-NEXT:    movl %edi, %ebp
; CHECK-NEXT:    movl %edi, %r13d
; CHECK-NEXT:    movl %edi, %r14d
; CHECK-NEXT:    movl %edi, %r8d
; CHECK-NEXT:    movl %edi, %r9d
; CHECK-NEXT:    movl %edi, %r10d
; CHECK-NEXT:    movl %edi, %r15d
; CHECK-NEXT:    movl %edi, %edx
; CHECK-NEXT:    movw %di, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    movw %di, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    movzwl 2(%rsi), %edi
; CHECK-NEXT:    ## kill: def $di killed $di def $edi
; CHECK-NEXT:    testb $4, %r11b
; CHECK-NEXT:    jne LBB12_7
; CHECK-NEXT:    jmp LBB12_8
; CHECK-NEXT:  LBB12_4:
; CHECK-NEXT:    movw %di, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    movl %edi, %r12d
; CHECK-NEXT:    movl %edi, %ebx
; CHECK-NEXT:    movl %edi, %ebp
; CHECK-NEXT:    movl %edi, %r13d
; CHECK-NEXT:    movl %edi, %r14d
; CHECK-NEXT:    movl %edi, %r8d
; CHECK-NEXT:    movl %edi, %r9d
; CHECK-NEXT:    movl %edi, %r10d
; CHECK-NEXT:    movl %edi, %r15d
; CHECK-NEXT:    movl %edi, %edx
; CHECK-NEXT:    movw %di, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    movw %di, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    testb $4, %r11b
; CHECK-NEXT:    je LBB12_8
; CHECK-NEXT:  LBB12_7: ## %cond.load4
; CHECK-NEXT:    movzwl 4(%rsi), %ecx
; CHECK-NEXT:    movw %cx, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:  LBB12_8: ## %else5
; CHECK-NEXT:    testb $8, %r11b
; CHECK-NEXT:    jne LBB12_9
; CHECK-NEXT:  ## %bb.10: ## %else8
; CHECK-NEXT:    testb $16, %r11b
; CHECK-NEXT:    jne LBB12_11
; CHECK-NEXT:  LBB12_12: ## %else11
; CHECK-NEXT:    testb $32, %r11b
; CHECK-NEXT:    jne LBB12_13
; CHECK-NEXT:  LBB12_14: ## %else14
; CHECK-NEXT:    testb $64, %r11b
; CHECK-NEXT:    jne LBB12_15
; CHECK-NEXT:  LBB12_16: ## %else17
; CHECK-NEXT:    testb $-128, %r11b
; CHECK-NEXT:    jne LBB12_17
; CHECK-NEXT:  LBB12_18: ## %else20
; CHECK-NEXT:    testl $256, %r11d ## imm = 0x100
; CHECK-NEXT:    jne LBB12_19
; CHECK-NEXT:  LBB12_20: ## %else23
; CHECK-NEXT:    testl $512, %r11d ## imm = 0x200
; CHECK-NEXT:    jne LBB12_21
; CHECK-NEXT:  LBB12_22: ## %else26
; CHECK-NEXT:    testl $1024, %r11d ## imm = 0x400
; CHECK-NEXT:    jne LBB12_23
; CHECK-NEXT:  LBB12_24: ## %else29
; CHECK-NEXT:    testl $2048, %r11d ## imm = 0x800
; CHECK-NEXT:    jne LBB12_25
; CHECK-NEXT:  LBB12_26: ## %else32
; CHECK-NEXT:    testl $4096, %r11d ## imm = 0x1000
; CHECK-NEXT:    je LBB12_28
; CHECK-NEXT:  LBB12_27: ## %cond.load34
; CHECK-NEXT:    movzwl 24(%rsi), %edx
; CHECK-NEXT:  LBB12_28: ## %else35
; CHECK-NEXT:    movw %dx, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    testl $8192, %r11d ## imm = 0x2000
; CHECK-NEXT:    jne LBB12_29
; CHECK-NEXT:  ## %bb.30: ## %else38
; CHECK-NEXT:    testl $16384, %r11d ## imm = 0x4000
; CHECK-NEXT:    jne LBB12_31
; CHECK-NEXT:  LBB12_32: ## %else41
; CHECK-NEXT:    testl $32768, %r11d ## imm = 0x8000
; CHECK-NEXT:    je LBB12_33
; CHECK-NEXT:  LBB12_34: ## %cond.load43
; CHECK-NEXT:    movzwl {{[-0-9]+}}(%r{{[sb]}}p), %ecx ## 2-byte Folded Reload
; CHECK-NEXT:    movzwl 30(%rsi), %esi
; CHECK-NEXT:    jmp LBB12_35
; CHECK-NEXT:  LBB12_9: ## %cond.load7
; CHECK-NEXT:    movzwl 6(%rsi), %r12d
; CHECK-NEXT:    testb $16, %r11b
; CHECK-NEXT:    je LBB12_12
; CHECK-NEXT:  LBB12_11: ## %cond.load10
; CHECK-NEXT:    movzwl 8(%rsi), %ebx
; CHECK-NEXT:    testb $32, %r11b
; CHECK-NEXT:    je LBB12_14
; CHECK-NEXT:  LBB12_13: ## %cond.load13
; CHECK-NEXT:    movzwl 10(%rsi), %ebp
; CHECK-NEXT:    testb $64, %r11b
; CHECK-NEXT:    je LBB12_16
; CHECK-NEXT:  LBB12_15: ## %cond.load16
; CHECK-NEXT:    movzwl 12(%rsi), %r13d
; CHECK-NEXT:    testb $-128, %r11b
; CHECK-NEXT:    je LBB12_18
; CHECK-NEXT:  LBB12_17: ## %cond.load19
; CHECK-NEXT:    movzwl 14(%rsi), %r14d
; CHECK-NEXT:    testl $256, %r11d ## imm = 0x100
; CHECK-NEXT:    je LBB12_20
; CHECK-NEXT:  LBB12_19: ## %cond.load22
; CHECK-NEXT:    movzwl 16(%rsi), %r8d
; CHECK-NEXT:    testl $512, %r11d ## imm = 0x200
; CHECK-NEXT:    je LBB12_22
; CHECK-NEXT:  LBB12_21: ## %cond.load25
; CHECK-NEXT:    movzwl 18(%rsi), %r9d
; CHECK-NEXT:    testl $1024, %r11d ## imm = 0x400
; CHECK-NEXT:    je LBB12_24
; CHECK-NEXT:  LBB12_23: ## %cond.load28
; CHECK-NEXT:    movzwl 20(%rsi), %r10d
; CHECK-NEXT:    testl $2048, %r11d ## imm = 0x800
; CHECK-NEXT:    je LBB12_26
; CHECK-NEXT:  LBB12_25: ## %cond.load31
; CHECK-NEXT:    movzwl 22(%rsi), %r15d
; CHECK-NEXT:    testl $4096, %r11d ## imm = 0x1000
; CHECK-NEXT:    jne LBB12_27
; CHECK-NEXT:    jmp LBB12_28
; CHECK-NEXT:  LBB12_29: ## %cond.load37
; CHECK-NEXT:    movzwl 26(%rsi), %ecx
; CHECK-NEXT:    movw %cx, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    testl $16384, %r11d ## imm = 0x4000
; CHECK-NEXT:    je LBB12_32
; CHECK-NEXT:  LBB12_31: ## %cond.load40
; CHECK-NEXT:    movzwl 28(%rsi), %ecx
; CHECK-NEXT:    movw %cx, {{[-0-9]+}}(%r{{[sb]}}p) ## 2-byte Spill
; CHECK-NEXT:    testl $32768, %r11d ## imm = 0x8000
; CHECK-NEXT:    jne LBB12_34
; CHECK-NEXT:  LBB12_33:
; CHECK-NEXT:    movzwl {{[-0-9]+}}(%r{{[sb]}}p), %ecx ## 2-byte Folded Reload
; CHECK-NEXT:    movl {{[-0-9]+}}(%r{{[sb]}}p), %esi ## 4-byte Reload
; CHECK-NEXT:  LBB12_35: ## %else44
; CHECK-NEXT:    movl {{[-0-9]+}}(%r{{[sb]}}p), %edx ## 4-byte Reload
; CHECK-NEXT:    movw %dx, (%rax)
; CHECK-NEXT:    movw %di, 2(%rax)
; CHECK-NEXT:    movw %cx, 4(%rax)
; CHECK-NEXT:    movw %r12w, 6(%rax)
; CHECK-NEXT:    movw %bx, 8(%rax)
; CHECK-NEXT:    movw %bp, 10(%rax)
; CHECK-NEXT:    movw %r13w, 12(%rax)
; CHECK-NEXT:    movw %r14w, 14(%rax)
; CHECK-NEXT:    movw %r8w, 16(%rax)
; CHECK-NEXT:    movw %r9w, 18(%rax)
; CHECK-NEXT:    movw %r10w, 20(%rax)
; CHECK-NEXT:    movw %r15w, 22(%rax)
; CHECK-NEXT:    movzwl {{[-0-9]+}}(%r{{[sb]}}p), %ecx ## 2-byte Folded Reload
; CHECK-NEXT:    movw %cx, 24(%rax)
; CHECK-NEXT:    movzwl {{[-0-9]+}}(%r{{[sb]}}p), %ecx ## 2-byte Folded Reload
; CHECK-NEXT:    movw %cx, 26(%rax)
; CHECK-NEXT:    movzwl {{[-0-9]+}}(%r{{[sb]}}p), %ecx ## 2-byte Folded Reload
; CHECK-NEXT:    movw %cx, 28(%rax)
; CHECK-NEXT:    movw %si, 30(%rax)
; CHECK-NEXT:    popq %rbx
; CHECK-NEXT:    popq %r12
; CHECK-NEXT:    popq %r13
; CHECK-NEXT:    popq %r14
; CHECK-NEXT:    popq %r15
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    retq
  %res = call <16 x half> @llvm.masked.load.v16f16(<16 x half>* %addr, i32 4, <16 x i1>%mask, <16 x half> zeroinitializer)
  ret <16 x half> %res
}
declare <16 x half> @llvm.masked.load.v16f16(<16 x half>*, i32, <16 x i1>, <16 x half>)

; Make sure we scalarize masked stores of f16.
define void @test_mask_store_16xf16(<16 x i1> %mask, <16 x half>* %addr, <16 x half> %val) {
; CHECK-LABEL: test_mask_store_16xf16:
; CHECK:       ## %bb.0:
; CHECK-NEXT:    vpsllw $7, %xmm0, %xmm0
; CHECK-NEXT:    vpmovmskb %xmm0, %eax
; CHECK-NEXT:    testb $1, %al
; CHECK-NEXT:    jne LBB13_1
; CHECK-NEXT:  ## %bb.2: ## %else
; CHECK-NEXT:    testb $2, %al
; CHECK-NEXT:    jne LBB13_3
; CHECK-NEXT:  LBB13_4: ## %else2
; CHECK-NEXT:    testb $4, %al
; CHECK-NEXT:    jne LBB13_5
; CHECK-NEXT:  LBB13_6: ## %else4
; CHECK-NEXT:    testb $8, %al
; CHECK-NEXT:    jne LBB13_7
; CHECK-NEXT:  LBB13_8: ## %else6
; CHECK-NEXT:    testb $16, %al
; CHECK-NEXT:    jne LBB13_9
; CHECK-NEXT:  LBB13_10: ## %else8
; CHECK-NEXT:    testb $32, %al
; CHECK-NEXT:    jne LBB13_11
; CHECK-NEXT:  LBB13_12: ## %else10
; CHECK-NEXT:    testb $64, %al
; CHECK-NEXT:    jne LBB13_13
; CHECK-NEXT:  LBB13_14: ## %else12
; CHECK-NEXT:    testb $-128, %al
; CHECK-NEXT:    jne LBB13_15
; CHECK-NEXT:  LBB13_16: ## %else14
; CHECK-NEXT:    testl $256, %eax ## imm = 0x100
; CHECK-NEXT:    jne LBB13_17
; CHECK-NEXT:  LBB13_18: ## %else16
; CHECK-NEXT:    testl $512, %eax ## imm = 0x200
; CHECK-NEXT:    jne LBB13_19
; CHECK-NEXT:  LBB13_20: ## %else18
; CHECK-NEXT:    testl $1024, %eax ## imm = 0x400
; CHECK-NEXT:    jne LBB13_21
; CHECK-NEXT:  LBB13_22: ## %else20
; CHECK-NEXT:    testl $2048, %eax ## imm = 0x800
; CHECK-NEXT:    jne LBB13_23
; CHECK-NEXT:  LBB13_24: ## %else22
; CHECK-NEXT:    testl $4096, %eax ## imm = 0x1000
; CHECK-NEXT:    jne LBB13_25
; CHECK-NEXT:  LBB13_26: ## %else24
; CHECK-NEXT:    testl $8192, %eax ## imm = 0x2000
; CHECK-NEXT:    jne LBB13_27
; CHECK-NEXT:  LBB13_28: ## %else26
; CHECK-NEXT:    testl $16384, %eax ## imm = 0x4000
; CHECK-NEXT:    jne LBB13_29
; CHECK-NEXT:  LBB13_30: ## %else28
; CHECK-NEXT:    testl $32768, %eax ## imm = 0x8000
; CHECK-NEXT:    jne LBB13_31
; CHECK-NEXT:  LBB13_32: ## %else30
; CHECK-NEXT:    retq
; CHECK-NEXT:  LBB13_1: ## %cond.store
; CHECK-NEXT:    movw %si, (%rdi)
; CHECK-NEXT:    testb $2, %al
; CHECK-NEXT:    je LBB13_4
; CHECK-NEXT:  LBB13_3: ## %cond.store1
; CHECK-NEXT:    movw %dx, 2(%rdi)
; CHECK-NEXT:    testb $4, %al
; CHECK-NEXT:    je LBB13_6
; CHECK-NEXT:  LBB13_5: ## %cond.store3
; CHECK-NEXT:    movw %cx, 4(%rdi)
; CHECK-NEXT:    testb $8, %al
; CHECK-NEXT:    je LBB13_8
; CHECK-NEXT:  LBB13_7: ## %cond.store5
; CHECK-NEXT:    movw %r8w, 6(%rdi)
; CHECK-NEXT:    testb $16, %al
; CHECK-NEXT:    je LBB13_10
; CHECK-NEXT:  LBB13_9: ## %cond.store7
; CHECK-NEXT:    movw %r9w, 8(%rdi)
; CHECK-NEXT:    testb $32, %al
; CHECK-NEXT:    je LBB13_12
; CHECK-NEXT:  LBB13_11: ## %cond.store9
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 10(%rdi)
; CHECK-NEXT:    testb $64, %al
; CHECK-NEXT:    je LBB13_14
; CHECK-NEXT:  LBB13_13: ## %cond.store11
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 12(%rdi)
; CHECK-NEXT:    testb $-128, %al
; CHECK-NEXT:    je LBB13_16
; CHECK-NEXT:  LBB13_15: ## %cond.store13
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 14(%rdi)
; CHECK-NEXT:    testl $256, %eax ## imm = 0x100
; CHECK-NEXT:    je LBB13_18
; CHECK-NEXT:  LBB13_17: ## %cond.store15
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 16(%rdi)
; CHECK-NEXT:    testl $512, %eax ## imm = 0x200
; CHECK-NEXT:    je LBB13_20
; CHECK-NEXT:  LBB13_19: ## %cond.store17
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 18(%rdi)
; CHECK-NEXT:    testl $1024, %eax ## imm = 0x400
; CHECK-NEXT:    je LBB13_22
; CHECK-NEXT:  LBB13_21: ## %cond.store19
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 20(%rdi)
; CHECK-NEXT:    testl $2048, %eax ## imm = 0x800
; CHECK-NEXT:    je LBB13_24
; CHECK-NEXT:  LBB13_23: ## %cond.store21
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 22(%rdi)
; CHECK-NEXT:    testl $4096, %eax ## imm = 0x1000
; CHECK-NEXT:    je LBB13_26
; CHECK-NEXT:  LBB13_25: ## %cond.store23
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 24(%rdi)
; CHECK-NEXT:    testl $8192, %eax ## imm = 0x2000
; CHECK-NEXT:    je LBB13_28
; CHECK-NEXT:  LBB13_27: ## %cond.store25
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 26(%rdi)
; CHECK-NEXT:    testl $16384, %eax ## imm = 0x4000
; CHECK-NEXT:    je LBB13_30
; CHECK-NEXT:  LBB13_29: ## %cond.store27
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %ecx
; CHECK-NEXT:    movw %cx, 28(%rdi)
; CHECK-NEXT:    testl $32768, %eax ## imm = 0x8000
; CHECK-NEXT:    je LBB13_32
; CHECK-NEXT:  LBB13_31: ## %cond.store29
; CHECK-NEXT:    movzwl {{[0-9]+}}(%rsp), %eax
; CHECK-NEXT:    movw %ax, 30(%rdi)
; CHECK-NEXT:    retq
  call void @llvm.masked.store.v16f16.p0v16f16(<16 x half> %val, <16 x half>* %addr, i32 4, <16 x i1>%mask)
  ret void
}
declare void @llvm.masked.store.v16f16.p0v16f16(<16 x half>, <16 x half>*, i32, <16 x i1>)
