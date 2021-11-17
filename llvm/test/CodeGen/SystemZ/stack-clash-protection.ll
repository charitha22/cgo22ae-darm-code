; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -O3 | FileCheck %s
;
; Test stack clash protection probing for static allocas.

; Small: one probe.
define i32 @fun0() #0 {
; CHECK-LABEL: fun0:
; CHECK:       # %bb.0:
; CHECK-NEXT:    aghi %r15, -560
; CHECK-NEXT:    .cfi_def_cfa_offset 720
; CHECK-NEXT:    cg %r0, 552(%r15)
; CHECK-NEXT:    mvhi 552(%r15), 1
; CHECK-NEXT:    l %r2, 160(%r15)
; CHECK-NEXT:    aghi %r15, 560
; CHECK-NEXT:    br %r14

  %a = alloca i32, i64 100
  %b = getelementptr inbounds i32, i32* %a, i64 98
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; Medium: two probes.
define i32 @fun1() #0 {
; CHECK-LABEL: fun1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    aghi %r15, -4096
; CHECK-NEXT:    .cfi_def_cfa_offset 4256
; CHECK-NEXT:    cg %r0, 4088(%r15)
; CHECK-NEXT:    aghi %r15, -4080
; CHECK-NEXT:    .cfi_def_cfa_offset 8336
; CHECK-NEXT:    cg %r0, 4072(%r15)
; CHECK-NEXT:    mvhi 976(%r15), 1
; CHECK-NEXT:    l %r2, 176(%r15)
; CHECK-NEXT:    aghi %r15, 8176
; CHECK-NEXT:    br %r14

  %a = alloca i32, i64 2000
  %b = getelementptr inbounds i32, i32* %a, i64 200
  store volatile i32 1, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; Large: Use a loop to allocate and probe in steps.
define i32 @fun2() #0 {
; CHECK-LABEL: fun2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lgr %r1, %r15
; CHECK-NEXT:    .cfi_def_cfa_register %r1
; CHECK-NEXT:    agfi %r1, -69632
; CHECK-NEXT:    .cfi_def_cfa_offset 69792
; CHECK-NEXT:  .LBB2_1: # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    aghi %r15, -4096
; CHECK-NEXT:    cg %r0, 4088(%r15)
; CHECK-NEXT:    clgrjh %r15, %r1, .LBB2_1
; CHECK-NEXT:  # %bb.2:
; CHECK-NEXT:    .cfi_def_cfa_register %r15
; CHECK-NEXT:    aghi %r15, -2544
; CHECK-NEXT:    .cfi_def_cfa_offset 72336
; CHECK-NEXT:    cg %r0, 2536(%r15)
; CHECK-NEXT:    lhi %r0, 1
; CHECK-NEXT:    mvhi 568(%r15), 1
; CHECK-NEXT:    sty %r0, 28968(%r15)
; CHECK-NEXT:    l %r2, 176(%r15)
; CHECK-NEXT:    agfi %r15, 72176
; CHECK-NEXT:    br %r14

  %a = alloca i32, i64 18000
  %b0 = getelementptr inbounds i32, i32* %a, i64 98
  %b1 = getelementptr inbounds i32, i32* %a, i64 7198
  store volatile i32 1, i32* %b0
  store volatile i32 1, i32* %b1
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; Ends evenly on the step so no remainder needed.
define void @fun3() #0 {
; CHECK-LABEL: fun3:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lgr %r1, %r15
; CHECK-NEXT:    .cfi_def_cfa_register %r1
; CHECK-NEXT:    aghi %r1, -28672
; CHECK-NEXT:    .cfi_def_cfa_offset 28832
; CHECK-NEXT:  .LBB3_1: # %entry
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    aghi %r15, -4096
; CHECK-NEXT:    cg %r0, 4088(%r15)
; CHECK-NEXT:    clgrjh %r15, %r1, .LBB3_1
; CHECK-NEXT:  # %bb.2: # %entry
; CHECK-NEXT:    .cfi_def_cfa_register %r15
; CHECK-NEXT:    mvhi 180(%r15), 0
; CHECK-NEXT:    l %r0, 180(%r15)
; CHECK-NEXT:    aghi %r15, 28672
; CHECK-NEXT:    br %r14
entry:
  %stack = alloca [7122 x i32], align 4
  %i = alloca i32, align 4
  %0 = bitcast [7122 x i32]* %stack to i8*
  %i.0.i.0..sroa_cast = bitcast i32* %i to i8*
  store volatile i32 0, i32* %i, align 4
  %i.0.i.0.6 = load volatile i32, i32* %i, align 4
  ret void
}

; Loop with bigger step.
define void @fun4() #0 "stack-probe-size"="8192" {
; CHECK-LABEL: fun4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lgr %r1, %r15
; CHECK-NEXT:    .cfi_def_cfa_register %r1
; CHECK-NEXT:    aghi %r1, -24576
; CHECK-NEXT:    .cfi_def_cfa_offset 24736
; CHECK-NEXT:  .LBB4_1: # %entry
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    aghi %r15, -8192
; CHECK-NEXT:    cg %r0, 8184(%r15)
; CHECK-NEXT:    clgrjh %r15, %r1, .LBB4_1
; CHECK-NEXT:  # %bb.2: # %entry
; CHECK-NEXT:    .cfi_def_cfa_register %r15
; CHECK-NEXT:    aghi %r15, -7608
; CHECK-NEXT:    .cfi_def_cfa_offset 32344
; CHECK-NEXT:    cg %r0, 7600(%r15)
; CHECK-NEXT:    mvhi 180(%r15), 0
; CHECK-NEXT:    l %r0, 180(%r15)
; CHECK-NEXT:    aghi %r15, 32184
; CHECK-NEXT:    br %r14
entry:
  %stack = alloca [8000 x i32], align 4
  %i = alloca i32, align 4
  %0 = bitcast [8000 x i32]* %stack to i8*
  %i.0.i.0..sroa_cast = bitcast i32* %i to i8*
  store volatile i32 0, i32* %i, align 4
  %i.0.i.0.6 = load volatile i32, i32* %i, align 4
  ret void
}

; Probe size should be modulo stack alignment.
define void @fun5() #0 "stack-probe-size"="4100" {
; CHECK-LABEL: fun5:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    aghi %r15, -4096
; CHECK-NEXT:    .cfi_def_cfa_offset 4256
; CHECK-NEXT:    cg %r0, 4088(%r15)
; CHECK-NEXT:    aghi %r15, -88
; CHECK-NEXT:    .cfi_def_cfa_offset 4344
; CHECK-NEXT:    cg %r0, 80(%r15)
; CHECK-NEXT:    mvhi 180(%r15), 0
; CHECK-NEXT:    l %r0, 180(%r15)
; CHECK-NEXT:    aghi %r15, 4184
; CHECK-NEXT:    br %r14
entry:
  %stack = alloca [1000 x i32], align 4
  %i = alloca i32, align 4
  %0 = bitcast [1000 x i32]* %stack to i8*
  %i.0.i.0..sroa_cast = bitcast i32* %i to i8*
  store volatile i32 0, i32* %i, align 4
  %i.0.i.0.6 = load volatile i32, i32* %i, align 4
  ret void
}

; The minimum probe size is the stack alignment.
define void @fun6() #0 "stack-probe-size"="5" {
; CHECK-LABEL: fun6:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lgr %r1, %r15
; CHECK-NEXT:    .cfi_def_cfa_register %r1
; CHECK-NEXT:    aghi %r1, -4184
; CHECK-NEXT:    .cfi_def_cfa_offset 4344
; CHECK-NEXT:  .LBB6_1: # %entry
; CHECK-NEXT:    # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    aghi %r15, -8
; CHECK-NEXT:    cg %r0, 0(%r15)
; CHECK-NEXT:    clgrjh %r15, %r1, .LBB6_1
; CHECK-NEXT:  # %bb.2: # %entry
; CHECK-NEXT:    .cfi_def_cfa_register %r15
; CHECK-NEXT:    mvhi 180(%r15), 0
; CHECK-NEXT:    l %r0, 180(%r15)
; CHECK-NEXT:    aghi %r15, 4184
; CHECK-NEXT:    br %r14
entry:
  %stack = alloca [1000 x i32], align 4
  %i = alloca i32, align 4
  %0 = bitcast [1000 x i32]* %stack to i8*
  %i.0.i.0..sroa_cast = bitcast i32* %i to i8*
  store volatile i32 0, i32* %i, align 4
  %i.0.i.0.6 = load volatile i32, i32* %i, align 4
  ret void
}

; Small with a natural probe (STMG) - needs no extra probe.
define i32 @fun7() #0 {
; CHECK-LABEL: fun7:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stmg %r14, %r15, 112(%r15)
; CHECK-NEXT:    .cfi_offset %r14, -48
; CHECK-NEXT:    .cfi_offset %r15, -40
; CHECK-NEXT:    aghi %r15, -3976
; CHECK-NEXT:    .cfi_def_cfa_offset 4136
; CHECK-NEXT:    brasl %r14, foo@PLT
; CHECK-NEXT:    st %r2, 568(%r15)
; CHECK-NEXT:    l %r2, 176(%r15)
; CHECK-NEXT:    lmg %r14, %r15, 4088(%r15)
; CHECK-NEXT:    br %r14
  %v = call i32 @foo()
  %a = alloca i32, i64 950
  %b = getelementptr inbounds i32, i32* %a, i64 98
  store volatile i32 %v, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

; Medium with an STMG - still needs probing.
define i32 @fun8() #0 {
; CHECK-LABEL: fun8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stmg %r14, %r15, 112(%r15)
; CHECK-NEXT:    .cfi_offset %r14, -48
; CHECK-NEXT:    .cfi_offset %r15, -40
; CHECK-NEXT:    aghi %r15, -3984
; CHECK-NEXT:    .cfi_def_cfa_offset 4144
; CHECK-NEXT:    cg %r0, 3976(%r15)
; CHECK-NEXT:    brasl %r14, foo@PLT
; CHECK-NEXT:    st %r2, 976(%r15)
; CHECK-NEXT:    l %r2, 176(%r15)
; CHECK-NEXT:    lmg %r14, %r15, 4096(%r15)
; CHECK-NEXT:    br %r14

  %v = call i32 @foo()
  %a = alloca i32, i64 952
  %b = getelementptr inbounds i32, i32* %a, i64 200
  store volatile i32 %v, i32* %b
  %c = load volatile i32, i32* %a
  ret i32 %c
}

declare i32 @foo()
attributes #0 = {  "probe-stack"="inline-asm"  }

