; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -dce -instcombine -S | FileCheck %s
; RUN: opt < %s -loop-vectorize -disable-basic-aa -S -pass-remarks-analysis='loop-vectorize' 2>&1 | FileCheck %s -check-prefix=FORCED_OPTSIZE

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we vectorize this loop:
; int foo(float *a, float *b, int n) {
;   for (int i=0; i<n; ++i)
;     a[i] = b[i] * 3;
; }

define i32 @foo(float* nocapture %a, float* nocapture %b, i32 %n) nounwind uwtable ssp {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP6:%.*]] = icmp sgt i32 [[N:%.*]], 0, !dbg !4
; CHECK-NEXT:    br i1 [[CMP6]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]], !dbg !4
; CHECK:       for.body.preheader:
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[N]], -1, !dbg !9
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64, !dbg !9
; CHECK-NEXT:    [[TMP2:%.*]] = add nuw nsw i64 [[TMP1]], 1, !dbg !9
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i32 [[TMP0]], 3, !dbg !9
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]], !dbg !9
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP3:%.*]] = add i32 [[N]], -1, !dbg !9
; CHECK-NEXT:    [[TMP4:%.*]] = zext i32 [[TMP3]] to i64, !dbg !9
; CHECK-NEXT:    [[TMP5:%.*]] = add nuw nsw i64 [[TMP4]], 1, !dbg !9
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr float, float* [[A:%.*]], i64 [[TMP5]], !dbg !9
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr float, float* [[B:%.*]], i64 [[TMP5]], !dbg !9
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ugt float* [[SCEVGEP4]], [[A]], !dbg !9
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ugt float* [[SCEVGEP]], [[B]], !dbg !9
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]], !dbg !9
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]], !dbg !9
; CHECK:       vector.ph:
; CHECK-NEXT:    [[TMP6:%.*]] = and i32 [[N]], 3, !dbg !9
; CHECK-NEXT:    [[N_MOD_VF:%.*]] = zext i32 [[TMP6]] to i64, !dbg !9
; CHECK-NEXT:    [[N_VEC:%.*]] = sub nsw i64 [[TMP2]], [[N_MOD_VF]], !dbg !9
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]], !dbg !9
; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, [[VECTOR_PH]] ], [ [[INDEX_NEXT:%.*]], [[VECTOR_BODY]] ], !dbg !9
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[INDEX]], !dbg !9
; CHECK-NEXT:    [[TMP8:%.*]] = bitcast float* [[TMP7]] to <4 x float>*, !dbg !9
; CHECK-NEXT:    [[WIDE_LOAD:%.*]] = load <4 x float>, <4 x float>* [[TMP8]], align 4, !dbg !9, !alias.scope !10
; CHECK-NEXT:    [[TMP9:%.*]] = fmul <4 x float> [[WIDE_LOAD]], <float 3.000000e+00, float 3.000000e+00, float 3.000000e+00, float 3.000000e+00>, !dbg !9
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDEX]], !dbg !9
; CHECK-NEXT:    [[TMP11:%.*]] = bitcast float* [[TMP10]] to <4 x float>*, !dbg !9
; CHECK-NEXT:    store <4 x float> [[TMP9]], <4 x float>* [[TMP11]], align 4, !dbg !9, !alias.scope !13, !noalias !10
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4, !dbg !9
; CHECK-NEXT:    [[TMP12:%.*]] = icmp eq i64 [[INDEX_NEXT]], [[N_VEC]], !dbg !9
; CHECK-NEXT:    br i1 [[TMP12]], label [[MIDDLE_BLOCK:%.*]], label [[VECTOR_BODY]], !dbg !9, !llvm.loop !15
; CHECK:       middle.block:
; CHECK-NEXT:    [[CMP_N:%.*]] = icmp eq i32 [[TMP6]], 0, !dbg !9
; CHECK-NEXT:    br i1 [[CMP_N]], label [[FOR_END_LOOPEXIT:%.*]], label [[SCALAR_PH]], !dbg !9
; CHECK:       scalar.ph:
; CHECK-NEXT:    [[BC_RESUME_VAL:%.*]] = phi i64 [ [[N_VEC]], [[MIDDLE_BLOCK]] ], [ 0, [[FOR_BODY_PREHEADER]] ], [ 0, [[VECTOR_MEMCHECK]] ], !dbg !9
; CHECK-NEXT:    br label [[FOR_BODY:%.*]], !dbg !9
; CHECK:       for.body:
; CHECK-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ], [ [[BC_RESUME_VAL]], [[SCALAR_PH]] ], !dbg !9
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[B]], i64 [[INDVARS_IV]], !dbg !9
; CHECK-NEXT:    [[TMP13:%.*]] = load float, float* [[ARRAYIDX]], align 4, !dbg !9
; CHECK-NEXT:    [[MUL:%.*]] = fmul float [[TMP13]], 3.000000e+00, !dbg !9
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, float* [[A]], i64 [[INDVARS_IV]], !dbg !9
; CHECK-NEXT:    store float [[MUL]], float* [[ARRAYIDX2]], align 4, !dbg !9
; CHECK-NEXT:    [[INDVARS_IV_NEXT]] = add i64 [[INDVARS_IV]], 1, !dbg !9
; CHECK-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32, !dbg !9
; CHECK-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[N]], !dbg !9
; CHECK-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT]], label [[FOR_BODY]], !dbg !9, !llvm.loop !17
; CHECK:       for.end.loopexit:
; CHECK-NEXT:    br label [[FOR_END]], !dbg !18
; CHECK:       for.end:
; CHECK-NEXT:    ret i32 undef, !dbg !18
;
; FORCED_OPTSIZE-LABEL: @foo(
; FORCED_OPTSIZE-NEXT:  entry:
; FORCED_OPTSIZE-NEXT:    [[CMP6:%.*]] = icmp sgt i32 [[N:%.*]], 0, !dbg !4
; FORCED_OPTSIZE-NEXT:    br i1 [[CMP6]], label [[FOR_BODY_PREHEADER:%.*]], label [[FOR_END:%.*]], !dbg !4
; FORCED_OPTSIZE:       for.body.preheader:
; FORCED_OPTSIZE-NEXT:    br label [[FOR_BODY:%.*]], !dbg !9
; FORCED_OPTSIZE:       for.body:
; FORCED_OPTSIZE-NEXT:    [[INDVARS_IV:%.*]] = phi i64 [ [[INDVARS_IV_NEXT:%.*]], [[FOR_BODY]] ], [ 0, [[FOR_BODY_PREHEADER]] ], !dbg !9
; FORCED_OPTSIZE-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds float, float* [[B:%.*]], i64 [[INDVARS_IV]], !dbg !9
; FORCED_OPTSIZE-NEXT:    [[TMP0:%.*]] = load float, float* [[ARRAYIDX]], align 4, !dbg !9
; FORCED_OPTSIZE-NEXT:    [[MUL:%.*]] = fmul float [[TMP0]], 3.000000e+00, !dbg !9
; FORCED_OPTSIZE-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds float, float* [[A:%.*]], i64 [[INDVARS_IV]], !dbg !9
; FORCED_OPTSIZE-NEXT:    store float [[MUL]], float* [[ARRAYIDX2]], align 4, !dbg !9
; FORCED_OPTSIZE-NEXT:    [[INDVARS_IV_NEXT]] = add i64 [[INDVARS_IV]], 1, !dbg !9
; FORCED_OPTSIZE-NEXT:    [[LFTR_WIDEIV:%.*]] = trunc i64 [[INDVARS_IV_NEXT]] to i32, !dbg !9
; FORCED_OPTSIZE-NEXT:    [[EXITCOND:%.*]] = icmp eq i32 [[LFTR_WIDEIV]], [[N]], !dbg !9
; FORCED_OPTSIZE-NEXT:    br i1 [[EXITCOND]], label [[FOR_END_LOOPEXIT:%.*]], label [[FOR_BODY]], !dbg !9
; FORCED_OPTSIZE:       for.end.loopexit:
; FORCED_OPTSIZE-NEXT:    br label [[FOR_END]], !dbg !10
; FORCED_OPTSIZE:       for.end:
; FORCED_OPTSIZE-NEXT:    ret i32 undef, !dbg !10
;
entry:
  %cmp6 = icmp sgt i32 %n, 0, !dbg !6
  br i1 %cmp6, label %for.body, label %for.end, !dbg !6

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ], !dbg !7
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv, !dbg !7
  %0 = load float, float* %arrayidx, align 4, !dbg !7
  %mul = fmul float %0, 3.000000e+00, !dbg !7
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvars.iv, !dbg !7
  store float %mul, float* %arrayidx2, align 4, !dbg !7
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !7
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32, !dbg !7
  %exitcond = icmp eq i32 %lftr.wideiv, %n, !dbg !7
  br i1 %exitcond, label %for.end, label %for.body, !dbg !7

for.end:                                          ; preds = %for.body, %entry
  ret i32 undef, !dbg !8
}

; Make sure that we try to vectorize loops with a runtime check if the
; dependency check fails.

; CHECK-LABEL: test_runtime_check
; CHECK:      <4 x float>
define void @test_runtime_check(float* %a, float %b, i64 %offset, i64 %offset2, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ind.sum = add i64 %iv, %offset
  %arr.idx = getelementptr inbounds float, float* %a, i64 %ind.sum
  %l1 = load float, float* %arr.idx, align 4
  %ind.sum2 = add i64 %iv, %offset2
  %arr.idx2 = getelementptr inbounds float, float* %a, i64 %ind.sum2
  %l2 = load float, float* %arr.idx2, align 4
  %m = fmul fast float %b, %l2
  %ad = fadd fast float %l1, %m
  store float %ad, float* %arr.idx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %loopexit, label %for.body

loopexit:
  ret void
}

; Check we do not generate runtime checks if we found a known dependence preventing
; vectorization. In this case, it is a read of c[i-1] followed by a write of c[i].
; The runtime checks would always fail.

; void test_runtime_check2(float *a, float b, unsigned offset, unsigned offset2, unsigned n, float *c) {
;   for (unsigned i = 1; i < n; i++) {
;     a[i+o1] += a[i+o2] + b;
;     c[i] = c[i-1] + b;
;   }
; }
;
; CHECK-LABEL: test_runtime_check2
; CHECK-NOT:      <4 x float>
define void @test_runtime_check2(float* %a, float %b, i64 %offset, i64 %offset2, i64 %n, float* %c) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %ind.sum = add i64 %iv, %offset
  %arr.idx = getelementptr inbounds float, float* %a, i64 %ind.sum
  %l1 = load float, float* %arr.idx, align 4
  %ind.sum2 = add i64 %iv, %offset2
  %arr.idx2 = getelementptr inbounds float, float* %a, i64 %ind.sum2
  %l2 = load float, float* %arr.idx2, align 4
  %m = fmul fast float %b, %l2
  %ad = fadd fast float %l1, %m
  store float %ad, float* %arr.idx, align 4
  %c.ind = add i64 %iv, -1
  %c.idx = getelementptr inbounds float, float* %c, i64 %c.ind
  %lc = load float, float* %c.idx, align 4
  %vc = fadd float %lc, 1.0
  %c.idx2 = getelementptr inbounds float, float* %c, i64 %iv
  store float %vc, float* %c.idx2
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %loopexit, label %for.body

loopexit:
  ret void
}

; CHECK: !9 = !DILocation(line: 101, column: 1, scope: !{{.*}})

define dso_local void @forced_optsize(i64* noalias nocapture readonly %x_p, i64* noalias nocapture readonly %y_p, i64* noalias nocapture %z_p) minsize optsize {
;
; xFORCED_OPTSIZE: remark: <unknown>:0:0: Code-size may be reduced by not forcing vectorization, or by source-code modifications eliminating the need for runtime checks (e.g., adding 'restrict').
; FORCED_OPTSIZE-LABEL: @forced_optsize(
; FORCED_OPTSIZE:       vector.body:
;
entry:
  br label %for.body

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %x_p, i64 %indvars.iv
  %0 = load i64, i64* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds i64, i64* %y_p, i64 %indvars.iv
  %1 = load i64, i64* %arrayidx2, align 8
  %add = add nsw i64 %1, %0
  %arrayidx4 = getelementptr inbounds i64, i64* %z_p, i64 %indvars.iv
  store i64 %add, i64* %arrayidx4, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 128
  br i1 %exitcond, label %for.cond.cleanup, label %for.body, !llvm.loop !12
}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!9}
!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}

!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = !DIFile(filename: "test.cpp", directory: "/tmp")
!5 = distinct !DISubprogram(name: "foo", scope: !4, file: !4, line: 99, type: !3, isLocal: false, isDefinition: true, scopeLine: 100, flags: DIFlagPrototyped, isOptimized: false, unit: !9, retainedNodes: !2)
!6 = !DILocation(line: 100, column: 1, scope: !5)
!7 = !DILocation(line: 101, column: 1, scope: !5)
!8 = !DILocation(line: 102, column: 1, scope: !5)
!9 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang",
                             file: !10,
                             isOptimized: true, flags: "-O2",
                             splitDebugFilename: "abc.debug", emissionKind: 2)
!10 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !{!12, !13, !14}
!13 = !{!"llvm.loop.vectorize.width", i32 2}
!14 = !{!"llvm.loop.vectorize.enable", i1 true}
